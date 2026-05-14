#!/usr/bin/env python3
"""
Fish Audio S2 Pro - cinEZma Edition
Auto-pulled from GitHub by the Kaggle launcher notebook.
Edit this file and push to GitHub; changes take effect on next Kaggle restart.
"""

# ── Model ─────────────────────────────────────────────────────────────────────
import glob
import logging
import mimetypes
import os
from pathlib import Path
import tempfile
import time
import wave

import numpy as np
import requests

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logging.getLogger("urllib3").setLevel(logging.WARNING)

FISH_MODEL = os.environ.get("FISH_MODEL") or os.environ.get("FISH_AUDIO_MODEL") or "s2-pro"
FISH_TTS_URL = os.environ.get("FISH_TTS_URL", "https://api.fish.audio/v1/tts")
FISH_ASR_URL = os.environ.get("FISH_ASR_URL", "https://api.fish.audio/v1/asr")
FISH_TIMEOUT = int(os.environ.get("FISH_TIMEOUT", "180"))
FISH_SAMPLE_RATE = int(os.environ.get("FISH_SAMPLE_RATE", "44100"))
FISH_FORMAT = "wav"
FISH_LATENCY = os.environ.get("FISH_LATENCY", "normal")
FISH_DEFAULT_REFERENCE_ID = os.environ.get("FISH_REFERENCE_ID", "").strip()

VOICES_DIR = "/kaggle/working/voice-assets/voices"
OUTPUT_DIR = os.environ.get("FISH_OUTPUT_DIR", "/kaggle/working/fish-audio-output")

_API_KEY_NAMES = ("FISH_API_KEY", "FISH_AUDIO_API_KEY")


def _get_fish_api_key():
    for name in _API_KEY_NAMES:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()

    try:
        from kaggle_secrets import UserSecretsClient

        secrets = UserSecretsClient()
        for name in _API_KEY_NAMES:
            try:
                value = secrets.get_secret(name)
            except Exception:
                value = None
            if value and value.strip():
                return value.strip()
    except Exception:
        pass

    return ""


def _language_code(language):
    if language and language != "Auto":
        return language.split("(")[-1].rstrip(")").strip() if "(" in language else language
    return None


def _clamp_float(value, default, minimum, maximum):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _clamp_int(value, default, minimum, maximum):
    return int(round(_clamp_float(value, default, minimum, maximum)))


def _pack_msgpack(payload):
    try:
        import ormsgpack

        return ormsgpack.packb(payload)
    except ImportError:
        try:
            import msgpack

            return msgpack.packb(payload, use_bin_type=True)
        except ImportError as exc:
            raise RuntimeError(
                "Reference-audio cloning needs MessagePack. In Kaggle, install one "
                "of these before launching: pip install ormsgpack"
            ) from exc


def _fish_error(response):
    try:
        body = response.json()
        detail = body.get("message") or body.get("detail") or body
    except Exception:
        detail = response.text[:1000]
    return f"Fish Audio API {response.status_code}: {detail}"


def _fish_headers(api_key, content_type):
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": content_type,
        "model": FISH_MODEL,
    }


def _post_fish_tts(payload, api_key, needs_msgpack):
    if needs_msgpack:
        response = requests.post(
            FISH_TTS_URL,
            data=_pack_msgpack(payload),
            headers=_fish_headers(api_key, "application/msgpack"),
            timeout=FISH_TIMEOUT,
        )
    else:
        response = requests.post(
            FISH_TTS_URL,
            json=payload,
            headers=_fish_headers(api_key, "application/json"),
            timeout=FISH_TIMEOUT,
        )

    if not response.ok:
        raise RuntimeError(_fish_error(response))
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        raise RuntimeError(f"Fish Audio API returned JSON instead of audio: {response.text[:1000]}")
    return response.content


def _guess_audio_mime(audio_path):
    guessed, _ = mimetypes.guess_type(audio_path)
    return guessed or "audio/wav"


def _transcribe_reference_audio(audio_path, lang_code, api_key):
    data = {"ignore_timestamps": "true"}
    if lang_code:
        data["language"] = lang_code

    with open(audio_path, "rb") as audio_file:
        response = requests.post(
            FISH_ASR_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            data=data,
            files={"audio": (Path(audio_path).name, audio_file, _guess_audio_mime(audio_path))},
            timeout=FISH_TIMEOUT,
        )

    if not response.ok:
        raise RuntimeError(_fish_error(response))

    transcript = response.json().get("text", "").strip()
    if not transcript:
        raise RuntimeError("Fish ASR returned an empty transcript. Please type the reference text.")
    return transcript


def _load_sidecar_transcript(audio_path):
    base = Path(audio_path)
    for suffix in (".txt", ".lab", ".transcript.txt"):
        candidate = base.with_suffix(suffix)
        if candidate.exists():
            text = candidate.read_text(encoding="utf-8").strip()
            if text:
                return text
    return ""


def _prepare_reference(audio_path, ref_text, lang_code, api_key):
    if not audio_path:
        raise RuntimeError("Please provide a reference audio.")
    if not os.path.exists(audio_path):
        raise RuntimeError(f"Reference audio not found: {audio_path}")

    transcript = (ref_text or "").strip()
    transcript_source = "typed transcript"
    if not transcript:
        transcript = _load_sidecar_transcript(audio_path)
        transcript_source = "sidecar transcript"
    if not transcript:
        transcript = _transcribe_reference_audio(audio_path, lang_code, api_key)
        transcript_source = "Fish ASR transcript"

    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    return [{"audio": audio_bytes, "text": transcript}], transcript_source


def _apply_design_tag(text, instruct):
    if instruct and instruct.strip():
        return f"[{instruct.strip()}] {text.strip()}"
    return text.strip()


def _write_audio_file(audio_bytes):
    out_dir = OUTPUT_DIR
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError:
        out_dir = tempfile.gettempdir()

    out_path = os.path.join(out_dir, f"fish_s2_pro_{int(time.time() * 1000)}.wav")
    with open(out_path, "wb") as output:
        output.write(audio_bytes)
    return out_path


def _wav_duration(audio_path):
    try:
        with wave.open(audio_path, "rb") as wav_file:
            sr = wav_file.getframerate()
            duration = wav_file.getnframes() / float(sr)
        return duration, sr
    except Exception:
        return None, FISH_SAMPLE_RATE

def get_characters():
    wavs = glob.glob(os.path.join(VOICES_DIR, "*.wav"))
    return {Path(w).stem: w for w in sorted(wavs)}

print(f"Fish Audio model configured: {FISH_MODEL}")
print(f"Fish Audio TTS endpoint: {FISH_TTS_URL}")
print(f"Preset voices found: {len(get_characters())}")


# ── Gradio UI + Relay ─────────────────────────────────────────────────────────
import gradio as gr

RELAY_URL = os.environ.get(
    "FISH_RELAY_URL",
    os.environ.get("RELAY_URL", "https://omnivoice-relay.zsage84869.workers.dev/register"),
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.gradio-container { max-width: 1000px !important; margin: auto !important; }
.brand-header { text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 28px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 10px 25px rgba(102,126,234,0.3); }
.brand-title { color: white; font-size: 2em; font-weight: 700; margin: 0 0 6px 0; }
.brand-subtitle { color: rgba(255,255,255,0.88); font-size: 1em; margin-bottom: 12px; }
.hint { color: rgba(255,255,255,0.84); font-size: 0.92em; }
button.primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; color: white !important; font-weight: 600 !important; border-radius: 12px !important; }
"""

BRAND_HTML = """
<div class="brand-header">
  <div class="brand-title">Fish Audio S2 Pro Demo</div>
  <div class="brand-subtitle">cinEZma Edition</div>
  <div class="hint">Preset character voices + custom clone + S2 inline control + relay auto-connect</div>
</div>
"""

LANGUAGES = [
    "Auto", "English (en)", "Chinese (zh)", "Japanese (ja)", "Korean (ko)",
    "French (fr)", "German (de)", "Spanish (es)", "Portuguese (pt)",
    "Russian (ru)", "Arabic (ar)", "Hindi (hi)", "Italian (it)",
    "Dutch (nl)", "Turkish (tr)", "Polish (pl)", "Swedish (sv)",
    "Thai (th)", "Vietnamese (vi)", "Indonesian (id)", "Malay (ms)",
]

def lang_dropdown():
    return gr.Dropdown(label="Language (optional)", choices=LANGUAGES, value="Auto")

def gen_settings():
    with gr.Accordion("Advanced Settings", open=False):
        ns = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Temperature")
        gs = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Top P")
        dn = gr.Slider(100, 300, value=200, step=10, label="Chunk Length")
        sp = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
        du = gr.Slider(256, 4096, value=1024, step=64, label="Max New Tokens")
        pp = gr.Checkbox(value=True, label="Normalize Text")
        po = gr.Checkbox(value=True, label="Condition On Previous Chunks")
    return ns, gs, dn, sp, du, pp, po

CATEGORIES = {
    "Gender": ["male", "female"],
    "Age": ["child", "teenager", "young adult", "middle-aged", "elderly"],
    "Pitch": ["very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"],
    "Style": ["whisper", "soft spoken", "excited", "sad", "angry", "dramatic", "broadcast tone"],
    "English Accent": ["american accent", "british accent", "australian accent",
                       "canadian accent", "indian accent", "chinese accent",
                       "japanese accent", "korean accent", "portuguese accent",
                       "russian accent"],
    "Chinese Dialect": ["四川话", "陕西话", "广东话", "东北话", "河南话",
                        "云南话", "贵州话", "桂林话", "济南话"],
    "Texture": ["breathy", "raspy", "clear", "warm", "low energy", "high energy"],
}

ATTR_INFO = {
    "Gender": "Speaker gender",
    "Age": "Approximate speaker age",
    "Pitch": "Voice pitch level",
    "Style": "Speaking style",
    "English Accent": "English accent (effective for English text)",
    "Chinese Dialect": "Chinese dialect (effective for Chinese text)",
    "Texture": "Extra voice texture or delivery direction",
}

def build_instruct(groups):
    selected = [g for g in groups if g and g != "Auto"]
    return ", ".join(selected) if selected else ""

def refresh_characters():
    chars = get_characters()
    choices = list(chars.keys())
    return gr.update(choices=choices, value=choices[0] if choices else None)

def generate_speech(text, language, ref_audio, instruct,
                    temperature, top_p, chunk_length,
                    speed, max_new_tokens, normalize_text, condition_on_previous_chunks,
                    mode="clone", ref_text=None):
    if not text or not text.strip():
        return None, "Please enter some text."

    try:
        api_key = _get_fish_api_key()
        if not api_key:
            return None, "Missing FISH_API_KEY. Add it as a Kaggle secret or environment variable."

        lang_code = _language_code(language)
        payload = {
            "text": _apply_design_tag(text, instruct if mode == "design" else None),
            "format": FISH_FORMAT,
            "sample_rate": FISH_SAMPLE_RATE,
            "temperature": _clamp_float(temperature, 0.7, 0.0, 1.0),
            "top_p": _clamp_float(top_p, 0.7, 0.0, 1.0),
            "chunk_length": _clamp_int(chunk_length, 200, 100, 300),
            "max_new_tokens": _clamp_int(max_new_tokens, 1024, 256, 4096),
            "normalize": bool(normalize_text),
            "condition_on_previous_chunks": bool(condition_on_previous_chunks),
            "latency": FISH_LATENCY,
            "prosody": {
                "speed": _clamp_float(speed, 1.0, 0.5, 2.0),
                "volume": 0,
                "normalize_loudness": True,
            },
        }

        transcript_source = None
        needs_msgpack = False

        if mode == "clone":
            payload["references"], transcript_source = _prepare_reference(
                ref_audio,
                ref_text,
                lang_code,
                api_key,
            )
            needs_msgpack = True
        elif FISH_DEFAULT_REFERENCE_ID:
            payload["reference_id"] = FISH_DEFAULT_REFERENCE_ID

        audio_bytes = _post_fish_tts(payload, api_key, needs_msgpack)
        audio_path = _write_audio_file(audio_bytes)
        duration_seconds, sr = _wav_duration(audio_path)
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"

    if duration_seconds is None:
        status = f"Generated audio with Fish Audio {FISH_MODEL}."
    else:
        status = f"Generated {duration_seconds:.1f}s audio with Fish Audio {FISH_MODEL} at {sr}Hz."
    if transcript_source:
        status += f" Reference used: {transcript_source}."
    return audio_path, status

def generate_preset(text, language, character_name, ref_text, ns, gs, dn, sp, du, pp, po):
    characters = get_characters()
    ref_audio = characters.get(character_name)
    if not ref_audio:
        return None, "No preset voice selected."
    return generate_speech(
        text, language, ref_audio, None, ns, gs, dn, sp, du, pp, po,
        mode="clone", ref_text=ref_text or None
    )

def generate_custom(text, language, ref_audio, ref_text, ns, gs, dn, sp, du, pp, po):
    return generate_speech(
        text, language, ref_audio, None, ns, gs, dn, sp, du, pp, po,
        mode="clone", ref_text=ref_text or None
    )


# ── Forced Alignment (CTC) ────────────────────────────────────────────────────
import json as _json
import os, re, tempfile
import numpy as np
import soundfile as sf

_ALIGNER_MODEL = None

_TARGET_SR = 16000  # MMS_FA requires 16 kHz — mismatched SR is the #1 cause of bad timestamps


def _resample_to_16k(data: np.ndarray, sr: int) -> np.ndarray:
    """Resample audio to 16 kHz using linear interpolation (no extra deps)."""
    if sr == _TARGET_SR:
        return data
    target_len = int(round(len(data) * _TARGET_SR / sr))
    return np.interp(
        np.linspace(0, len(data) - 1, target_len),
        np.arange(len(data)),
        data,
    ).astype(np.float32)


def _detect_silences(data: np.ndarray, sr: int,
                     min_silence_sec: float = 0.15,
                     rms_threshold: float = 0.01,
                     frame_ms: int = 20) -> list:
    """
    Fast RMS-based silence detector.  Returns a list of
    {"start": float, "end": float} dicts for every gap whose RMS
    stays below rms_threshold for at least min_silence_sec.
    """
    frame_len = int(sr * frame_ms / 1000)
    n_frames   = len(data) // frame_len
    silences, in_silence, gap_start = [], False, 0.0

    for i in range(n_frames):
        chunk = data[i * frame_len: (i + 1) * frame_len]
        rms   = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        t     = i * frame_ms / 1000.0

        if rms < rms_threshold:
            if not in_silence:
                in_silence, gap_start = True, t
        else:
            if in_silence:
                in_silence = False
                if t - gap_start >= min_silence_sec:
                    silences.append({"start": round(gap_start, 3),
                                     "end":   round(t, 3)})

    if in_silence:
        t_end = n_frames * frame_ms / 1000.0
        if t_end - gap_start >= min_silence_sec:
            silences.append({"start": round(gap_start, 3),
                             "end":   round(t_end, 3)})
    return silences


def align_japanese(audio_path: str, japanese_text: str, translation_map_json: str = "") -> str:
    """
    CTC forced alignment — returns a JSON object with four keys cinEZma needs:

      romaji_stamps  – list of {text, start, end} for every romaji word
      jp_tokens      – original Japanese surface forms (for verification)
      jp_counts      – "sticky note" list: jp_counts[i] = how many romaji
                       words came from jp_tokens[i]. sum(jp_counts) always
                       equals len(romaji_stamps) (verified before return).
      silence_ranges – list of {start, end} silence gaps detected by RMS scan
      errors         – null, or a short message if something went wrong
    """
    if not audio_path:
        return _json.dumps({"error": "Missing audio file."}, ensure_ascii=False)
    if not japanese_text or not japanese_text.strip():
        return _json.dumps({
            "error": "Missing Japanese Script. Forced alignment needs the exact text spoken in the audio."
        }, ensure_ascii=False)

    try:
        from ctc_forced_aligner import get_word_stamps
        import cutlet
        import fugashi
    except ImportError as e:
        return _json.dumps({"error": f"Missing dependency: {e}"}, ensure_ascii=False)

    try:
        global _ALIGNER_MODEL

        # ── Step 1-2: tokenize JP → romaji, track counts per JP token ──────
        token_pairs, romaji_words = _tokenize_with_index(japanese_text, cutlet, fugashi)
        if not romaji_words:
            return _json.dumps({
                "error": "Japanese Script produced no romanizable tokens."
            }, ensure_ascii=False)

        jp_tokens  = [jp  for jp,  _ in token_pairs]
        jp_counts  = [cnt for _,  cnt in token_pairs]
        romaji_transcript = " ".join(romaji_words)
        print(f"[Align] JP tokens : {jp_tokens}")
        print(f"[Align] jp_counts : {jp_counts}  (sum={sum(jp_counts)}, words={len(romaji_words)})")
        print(f"[Align] Romaji    : {romaji_transcript}")

        # ── Step 3: load + fix audio ────────────────────────────────────────
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_data, raw_sr = sf.read(audio_path)

            # Collapse stereo → mono
            if raw_data.ndim > 1:
                raw_data = raw_data.mean(axis=1)

            # CRITICAL: resample to exactly 16 kHz so timestamps are correct
            audio_16k = _resample_to_16k(raw_data.astype(np.float32), raw_sr)

            # Run silence detection on the 16 kHz mono signal
            silence_ranges = _detect_silences(audio_16k, _TARGET_SR)
            print(f"[Align] Silence gaps : {silence_ranges}")

            wav_path = os.path.join(tmpdir, "input.wav")
            sf.write(wav_path, audio_16k, _TARGET_SR)

            transcript_path = os.path.join(tmpdir, "transcript.txt")
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(romaji_transcript)

            result = get_word_stamps(wav_path, transcript_path,
                                     model=_ALIGNER_MODEL, model_type="MMS_FA")

        if isinstance(result, tuple):
            raw_stamps = result[0]
            if len(result) > 1:
                _ALIGNER_MODEL = result[1]
        else:
            raw_stamps = result

        # ── Step 4: clean stamps ────────────────────────────────────────────
        stamps = []
        for wt in raw_stamps or []:
            if not isinstance(wt, dict):
                continue
            text = wt.get("text") or wt.get("word") or wt.get("label") or ""
            text = re.sub(r"[^A-Za-z']", " ", str(text))
            text = re.sub(r"\s+", " ", text).strip().lower()
            if not text:
                continue
            stamps.append({
                "text":  text,
                "start": round(float(wt.get("start", 0)), 3),
                "end":   round(float(wt.get("end",   0)), 3),
            })

        if not stamps:
            return _json.dumps({
                "error": "Alignment returned no timestamps.",
                "romaji_transcript": romaji_transcript,
            }, ensure_ascii=False)

        # ── Step 5: sanity check sum(jp_counts) == len(stamps) ─────────────
        count_sum = sum(jp_counts)
        count_ok  = count_sum == len(stamps)
        errors    = None if count_ok else (
            f"jp_counts sum ({count_sum}) != romaji stamp count ({len(stamps)}). "
            "The app should fall back to raw stamps."
        )
        if errors:
            print(f"[Align] WARNING: {errors}")

        print(f"[Align] Romaji stamps : {stamps[:5]} ...")
        return _json.dumps({
            "romaji_stamps":  stamps,
            "jp_tokens":      jp_tokens,
            "jp_counts":      jp_counts,
            "silence_ranges": silence_ranges,
            "errors":         errors,
        }, ensure_ascii=False)

    except Exception as e:
        import traceback
        return _json.dumps({
            "error": f"Alignment failed: {e}",
            "trace": traceback.format_exc()[-1200:],
        }, ensure_ascii=False)


def _tokenize_with_index(text: str, cutlet_module, fugashi_module):
    """
    Tokenize Japanese into surface forms using fugashi/MeCab.
    Romanize each token individually with cutlet, keeping the jp<->romaji index.

    Returns:
      token_pairs  -- [(jp_surface, romaji_word_count), ...]
      romaji_words -- flat list of romaji words in the same order
    """
    tagger = fugashi_module.Tagger()
    katsu  = cutlet_module.Cutlet()
    token_pairs  = []
    romaji_words = []

    for word in tagger(text.strip()):
        surface = word.surface
        if not surface.strip():
            continue
        r = katsu.romaji(surface)
        r = re.sub(r"[^A-Za-z']", " ", r)
        r = re.sub(r"\s+", " ", r).strip().lower()
        words = r.split()
        if not words:
            continue  # Punctuation / symbol token — skip
        token_pairs.append((surface, len(words)))
        romaji_words.extend(words)

    return token_pairs, romaji_words




with gr.Blocks(title="Fish Audio S2 Pro Demo") as demo:
    gr.HTML(BRAND_HTML)

    with gr.Tabs():
        with gr.TabItem("Character Voices"):
            with gr.Row():
                with gr.Column(scale=1):
                    characters = get_characters()
                    char_choices = list(characters.keys())
                    preset_name = gr.Dropdown(
                        label="Preset Character",
                        choices=char_choices,
                        value=char_choices[0] if char_choices else None,
                        info="Pulled from your GitHub voices folder",
                    )
                    gr.Button("Refresh Voices", size="sm").click(refresh_characters, outputs=[preset_name])
                    pv_text = gr.Textbox(
                        label="Text to Synthesize",
                        lines=4,
                        placeholder="Enter text here...",
                    )
                    pv_ref_text = gr.Textbox(
                        label="Reference Text (optional)",
                        lines=2,
                        placeholder="Transcript of the preset clip. Leave blank to use sidecar text or Fish ASR.",
                    )
                    pv_lang = lang_dropdown()
                    pv_ns, pv_gs, pv_dn, pv_sp, pv_du, pv_pp, pv_po = gen_settings()
                    pv_btn = gr.Button("Generate", variant="primary", size="lg")
                with gr.Column(scale=1):
                    pv_audio = gr.Audio(label="Output Audio", type="filepath")
                    pv_status = gr.Textbox(label="Status", lines=2)

            pv_btn.click(
                generate_preset,
                inputs=[pv_text, pv_lang, preset_name, pv_ref_text,
                        pv_ns, pv_gs, pv_dn, pv_sp, pv_du, pv_pp, pv_po],
                outputs=[pv_audio, pv_status],
                api_name="generate_preset",
            )

        with gr.TabItem("Custom Voice Clone"):
            with gr.Row():
                with gr.Column(scale=1):
                    vc_text = gr.Textbox(
                        label="Text to Synthesize",
                        lines=4,
                        placeholder="Enter text here... You can use Fish tags like [laughs], [sighs], etc.",
                    )
                    vc_ref_audio = gr.Audio(
                        label="Reference Audio (3-10s recommended)",
                        type="filepath",
                    )
                    vc_ref_text = gr.Textbox(
                        label="Reference Text (optional)",
                        lines=2,
                        placeholder="Transcript of ref audio. Leave blank to use Fish ASR.",
                    )
                    vc_lang = lang_dropdown()
                    vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po = gen_settings()
                    vc_btn = gr.Button("Generate", variant="primary", size="lg")
                with gr.Column(scale=1):
                    vc_audio = gr.Audio(label="Output Audio", type="filepath")
                    vc_status = gr.Textbox(label="Status", lines=2)

            vc_btn.click(
                generate_custom,
                inputs=[vc_text, vc_lang, vc_ref_audio, vc_ref_text,
                        vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po],
                outputs=[vc_audio, vc_status],
                api_name="generate_custom",
            )

        with gr.TabItem("Voice Design"):
            with gr.Row():
                with gr.Column(scale=1):
                    vd_text = gr.Textbox(label="Text to Synthesize", lines=4, placeholder="Enter text here...")
                    vd_lang = lang_dropdown()
                    vd_groups = []
                    for cat, choices in CATEGORIES.items():
                        vd_groups.append(
                            gr.Dropdown(label=cat, choices=["Auto"] + choices, value="Auto", info=ATTR_INFO.get(cat))
                        )
                    vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po = gen_settings()
                    vd_btn = gr.Button("Generate", variant="primary", size="lg")
                with gr.Column(scale=1):
                    vd_audio = gr.Audio(label="Output Audio", type="filepath")
                    vd_status = gr.Textbox(label="Status", lines=2)

            def design_fn(text, lang, ns, gs, dn, sp, du, pp, po, *groups):
                return generate_speech(
                    text, lang, None, build_instruct(groups), ns, gs, dn, sp, du, pp, po, mode="design"
                )

            vd_btn.click(
                design_fn,
                inputs=[vd_text, vd_lang, vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po] + vd_groups,
                outputs=[vd_audio, vd_status],
            )
        with gr.TabItem("MFA Align"):
            with gr.Row():
                with gr.Column():
                    mfa_audio   = gr.Audio(label="Generated Audio", type="filepath")
                    mfa_jp_text = gr.Textbox(label="Japanese Script", lines=4)
                    mfa_map     = gr.Textbox(
                        label="Translation Map JSON (optional)", lines=4, value="[]",
                        placeholder="Leave as [] to test raw alignment; cinEZma fills this automatically.",
                    )
                    mfa_btn = gr.Button("Align", variant="primary", size="lg")
                with gr.Column():
                    mfa_output = gr.Textbox(label="Alignment JSON", lines=10)

            mfa_btn.click(
                align_japanese,
                inputs=[mfa_audio, mfa_jp_text, mfa_map],
                outputs=[mfa_output],
                api_name="align_japanese",
            )

# ── Launch Gradio + relay registration ───────────────────────────────────────
# Key insight: demo.share_url IS set while server runs (confirmed).
# We start a watcher thread first, then call launch() WITHOUT
# prevent_thread_lock — launch() blocks the main thread, which is fine.
# The watcher thread sees share_url appear, registers, and exits.
import threading, time

def _register():
    print("[Relay] Watching for share URL...")
    for _ in range(90):  # up to 3 min
        url = getattr(demo, "share_url", None)
        if url and "gradio.live" in str(url):
            url = str(url).strip()
            print(f"[Relay] Got URL: {url}")
            for attempt in range(1, 4):
                try:
                    r = requests.post(RELAY_URL, json={"url": url}, timeout=15)
                    if r.status_code == 200:
                        print(f"[Relay] ✓ Registered (attempt {attempt})")
                        return
                    else:
                        print(f"[Relay] Attempt {attempt} HTTP {r.status_code}: {r.text}")
                except Exception as e:
                    print(f"[Relay] Attempt {attempt} error: {e}")
                time.sleep(3)
            print(f"[Relay] ⚠ Failed. Paste manually: {url}")
            return
        time.sleep(2)
    print("[Relay] ⚠ share_url never appeared after 3 min.")

threading.Thread(target=_register, daemon=True).start()

demo.queue()
# No prevent_thread_lock — launch() blocks here, keeping the cell alive.
# share_url gets set by Gradio's tunnel thread within ~30s of blocking.
demo.launch(share=True, debug=True, theme=gr.themes.Soft(), css=CSS)
