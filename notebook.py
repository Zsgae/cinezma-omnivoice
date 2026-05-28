#!/usr/bin/env python3
"""
OmniVoice - cinEZma Edition
Auto-pulled from GitHub by the Kaggle launcher notebook.
Edit this file and push to GitHub — changes take effect on next Kaggle restart.
"""

# ── Model ─────────────────────────────────────────────────────────────────────
import glob
import logging
import os
from pathlib import Path

import numpy as np
import torch
from omnivoice import OmniVoice, OmniVoiceGenerationConfig

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logging.getLogger("omnivoice").setLevel(logging.INFO)

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
_CACHE_BACKUP = "/kaggle/working/hf-cache-backup"
if os.path.exists(_CACHE_BACKUP):
    os.environ["HF_HOME"] = _CACHE_BACKUP
    print("[Cache] Found local backup — skipping HuggingFace download.")
torch.backends.cuda.matmul.allow_tf32 = True

CHECKPOINT = "k2-fsa/OmniVoice"
VOICES_DIR = "/kaggle/working/voice-assets/voices"

print(f"Loading model from {CHECKPOINT} ...")
model = OmniVoice.from_pretrained(
    CHECKPOINT,
    device_map="cuda",
    dtype=torch.float16,
    load_asr=True,
    token=False,
)
sampling_rate = model.sampling_rate

def get_characters():
    wavs = glob.glob(os.path.join(VOICES_DIR, "*.wav"))
    return {Path(w).stem: w for w in sorted(wavs)}

print(f"Model loaded. Sampling rate: {sampling_rate} Hz")
print(f"Preset voices found: {len(get_characters())}")

import shutil, threading as _threading
def _backup_cache():
    src = Path.home() / ".cache" / "huggingface"
    dst = Path(_CACHE_BACKUP)
    if src.exists() and not dst.exists():
        print("[Cache] Backing up HF cache for fast restarts...")
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print("[Cache] ✓ Backup done.")
_threading.Thread(target=_backup_cache, daemon=True).start()


# ── Gradio UI + Relay ─────────────────────────────────────────────────────────
import gradio as gr
import requests
import time

SESSION_START_TIME = int(time.time() * 1000)  # ms timestamp, set once at boot

IDLE_TIMEOUT_SEC = 10 * 60  # 10 minutes — auto-kill if no generation request
_last_activity_time = time.time()  # updated on every generate / align call

RELAY_URL = "https://omnivoice-relay.zsage84869.workers.dev/register"

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
  <div class="brand-title">OmniVoice Demo</div>
  <div class="brand-subtitle">cinEZma Edition</div>
  <div class="hint">Preset character voices + custom clone + voice design + relay auto-connect</div>
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
        ns = gr.Slider(8, 64, value=32, step=1, label="Inference Steps")
        gs = gr.Slider(0.0, 10.0, value=3.0, step=0.1, label="Guidance Scale")
        dn = gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="Denoise Ratio")
        sp = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
        du = gr.Slider(0, 30, value=0, step=0.5, label="Duration (0 = auto)")
        pp = gr.Checkbox(value=True, label="Preprocess Prompt")
        po = gr.Checkbox(value=True, label="Postprocess Output")
    return ns, gs, dn, sp, du, pp, po

CATEGORIES = {
    "Gender": ["male", "female"],
    "Age": ["child", "teenager", "young adult", "middle-aged", "elderly"],
    "Pitch": ["very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"],
    "Style": ["whisper"],
    "English Accent": ["american accent", "british accent", "australian accent",
                       "canadian accent", "indian accent", "chinese accent",
                       "japanese accent", "korean accent", "portuguese accent",
                       "russian accent"],
    "Chinese Dialect": ["四川话", "陕西话", "广东话", "东北话", "河南话",
                        "云南话", "贵州话", "桂林话", "济南话"],
}

ATTR_INFO = {
    "Gender": "Speaker gender",
    "Age": "Approximate speaker age",
    "Pitch": "Voice pitch level",
    "Style": "Speaking style",
    "English Accent": "English accent (effective for English text)",
    "Chinese Dialect": "Chinese dialect (effective for Chinese text)",
}

def build_instruct(groups):
    selected = [g for g in groups if g and g != "Auto"]
    return ", ".join(selected) if selected else ""

def refresh_characters():
    chars = get_characters()
    choices = list(chars.keys())
    return gr.update(choices=choices, value=choices[0] if choices else None)

def generate_speech(text, language, ref_audio, instruct,
                    num_step, guidance_scale, denoise,
                    speed, duration, preprocess_prompt, postprocess_output,
                    mode="clone", ref_text=None):
    global _last_activity_time
    _last_activity_time = time.time()  # reset idle timer on every generation

    global _last_request
    _last_request = _time.time()
    if not text or not text.strip():
        return None, "Please enter some text."

    lang_code = None
    if language and language != "Auto":
        lang_code = language.split("(")[-1].rstrip(")").strip() if "(" in language else language

    gen_config = OmniVoiceGenerationConfig(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
        denoise=bool(denoise) if denoise is not None else True,
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
    )

    kw = {
        "text": text.strip(),
        "language": lang_code,
        "generation_config": gen_config,
    }

    if speed is not None and float(speed) != 1.0:
        kw["speed"] = float(speed)
    if duration is not None and float(duration) > 0:
        kw["duration"] = float(duration)

    if mode == "clone":
        if ref_audio is None:
            return None, "Please provide a reference audio."
        kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
        )

    if mode == "design" and instruct and instruct.strip():
        kw["instruct"] = instruct.strip()

    try:
        audio = model.generate(**kw)
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"

    waveform = audio[0].squeeze()
    if hasattr(waveform, "numpy"):
        waveform = waveform.numpy()
    waveform = (waveform * 32767).astype(np.int16)
    return (sampling_rate, waveform), f"Generated {waveform.shape[-1] / sampling_rate:.1f}s audio at {sampling_rate}Hz"

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
    global _last_activity_time
    _last_activity_time = time.time()  # reset idle timer on every alignment call

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

        # ── Step 5: sanity check sum(jp_counts) ≈ len(stamps) ──────────────
        # Allow a small tolerance: MFA sometimes merges adjacent short sounds
        # (particles, small kana) producing slightly fewer stamps than words.
        # A mismatch of <= 10% is acceptable — we distribute the extras proportionally.
        count_sum = sum(jp_counts)
        tolerance = max(2, round(count_sum * 0.10))  # 10% tolerance, minimum 2
        count_ok  = abs(count_sum - len(stamps)) <= tolerance
        errors    = None if count_ok else (
            f"jp_counts sum ({count_sum}) != romaji stamp count ({len(stamps)}) "
            f"(delta={abs(count_sum - len(stamps))}, tolerance={tolerance}). "
            "The app should fall back to raw stamps."
        )
        if errors:
            print(f"[Align] WARNING: {errors}")
        elif count_sum != len(stamps):
            # Within tolerance but not exact — adjust jp_counts to match stamp count
            delta = len(stamps) - count_sum
            print(f"[Align] INFO: Adjusting jp_counts by {delta} (within tolerance)")
            # Add/remove from the longest tokens
            sorted_idx = sorted(range(len(jp_counts)), key=lambda i: -jp_counts[i])
            for i in range(abs(delta)):
                jp_counts[sorted_idx[i % len(sorted_idx)]] += (1 if delta > 0 else -1)

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


# Japanese particles — these are too short for MFA to pin reliably on their own.
# We merge them into the preceding token so MFA has fewer, longer targets.
_JP_PARTICLES = {
    'が', 'は', 'を', 'に', 'で', 'と', 'も', 'か', 'や', 'な', 'の',
    'へ', 'より', 'から', 'まで', 'て', 'で', 'ば', 'し',
    'た', 'だ', 'な', 'ね', 'よ', 'わ', 'さ', 'ぞ', 'ぜ',
}

# Katakana-only check — loanwords that cutlet might leave as English
_KATAKANA_RE = re.compile(r'^[\u30A0-\u30FF\u30FC]+$')

def _katakana_to_romaji(surface: str, katsu) -> str:
    """
    Force katakana loanwords through phonetic romaji conversion.
    cutlet sometimes returns English for well-known loanwords (e.g. ステンド → stained).
    We detect this and force proper kana romaji instead.
    """
    r = katsu.romaji(surface)
    # If output contains spaces and looks like English words, it's a loanword leak
    # Re-romanize character by character using hepburn table via cutlet's kana map
    r_clean = re.sub(r"[^A-Za-z\'\-]", " ", r).strip()
    # Heuristic: if the romaji has fewer chars than expected for pure kana, force it
    kana_chars = len([c for c in surface if ord(c) >= 0x30A0 and ord(c) <= 0x30FF])
    if kana_chars > 0 and len(r_clean.replace(" ", "")) < kana_chars:
        # Use cutlet's internal kana conversion, bypass dictionary lookup
        r = katsu.romaji(surface, ensure_ascii=False)
    return r


def _tokenize_with_index(text: str, cutlet_module, fugashi_module):
    """
    Tokenize Japanese into surface forms using fugashi/MeCab.
    Romanize each token with cutlet, merge particles into preceding token.

    Key improvements over v1:
    - Katakana loanwords are forced through phonetic romaji (no English leaking in)
    - Particles (が、は、を etc.) are merged into the previous token
      → fewer, longer MFA targets → fewer missed stamps → cleaner count checks

    Returns:
      token_pairs  -- [(jp_surface, romaji_word_count), ...]
      romaji_words -- flat list of romaji words in the same order
    """
    tagger = fugashi_module.Tagger()
    katsu  = cutlet_module.Cutlet()

    # First pass: collect all (surface, romaji_words) tuples
    raw_tokens = []
    for word in tagger(text.strip()):
        surface = word.surface
        if not surface.strip():
            continue

        # Force katakana through phonetic romaji to prevent English loanword leakage
        if _KATAKANA_RE.match(surface):
            # Build romaji syllable-by-syllable using cutlet's kana table
            r = katsu.romaji(surface)
            # Sanity check: if result contains spaces suggesting multi-word English,
            # and it doesn't look like Japanese phonetics, normalize it
            r_words = re.sub(r"[^A-Za-z\'\-]", " ", r).split()
            # If any word is > 6 chars and the surface is short katakana, likely English
            if any(len(w) > 6 for w in r_words) and len(surface) <= 4:
                # Fall back to direct hiragana-style reading via katakana map
                # Full katakana → romaji table including digraphs and long vowel mark
                # Process longest matches first (digraphs before single chars)
                kana_map_digraph = {
                    "キャ":"kya","キュ":"kyu","キョ":"kyo",
                    "シャ":"sha","シュ":"shu","ショ":"sho",
                    "チャ":"cha","チュ":"chu","チョ":"cho",
                    "ニャ":"nya","ニュ":"nyu","ニョ":"nyo",
                    "ヒャ":"hya","ヒュ":"hyu","ヒョ":"hyo",
                    "ミャ":"mya","ミュ":"myu","ミョ":"myo",
                    "リャ":"rya","リュ":"ryu","リョ":"ryo",
                    "ギャ":"gya","ギュ":"gyu","ギョ":"gyo",
                    "ジャ":"ja","ジュ":"ju","ジョ":"jo",
                    "ビャ":"bya","ビュ":"byu","ビョ":"byo",
                    "ピャ":"pya","ピュ":"pyu","ピョ":"pyo",
                    "ファ":"fa","フィ":"fi","フェ":"fe","フォ":"fo",
                    "ティ":"ti","ディ":"di","トゥ":"tu","ドゥ":"du",
                    "ウィ":"wi","ウェ":"we","ウォ":"wo",
                    "ヴァ":"va","ヴィ":"vi","ヴ":"vu","ヴェ":"ve","ヴォ":"vo",
                }
                kana_map = {
                    "ア":"a","イ":"i","ウ":"u","エ":"e","オ":"o",
                    "カ":"ka","キ":"ki","ク":"ku","ケ":"ke","コ":"ko",
                    "サ":"sa","シ":"shi","ス":"su","セ":"se","ソ":"so",
                    "タ":"ta","チ":"chi","ツ":"tsu","テ":"te","ト":"to",
                    "ナ":"na","ニ":"ni","ヌ":"nu","ネ":"ne","ノ":"no",
                    "ハ":"ha","ヒ":"hi","フ":"fu","ヘ":"he","ホ":"ho",
                    "マ":"ma","ミ":"mi","ム":"mu","メ":"me","モ":"mo",
                    "ヤ":"ya","ユ":"yu","ヨ":"yo",
                    "ラ":"ra","リ":"ri","ル":"ru","レ":"re","ロ":"ro",
                    "ワ":"wa","ヲ":"wo","ン":"n",
                    "ガ":"ga","ギ":"gi","グ":"gu","ゲ":"ge","ゴ":"go",
                    "ザ":"za","ジ":"ji","ズ":"zu","ゼ":"ze","ゾ":"zo",
                    "ダ":"da","デ":"de","ド":"do",
                    "バ":"ba","ビ":"bi","ブ":"bu","ベ":"be","ボ":"bo",
                    "パ":"pa","ピ":"pi","プ":"pu","ペ":"pe","ポ":"po",
                    "ッ":"t",  # geminate — will be doubled by next consonant
                }
                # Build romaji by processing digraphs first, then singles
                result_r = ""
                i_k = 0
                while i_k < len(surface):
                    two = surface[i_k:i_k+2]
                    if two in kana_map_digraph:
                        result_r += kana_map_digraph[two]
                        i_k += 2
                    elif surface[i_k] == "ー":
                        # Long vowel: repeat previous vowel
                        result_r += result_r[-1] if result_r and result_r[-1] in "aeiou" else "o"
                        i_k += 1
                    else:
                        result_r += kana_map.get(surface[i_k], surface[i_k])
                        i_k += 1
                r = result_r
        else:
            r = katsu.romaji(surface)

        r = re.sub(r"[^A-Za-z\'\-]", " ", r)
        r = re.sub(r"\s+", " ", r).strip().lower()
        words = r.split()
        if not words:
            continue  # Punctuation / symbol — skip
        raw_tokens.append((surface, words))

    if not raw_tokens:
        return [], []

    # Second pass: merge particles into the preceding token
    merged = []
    for surface, words in raw_tokens:
        if surface in _JP_PARTICLES and merged:
            # Absorb this particle into the previous token
            prev_surface, prev_words = merged[-1]
            merged[-1] = (prev_surface + surface, prev_words + words)
        else:
            merged.append((surface, words))

    token_pairs  = [(s, len(w)) for s, w in merged]
    romaji_words = [w for _, words in merged for w in words]

    return token_pairs, romaji_words




with gr.Blocks(title="OmniVoice Demo") as demo:
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
                        placeholder="Transcript of the preset clip. Leave blank for auto-transcription.",
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
                        placeholder="Enter text here... You can use tags like [laughter], [sigh], etc.",
                    )
                    vc_ref_audio = gr.Audio(
                        label="Reference Audio (3-10s recommended)",
                        type="filepath",
                    )
                    vc_ref_text = gr.Textbox(
                        label="Reference Text (optional)",
                        lines=2,
                        placeholder="Transcript of ref audio. Leave blank for auto-transcription.",
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
        with gr.TabItem("✍️ WriterBot"):

            # ── WriterBot: cinEZma Narrative Engine (Python mirror) ───────────
            import google.generativeai as _genai
            import json as _wb_json
            import math as _wb_math

            _genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
            _WB_MODEL = "gemini-2.5-flash"

            # ── Contracts (mirrored 1:1 from WriterBot.tsx) ───────────────────
            _STORY_JSON_CONTRACT = """Return valid JSON only with this shape:
{
  "story": [
    {
      "panelIndex": 0,
      "text": "finished English narration for this panel (must match translationMap english phrases concatenated in order)",
      "japaneseScript": "natural Japanese voiceover (must match translationMap japanese phrases concatenated in order)",
      "translationMap": [{ "jp": "natural Japanese phrase representing a logical spoken breath unit", "en": "matching English translation segment representing the exact same pause/breath boundaries" }],
      "cameraMove": "STATIC | PAN_DOWN | PAN_UP | PAN_RIGHT | PAN_LEFT | ZOOM_IN | ZOOM_OUT | TILT_UP | TILT_DOWN | DRAMATIC_ZOOM | HANDHELD",
      "visualEffect": "NONE | SHAKE | FLASH | BLOOM | GRAIN | VIGNETTE | CINEMATIC_NOIR | WARM_ADVENTURE"
    }
  ]
}"""

            _TRANSLATION_MAP_CONTRACT = """Translation map rules:
- Co-Design English and Japanese together as a synchronized unit: identical pause and rhetorical breath points, sharing the exact same pacing agreement.
- Ensure breaths match perfectly: every comma, punctuation marker, or natural rhythmic pause in Japanese (jp) must correspond to a logical split-point or comma in English (en).
- Use phrase-level chunks only, like natural subtitle cards. Aim for 2-4 chunks per panel. Short panels can use 1 chunk.
- Both jp and en snippets must represent a complete, spoken-out-loud-able sub-phrase. Never leave dangling modifiers, prepositions, or articles at the edge of a chunk.
- Follow English reading order.
- Cover every English word and every Japanese character with no gaps.
- The text field is pure final English narration. No commentary, no labels, no analysis."""

            # ── AnalysisUtils (mirrored from studioUtils.ts) ──────────────────
            def _wb_estimate_read_time(word_count: int, panel_count: int) -> int:
                text_sec = (word_count / 200) * 60
                visual_sec = panel_count * 3.0
                return math.ceil(text_sec + visual_sec)

            def _wb_get_phase_distribution(total_panels: int, total_budget: int):
                distribution = [
                    {"name": "SETUP",               "weight": 0.20, "desc": "Context"},
                    {"name": "MOVEMENT",             "weight": 0.25, "desc": "Build"},
                    {"name": "TENSION",              "weight": 0.30, "desc": "Strain"},
                    {"name": "RELEASE",              "weight": 0.15, "desc": "Break"},
                    {"name": "Lingering AFTERMATH",  "weight": 0.10, "desc": "Don't wrap it up. Leave the final moment suspended and unresolved"},
                ]
                if total_panels <= 0:
                    return []
                desired = [total_panels * d["weight"] for d in distribution]
                counts = [int(d) for d in desired]
                assigned = sum(counts)
                while assigned < total_panels:
                    best = max(range(len(counts)), key=lambda i: desired[i] - counts[i])
                    counts[best] += 1
                    assigned += 1
                active_weight = sum(d["weight"] for d, c in zip(distribution, counts) if c > 0)
                safe_budget = max(total_panels, total_budget or total_panels)
                current_panel = 0
                assigned_budget = 0
                active = [(i, c) for i, c in enumerate(counts) if c > 0]
                result = []
                for pos, (idx, count) in enumerate(active):
                    phase = distribution[idx]
                    start = current_panel
                    end = min(total_panels, start + count)
                    is_last = pos == len(active) - 1
                    budget = (max(1, safe_budget - assigned_budget) if is_last
                              else max(1, round(safe_budget * (phase["weight"] / active_weight))))
                    current_panel = end
                    assigned_budget += budget
                    result.append({**phase, "range": {"start": start, "end": end}, "budget": budget})
                return result

            def _wb_build_pacing_budget(total_panels, total_budget_sec, multiplier, phases):
                panel_count = max(1, total_panels)
                safe_mult = multiplier if math.isfinite(multiplier) else 2.0
                safe_budget = max(panel_count * 4, total_budget_sec or panel_count * 8)
                visual_breath = min(safe_budget * 0.28, panel_count * 1.5)
                spoken_sec = max(panel_count * 2.5, safe_budget - visual_breath)
                target_words = max(panel_count * 8, round(spoken_sec * 2.35))
                min_words = max(panel_count * 6, round(target_words * 0.86))
                max_words = round(target_words * 1.18)
                panels_budget = []
                for pi in range(panel_count):
                    phase = next((p for p in phases if p["range"]["start"] <= pi < p["range"]["end"]), None)
                    phase_count = max(1, phase["range"]["end"] - phase["range"]["start"]) if phase else panel_count
                    phase_spoken = (max(phase_count * 2.5, phase["budget"] - min(phase["budget"] * 0.28, phase_count * 1.5))
                                    if phase else spoken_sec / panel_count)
                    phase_target = max(phase_count * 8, round(phase_spoken * 2.35)) if phase else target_words
                    tgt = max(6, round(phase_target / phase_count))
                    panels_budget.append({"panelIndex": pi, "targetWords": tgt,
                                          "minWords": max(4, round(tgt * 0.76)),
                                          "maxWords": max(8, round(tgt * 1.28))})
                return {"multiplier": safe_mult, "targetTotalWords": target_words,
                        "minTotalWords": min_words, "maxTotalWords": max_words, "panels": panels_budget}

            def _wb_format_pacing(budget):
                lines = [
                    f"Pacing multiplier: {budget['multiplier']:.1f}x. This controls actual text length, not just mood.",
                    f"Target total English narration: {budget['targetTotalWords']} words. Acceptable range: {budget['minTotalWords']}-{budget['maxTotalWords']} words.",
                    "Panel word budgets:\n" + "\n".join(
                        f"- Panel ID {p['panelIndex']}: aim for {p['targetWords']} words, acceptable {p['minWords']}-{p['maxWords']}"
                        for p in budget["panels"]
                    ),
                    "If the multiplier is higher, spend the extra words on lived-in reaction, sensory texture, and micro-reflection. If it is lower, compress the line instead."
                ]
                return "\n\n".join(lines)

            def _wb_count_words(text): return len((text or "").split())
            def _wb_count_story_words(story): return sum(_wb_count_words(p.get("text","")) for p in story)
            def _wb_join(*blocks): return "\n\n".join(b for b in blocks if b and b.strip())

            def _wb_call_gemini(system_instruction, contents, json_mode=True):
                model_obj = _genai.GenerativeModel(
                    model_name=_WB_MODEL,
                    system_instruction=system_instruction or None,
                )
                cfg = {"response_mime_type": "application/json"} if json_mode else {}
                response = model_obj.generate_content(contents, generation_config=cfg if cfg else None)
                return response.text

            def _wb_parse_json(text):
                clean = (text or "").replace("```json","").replace("```","").strip()
                return _wb_json.loads(clean)

            # ── Main generation function ──────────────────────────────────────
            def writerbot_generate(panels_json, base_persona, tone, pacing_mult, context, style_dna):
                import math
                panels_json = (panels_json or "").strip()
                if not panels_json:
                    return "", "⚠ Paste your panels JSON first."
                try:
                    panels = _wb_json.loads(panels_json)
                    if not isinstance(panels, list):
                        panels = panels.get("story") or panels.get("panels") or []
                except Exception as e:
                    return "", f"⚠ Invalid JSON: {e}"

                total_panels = len(panels)
                if total_panels == 0:
                    return "", "⚠ No panels found in JSON."

                try:
                    pacing_mult = float(pacing_mult)
                except:
                    pacing_mult = 1.8

                # ── Build pacing budget ───────────────────────────────────────
                seed_words = sum(
                    _wb_count_words(p.get("text",""))
                    for p in panels
                    if 0 < _wb_count_words(p.get("text","")) <= 8
                )
                est_words = max(seed_words, total_panels * 15)
                read_time = _wb_estimate_read_time(est_words, total_panels)
                ntb = math.ceil(read_time * pacing_mult)
                phases = _wb_get_phase_distribution(total_panels, ntb)
                budget = _wb_build_pacing_budget(total_panels, ntb, pacing_mult, phases)
                pacing_text = _wb_format_pacing(budget)

                rhythm_plan = "\n".join(
                    f"- {p['name']}: panels {p['range']['start']}-{p['range']['end']-1}, about {p['budget']}s"
                    for p in phases
                )

                # ── Build panel prompt parts ──────────────────────────────────
                panel_hints = []
                for i, p in enumerate(panels):
                    has_text = 0 < _wb_count_words(p.get("text","")) <= 8
                    hint = f"[Panel ID: {i}" + (f' | Context: "{p["text"]}"' if has_text else "") + "]"
                    panel_hints.append(hint)
                panel_block = "\n".join(panel_hints)

                # ── Phase 1: Initial pass ─────────────────────────────────────
                system_1 = _wb_join(
                    base_persona,
                    tone and f"The current emotional weather is {tone}. Let it color word choice and rhythm without naming it unless the story naturally would.",
                    "Create the first usable version from inside the scene.",
                    "Output only finished content inside the requested JSON fields. Do not explain your process or announce style choices."
                )

                prompt_1 = _wb_join(
                    f"Write finished narratable content for {total_panels} panels.",
                    rhythm_plan and f"Rhythm plan:\n{rhythm_plan}",
                    pacing_text,
                    context and f"Project context:\n{context}",
                    panel_block,
                    "Use the panel hints as the source of truth. Decide what matters emotionally, then write the line as content, not as analysis.",
                    _STORY_JSON_CONTRACT,
                    _TRANSLATION_MAP_CONTRACT
                )

                # Style DNA priming (same conversation-history trick as cinEZma)
                if style_dna and style_dna.strip():
                    contents_1 = [
                        {"role": "user",  "parts": [{"text": style_dna.strip()}]},
                        {"role": "model", "parts": [{"text": "understood."}]},
                        {"role": "user",  "parts": [{"text": prompt_1}]},
                    ]
                else:
                    contents_1 = prompt_1

                try:
                    raw_1 = _wb_call_gemini(system_1, contents_1)
                    result_1 = _wb_parse_json(raw_1)
                    stories = result_1.get("story", [])
                    if not stories:
                        return "", "⚠ Phase 1 returned no story data."
                except Exception as e:
                    return "", f"⚠ Phase 1 failed: {e}"

                # ── Phase 2: Refinement pass ──────────────────────────────────
                system_2 = _wb_join(
                    base_persona,
                    tone and f"The current emotional weather is {tone}. Let it color word choice and rhythm without naming it unless the story naturally would.",
                    "Revise the draft as the same storyteller. Preserve story facts, panel indexes, camera choices, and the JSON shape while making the narration feel less instructed and more lived-in.",
                    "Output only finished content inside the requested JSON fields. Do not explain your process or announce style choices."
                )

                prompt_2 = _wb_join(
                    "Revise this draft into final narratable content. Keep every panelIndex and preserve the core event in each panel.",
                    "Keep cameraMove and visualEffect unless a value is missing or invalid. Do not add analysis, labels, or explanations.",
                    pacing_text,
                    f"Draft JSON:\n{_wb_json.dumps(stories)}",
                    _STORY_JSON_CONTRACT,
                    _TRANSLATION_MAP_CONTRACT
                )

                if style_dna and style_dna.strip():
                    contents_2 = [
                        {"role": "user",  "parts": [{"text": style_dna.strip()}]},
                        {"role": "model", "parts": [{"text": "understood."}]},
                        {"role": "user",  "parts": [{"text": prompt_2}]},
                    ]
                else:
                    contents_2 = prompt_2

                try:
                    raw_2 = _wb_call_gemini(system_2, contents_2)
                    result_2 = _wb_parse_json(raw_2)
                    stories = result_2.get("story", stories)  # fallback to phase 1 if broken
                except Exception as e:
                    pass  # Phase 1 output is kept as fallback

                # ── Pacing repair pass (if outside budget) ───────────────────
                actual_words = _wb_count_story_words(stories)
                if actual_words < budget["minTotalWords"] or actual_words > budget["maxTotalWords"]:
                    should_expand = actual_words < budget["minTotalWords"]
                    prompt_repair = _wb_join(
                        f"The previous draft missed the pacing budget at {actual_words} English words. "
                        + ("Expand the narration so the multiplier creates visibly more text." if should_expand
                           else "Compress the narration so the multiplier creates visibly less text."),
                        pacing_text,
                        "Preserve panelIndex values, story facts, cameraMove, visualEffect, and valid translationMap coverage. Rewrite japaneseScript and translationMap so they match the revised English text.",
                        f"Draft JSON:\n{_wb_json.dumps(stories)}",
                        _STORY_JSON_CONTRACT,
                        _TRANSLATION_MAP_CONTRACT
                    )
                    try:
                        raw_r = _wb_call_gemini(system_2, prompt_repair)
                        result_r = _wb_parse_json(raw_r)
                        repaired = result_r.get("story", [])
                        if repaired:
                            stories = repaired
                    except Exception:
                        pass  # keep current stories

                output_json = _wb_json.dumps({"story": stories}, ensure_ascii=False, indent=2)
                final_words = _wb_count_story_words(stories)
                status = f"✅ Done — {len(stories)} panels, {final_words} words (target {budget['targetTotalWords']})"
                return output_json, status

            # ── Gradio UI ─────────────────────────────────────────────────────
            with gr.Row():
                with gr.Column(scale=1):
                    wb_panels = gr.Textbox(
                        label="Panels JSON  (paste your cinEZma scriptLines array here)",
                        lines=10,
                        placeholder='[{"text": "dust settles over the valley"}, {"text": ""}, ...]',
                    )
                    with gr.Accordion("WriterBot Settings", open=True):
                        wb_persona = gr.Textbox(
                            label="Base Persona",
                            lines=3,
                            placeholder="Define the narrator voice / identity...",
                        )
                        wb_tone = gr.Textbox(
                            label="Target Tone",
                            placeholder="e.g. Gritty, Noir, Melancholic",
                        )
                        wb_pacing = gr.Slider(
                            1.0, 3.0, value=1.8, step=0.1,
                            label="Pacing Multiplier (1.0 = fast, 3.0 = slow/dense)",
                        )
                        wb_context = gr.Textbox(
                            label="Global Context (optional)",
                            lines=2,
                            placeholder="Any project-level context for the AI...",
                        )
                        wb_style = gr.Textbox(
                            label="Style DNA (paste your own writing — rhythm primer)",
                            lines=4,
                            placeholder="Paste raw personal writing here. No plot, just how your thoughts move...",
                        )
                    wb_btn = gr.Button("⚡ Execute Generation Sequence", variant="primary", size="lg")
                with gr.Column(scale=1):
                    wb_output = gr.Textbox(label="Output JSON  (paste back into cinEZma)", lines=28)
                    wb_status = gr.Textbox(label="Status", lines=1)
                    wb_clear = gr.Button("Clear", size="sm")

            wb_btn.click(
                writerbot_generate,
                inputs=[wb_panels, wb_persona, wb_tone, wb_pacing, wb_context, wb_style],
                outputs=[wb_output, wb_status],
                api_name="writerbot",
            )
            wb_clear.click(lambda: ("", "", ""), outputs=[wb_panels, wb_output, wb_status])

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

def _play_ready_chime():
    """Fire a chime in the Kaggle browser tab — server is live."""
    try:
        from IPython.display import display, Javascript
        display(Javascript("""
        (function() {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            // C major arpeggio: C5 → E5 → G5 → C6
            [523.25, 659.25, 783.99, 1046.50].forEach((freq, i) => {
                const osc  = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.type = 'sine';
                osc.frequency.value = freq;
                const t = ctx.currentTime + i * 0.18;
                gain.gain.setValueAtTime(0, t);
                gain.gain.linearRampToValueAtTime(0.25, t + 0.02);
                gain.gain.exponentialRampToValueAtTime(0.001, t + 0.5);
                osc.start(t);
                osc.stop(t + 0.5);
            });
        })();
        """))
    except Exception as e:
        print(f"[Chime] Could not play sound: {e}")

def _register():
    print("[Relay] Watching for share URL...")
    for _ in range(90):  # up to 3 min
        url = getattr(demo, "share_url", None)
        if url and "gradio.live" in str(url):
            url = str(url).strip()
            print(f"[Relay] Got URL: {url}")
            for attempt in range(1, 4):
                try:
                    r = requests.post(RELAY_URL, json={"url": url, "started_at": SESSION_START_TIME}, timeout=15)
                    if r.status_code == 200:
                        print(f"[Relay] ✓ Registered (attempt {attempt})")
                        _play_ready_chime()  # 🔔 server is up and relay is live
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

import time as _time
_last_request = _time.time()
_IDLE_TIMEOUT = 600

def _watchdog():
    while True:
        _time.sleep(30)
        if _time.time() - _last_request > _IDLE_TIMEOUT:
            print("[Watchdog] Idle — closing tunnel. Re-run cell 4 to reopen.")
            demo.close()
            import os as _os
            _os._exit(0)

threading.Thread(target=_watchdog, daemon=True).start()

demo.queue()

# ── Background cache warm-up ──────────────────────────────────────────────────
# Runs concurrently with the live server — no waiting at startup.
# Waits for share_url to confirm the server is actually accepting requests
# before doing any heavy work (avoids race with model init).
def _warm_cache():
    print("[Cache] Waiting for server to come up before warming cache...")
    for _ in range(90):
        if getattr(demo, "share_url", None):
            break
        time.sleep(2)
    else:
        print("[Cache] ⚠ Server never came up — skipping cache warm.")
        return

    print("[Cache] Server is up. Starting background cache warm-up...")
    try:
        # ── Pre-load the MFA aligner model ───────────────────────────────────
        # First align_japanese call is slow because it downloads MMS_FA weights.
        # We trigger a no-op alignment here so the model is hot before real use.
        global _ALIGNER_MODEL
        if _ALIGNER_MODEL is None:
            from ctc_forced_aligner import get_word_stamps
            import tempfile, soundfile as sf
            # Minimal 0.5s silent WAV at 16kHz — just enough to init the model
            silent = np.zeros(int(_TARGET_SR * 0.5), dtype=np.float32)
            with tempfile.TemporaryDirectory() as tmp:
                wav = os.path.join(tmp, "warm.wav")
                txt = os.path.join(tmp, "warm.txt")
                sf.write(wav, silent, _TARGET_SR)
                with open(txt, "w") as f:
                    f.write("a")
                try:
                    result = get_word_stamps(wav, txt, model=None, model_type="MMS_FA")
                    if isinstance(result, tuple) and len(result) > 1:
                        _ALIGNER_MODEL = result[1]
                        print("[Cache] ✓ MFA aligner model loaded and cached.")
                except Exception as e:
                    print(f"[Cache] MFA warm-up failed (non-fatal): {e}")

        # ── Add more warm-up steps here as needed ────────────────────────────
        # e.g. pre-load a voice clone prompt for the first preset character:
        # chars = get_characters()
        # if chars:
        #     first_wav = next(iter(chars.values()))
        #     model.create_voice_clone_prompt(ref_audio=first_wav, ref_text=None)
        #     print("[Cache] ✓ Voice clone prompt pre-warmed.")

        print("[Cache] ✓ Cache warm-up complete.")
    except Exception as e:
        print(f"[Cache] ⚠ Warm-up error (non-fatal): {e}")

threading.Thread(target=_warm_cache, daemon=True).start()

# ── Idle auto-kill ────────────────────────────────────────────────────────────
IDLE_WARN_SEC = 8 * 60  # warn at 8 min, kill at 10 min

def _idle_watcher():
    warned = False
    while True:
        time.sleep(60)
        idle = time.time() - _last_activity_time
        remaining = IDLE_TIMEOUT_SEC - idle
        if idle >= IDLE_TIMEOUT_SEC:
            print(f"[IdleTimer] ⏹  No activity for {IDLE_TIMEOUT_SEC // 60} min — shutting down OmniVoice.")
            demo.close()
            os._exit(0)
        elif idle >= IDLE_WARN_SEC and not warned:
            warned = True
            print(f"[IdleTimer] ⚠  Idle for {int(idle // 60)} min. "
                  f"Auto-kill in ~{int(remaining // 60)} min if no requests arrive.")
        elif idle < IDLE_WARN_SEC:
            warned = False  # reset so warning fires again after activity resumes

threading.Thread(target=_idle_watcher, daemon=True).start()

demo.queue()
# No prevent_thread_lock — launch() blocks here, keeping the cell alive.
# share_url gets set by Gradio's tunnel thread within ~30s of blocking.
demo.launch(share=True, debug=True, theme=gr.themes.Soft(), css=CSS)
