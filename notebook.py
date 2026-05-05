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


# ── Gradio UI + Relay ─────────────────────────────────────────────────────────
import gradio as gr
import requests
import time

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
                    silences.append([round(gap_start, 3), round(t, 3)])

    if in_silence:
        t_end = n_frames * frame_ms / 1000.0
        if t_end - gap_start >= min_silence_sec:
            silences.append([round(gap_start, 3), round(t_end, 3)])
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
                "word":  text,
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
