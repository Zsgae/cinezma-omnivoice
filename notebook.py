#!/usr/bin/env python3
"""
Fish Audio S2 Pro - cinEZma Edition
Auto-pulled from GitHub by the Kaggle launcher notebook.
Edit this file and push to GitHub; changes take effect on next Kaggle restart.
"""

# ── Model ─────────────────────────────────────────────────────────────────────
import glob
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
import tempfile
import time
import wave

import numpy as np

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logging.getLogger("urllib3").setLevel(logging.WARNING)

FISH_MODEL = os.environ.get("FISH_MODEL") or "fishaudio/s2-pro"
FISH_REPO_URL = os.environ.get("FISH_REPO_URL", "https://github.com/fishaudio/fish-speech.git")
FISH_REPO_REF = os.environ.get("FISH_REPO_REF", "main")
FISH_ROOT = Path(os.environ.get("FISH_ROOT", "/kaggle/working/fish-speech"))
FISH_CHECKPOINT_DIR = Path(
    os.environ.get("FISH_CHECKPOINT_DIR", str(FISH_ROOT / "checkpoints" / "s2-pro"))
)
VOICES_DIR = "/kaggle/working/voice-assets/voices"
OUTPUT_DIR = Path(os.environ.get("FISH_OUTPUT_DIR", "/kaggle/working/fish-audio-output"))
FISH_DEVICE = os.environ.get("FISH_DEVICE", "cuda")
FISH_CODEC_DEVICE = os.environ.get("FISH_CODEC_DEVICE", "cpu")
FISH_TOP_K = int(os.environ.get("FISH_TOP_K", "30"))
FISH_SEED = int(os.environ.get("FISH_SEED", "42"))


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


FISH_AUTO_SETUP = _env_flag("FISH_AUTO_SETUP", True)
FISH_APT_SETUP = _env_flag("FISH_APT_SETUP", True)
FISH_HALF = _env_flag("FISH_HALF", True)
FISH_COMPILE = _env_flag("FISH_COMPILE", False)
FISH_LOCAL_ASR = _env_flag("FISH_LOCAL_ASR", False)
FISH_LOCAL_ASR_MODEL = os.environ.get("FISH_LOCAL_ASR_MODEL", "small")

_FISH_READY = False
_TEXT_MODEL = None
_DECODE_ONE_TOKEN = None
_CODEC_MODEL = None
_MODEL_LOAD_SECONDS = None


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


def _run_cmd(cmd, cwd=None, timeout=None):
    printable = " ".join(str(part) for part in cmd)
    print(f"[Fish Local] {printable}")
    result = subprocess.run(
        [str(part) for part in cmd],
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        tail = (result.stdout or "")[-4000:]
        raise RuntimeError(f"Command failed ({result.returncode}): {printable}\n{tail}")
    if result.stdout:
        print(result.stdout[-1200:])
    return result.stdout or ""


def _ensure_fish_repo():
    if (FISH_ROOT / "fish_speech").exists():
        return
    if not FISH_AUTO_SETUP:
        raise RuntimeError(f"Fish Speech repo not found at {FISH_ROOT}.")
    if not shutil.which("git"):
        raise RuntimeError("git is required to clone fish-speech into Kaggle working storage.")

    FISH_ROOT.parent.mkdir(parents=True, exist_ok=True)
    _run_cmd(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            FISH_REPO_REF,
            FISH_REPO_URL,
            str(FISH_ROOT),
        ],
        timeout=None,
    )


def _verify_fish_runtime_imports():
    import importlib

    importlib.invalidate_caches()
    from fish_speech.models.text2semantic.inference import (  # noqa: F401
        decode_to_audio,
        generate_long,
        init_model,
        load_codec_model,
    )
    import soundfile  # noqa: F401


def _ensure_fish_package():
    if str(FISH_ROOT) not in sys.path:
        sys.path.insert(0, str(FISH_ROOT))
    try:
        _verify_fish_runtime_imports()

        return
    except ModuleNotFoundError as exc:
        missing_dep = exc.name
        if not FISH_AUTO_SETUP:
            raise
        print(f"[Fish Local] Missing Python dependency '{missing_dep}'. Installing Fish Speech package deps ...")
    except Exception:
        if not FISH_AUTO_SETUP:
            raise
        print("[Fish Local] Fish Speech runtime import failed. Installing Fish Speech package deps ...")

    install_target = os.environ.get("FISH_PIP_TARGET", ".")
    _run_cmd([sys.executable, "-m", "pip", "install", "-e", install_target], cwd=FISH_ROOT, timeout=None)

    if str(FISH_ROOT) not in sys.path:
        sys.path.insert(0, str(FISH_ROOT))
    _verify_fish_runtime_imports()


def _ensure_system_deps():
    if not FISH_AUTO_SETUP or not FISH_APT_SETUP or not shutil.which("apt-get"):
        return
    marker = Path("/kaggle/working/.fish_speech_system_deps_ok")
    if marker.exists():
        return
    _run_cmd(["apt-get", "update"], timeout=None)
    _run_cmd(["apt-get", "install", "-y", "portaudio19-dev", "libsox-dev", "ffmpeg"], timeout=None)
    try:
        marker.touch()
    except OSError:
        pass


def _ensure_fish_weights():
    model_path = FISH_CHECKPOINT_DIR / "model.pth"
    codec_path = FISH_CHECKPOINT_DIR / "codec.pth"
    if model_path.exists() and codec_path.exists():
        return
    if not FISH_AUTO_SETUP:
        raise RuntimeError(f"Fish S2 Pro weights missing at {FISH_CHECKPOINT_DIR}.")

    try:
        from huggingface_hub import snapshot_download
    except Exception:
        _run_cmd([sys.executable, "-m", "pip", "install", "huggingface_hub"], timeout=None)
        from huggingface_hub import snapshot_download

    FISH_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Fish Local] Downloading {FISH_MODEL} to {FISH_CHECKPOINT_DIR} ...")
    snapshot_download(
        repo_id=FISH_MODEL,
        local_dir=str(FISH_CHECKPOINT_DIR),
        local_dir_use_symlinks=False,
    )


def _ensure_local_fish_ready():
    global _FISH_READY
    if _FISH_READY:
        return
    _ensure_fish_repo()
    _ensure_system_deps()
    _ensure_fish_package()
    _ensure_fish_weights()
    _FISH_READY = True


def _load_local_models():
    global _TEXT_MODEL, _DECODE_ONE_TOKEN, _CODEC_MODEL, _MODEL_LOAD_SECONDS
    _ensure_local_fish_ready()
    if _TEXT_MODEL is not None and _CODEC_MODEL is not None:
        return

    import torch
    from fish_speech.models.text2semantic.inference import init_model, load_codec_model

    if FISH_DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("FISH_DEVICE is cuda, but CUDA is not available in this Kaggle session.")

    precision = torch.half if FISH_HALF else torch.bfloat16
    codec_precision = torch.float32 if FISH_CODEC_DEVICE == "cpu" else precision
    start = time.time()
    print(f"[Fish Local] Loading {FISH_MODEL} from {FISH_CHECKPOINT_DIR} on {FISH_DEVICE} ...")
    _TEXT_MODEL, _DECODE_ONE_TOKEN = init_model(
        FISH_CHECKPOINT_DIR,
        FISH_DEVICE,
        precision,
        compile=FISH_COMPILE,
    )
    with torch.device(FISH_DEVICE):
        _TEXT_MODEL.setup_caches(
            max_batch_size=1,
            max_seq_len=_TEXT_MODEL.config.max_seq_len,
            dtype=next(_TEXT_MODEL.parameters()).dtype,
        )
    _CODEC_MODEL = load_codec_model(FISH_CHECKPOINT_DIR / "codec.pth", FISH_CODEC_DEVICE, codec_precision)
    _MODEL_LOAD_SECONDS = time.time() - start
    print(f"[Fish Local] Models loaded in {_MODEL_LOAD_SECONDS:.1f}s.")


def _transcribe_reference_audio(audio_path, lang_code=None):
    if not FISH_LOCAL_ASR:
        raise RuntimeError(
            "Local Fish S2 cloning needs the reference transcript. Type it in, "
            "or place a .txt/.lab sidecar next to the reference wav. To try local "
            "Whisper auto-transcription, set FISH_LOCAL_ASR=1 before launch."
        )

    try:
        from faster_whisper import WhisperModel
    except Exception:
        if not FISH_AUTO_SETUP:
            raise
        _run_cmd([sys.executable, "-m", "pip", "install", "faster-whisper"], timeout=None)
        from faster_whisper import WhisperModel

    device = os.environ.get("FISH_LOCAL_ASR_DEVICE", "cpu")
    compute_type = os.environ.get("FISH_LOCAL_ASR_COMPUTE_TYPE", "int8" if device == "cpu" else "float16")
    model = WhisperModel(FISH_LOCAL_ASR_MODEL, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(audio_path, language=lang_code if lang_code else None, beam_size=5)
    transcript = " ".join(segment.text.strip() for segment in segments).strip()
    if not transcript:
        raise RuntimeError("Local Whisper returned an empty transcript. Please type the reference text.")
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


def _prepare_reference(audio_path, ref_text, lang_code):
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
        transcript = _transcribe_reference_audio(audio_path, lang_code)
        transcript_source = "local Whisper transcript"

    return transcript, transcript_source


def _apply_design_tag(text, instruct):
    if instruct and instruct.strip():
        return f"[{instruct.strip()}] {text.strip()}"
    return text.strip()


def _apply_speed_hint(text, speed):
    speed = _clamp_float(speed, 1.0, 0.5, 2.0)
    if speed >= 1.15:
        return f"[speaking quickly] {text}"
    if speed <= 0.85:
        return f"[speaking slowly] {text}"
    return text


def _resample_audio(data, sr, target_sr):
    if sr == target_sr:
        return data.astype(np.float32)

    try:
        import math
        from scipy.signal import resample_poly

        gcd = math.gcd(int(sr), int(target_sr))
        return resample_poly(data, target_sr // gcd, sr // gcd).astype(np.float32)
    except Exception:
        target_len = max(1, int(round(len(data) * target_sr / sr)))
        return np.interp(
            np.linspace(0, len(data) - 1, target_len),
            np.arange(len(data)),
            data,
        ).astype(np.float32)


def _encode_audio_local(audio_path, codec, device):
    import soundfile as sf
    import torch

    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.size == 0:
        raise RuntimeError(f"Reference audio is empty: {audio_path}")

    target_sr = int(codec.sample_rate)
    data = _resample_audio(data.astype(np.float32), int(sr), target_sr)

    wav = torch.from_numpy(data).to(device)
    model_dtype = next(codec.parameters()).dtype
    audios = wav[None, None].to(dtype=model_dtype)
    audio_lengths = torch.tensor([wav.numel()], device=device, dtype=torch.long)
    indices, feature_lengths = codec.encode(audios, audio_lengths)
    return indices[0, :, : feature_lengths[0]]


def _new_output_path():
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_dir = OUTPUT_DIR
    except OSError:
        out_dir = Path(tempfile.gettempdir())
    return str(out_dir / f"fish_s2_pro_{int(time.time() * 1000)}.wav")


def _wav_duration(audio_path):
    try:
        with wave.open(audio_path, "rb") as wav_file:
            sr = wav_file.getframerate()
            duration = wav_file.getnframes() / float(sr)
        return duration, sr
    except Exception:
        return None, 44100

def get_characters():
    wavs = glob.glob(os.path.join(VOICES_DIR, "*.wav"))
    return {Path(w).stem: w for w in sorted(wavs)}

print(f"Fish Speech local model configured: {FISH_MODEL}")
print(f"Fish Speech repo path: {FISH_ROOT}")
print(f"Fish Speech checkpoint path: {FISH_CHECKPOINT_DIR}")
print(f"Fish Speech devices: model={FISH_DEVICE}, codec={FISH_CODEC_DEVICE}")
print(f"Preset voices found: {len(get_characters())}")


# ── Gradio UI + Relay ─────────────────────────────────────────────────────────
import gradio as gr
import requests

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
  <div class="brand-title">Fish Speech S2 Pro Local</div>
  <div class="brand-subtitle">cinEZma Edition</div>
  <div class="hint">Kaggle GPU generation + preset voices + custom clone + S2 inline control</div>
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
        po = gr.Checkbox(value=True, label="Iterative Prompt")
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
        _load_local_models()
        import soundfile as sf
        import torch
        from fish_speech.models.text2semantic.inference import (
            decode_to_audio,
            generate_long,
        )

        lang_code = _language_code(language)
        prompt_text = None
        prompt_tokens = None
        transcript_source = None
        clean_text = _apply_design_tag(text, instruct if mode == "design" else None)
        clean_text = _apply_speed_hint(clean_text, speed)

        if mode == "clone":
            prompt_text, transcript_source = _prepare_reference(
                ref_audio,
                ref_text,
                lang_code,
            )
            prompt_tokens = [_encode_audio_local(ref_audio, _CODEC_MODEL, FISH_CODEC_DEVICE).cpu()]

        torch.manual_seed(FISH_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(FISH_SEED)

        generator = generate_long(
            model=_TEXT_MODEL,
            device=FISH_DEVICE,
            decode_one_token=_DECODE_ONE_TOKEN,
            text=clean_text.strip(),
            num_samples=1,
            max_new_tokens=_clamp_int(max_new_tokens, 1024, 0, 4096),
            top_p=_clamp_float(top_p, 0.7, 0.01, 1.0),
            top_k=FISH_TOP_K,
            temperature=_clamp_float(temperature, 0.7, 0.01, 1.99),
            compile=FISH_COMPILE,
            iterative_prompt=bool(condition_on_previous_chunks),
            chunk_length=_clamp_int(chunk_length, 300, 100, 300),
            prompt_text=[prompt_text] if prompt_text else None,
            prompt_tokens=prompt_tokens,
        )

        codes = []
        for response in generator:
            if response.action == "sample":
                codes.append(response.codes)

        if not codes:
            return None, "Fish S2 Pro returned no audio codes."

        merged_codes = torch.cat(codes, dim=1).to(FISH_CODEC_DEVICE)
        audio = decode_to_audio(merged_codes, _CODEC_MODEL)
        audio_path = _new_output_path()
        sf.write(audio_path, audio.cpu().float().numpy(), _CODEC_MODEL.sample_rate)
        duration_seconds, sr = _wav_duration(audio_path)
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"

    if duration_seconds is None:
        status = f"Generated audio locally with Fish Speech {FISH_MODEL}."
    else:
        status = f"Generated {duration_seconds:.1f}s locally with Fish Speech {FISH_MODEL} at {sr}Hz."
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




with gr.Blocks(title="Fish Speech S2 Pro Local") as demo:
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
                        placeholder="Transcript of the preset clip. Blank uses .txt/.lab sidecar or local Whisper if enabled.",
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
                        placeholder="Transcript of ref audio. Blank uses local Whisper only if FISH_LOCAL_ASR=1.",
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
