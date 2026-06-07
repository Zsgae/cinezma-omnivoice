#!/usr/bin/env python3
"""
OmniVoice + Qwen3-TTS — cinEZma Edition
Two TTS engines, one shared voice library, one relay.

Engines:
  • OmniVoice  (k2-fsa/OmniVoice)     — Japanese + multi-lang, MFA alignment
  • Qwen3-TTS  (Qwen/Qwen3-TTS-*)     — Voice clone / Custom / Voice Design

Shared voice library:
  /kaggle/working/voice-assets/voices/*.wav
  — jp_charon (and any future voices you add) work in BOTH engines.

Relay: same Cloudflare Worker as before.
API endpoints preserved for cinEZma:
  generate_preset   → OmniVoice preset clone
  generate_custom   → OmniVoice custom clone
  align_japanese    → MFA forced alignment
  qwen_clone        → Qwen3 voice clone
  qwen_custom       → Qwen3 preset character voice
  qwen_design       → Qwen3 voice design
"""

# ── Shared Imports ─────────────────────────────────────────────────────────────
import gc
import glob
import json as _json
import logging
import os
import re
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logging.getLogger("omnivoice").setLevel(logging.INFO)

# ── GPU Optimizations ──────────────────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# ── HF Cache ──────────────────────────────────────────────────────────────────
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
_CACHE_BACKUP = "/kaggle/working/hf-cache-backup"
if os.path.exists(_CACHE_BACKUP):
    os.environ["HF_HOME"] = _CACHE_BACKUP
    print("[Cache] Found local backup — skipping HuggingFace download.")

# ── Shared Voice Library Path ──────────────────────────────────────────────────
VOICES_DIR = "/kaggle/working/voice-assets/voices"

def get_characters():
    """Return {stem: path} for every .wav in the shared voice library."""
    wavs = glob.glob(os.path.join(VOICES_DIR, "*.wav"))
    return {Path(w).stem: w for w in sorted(wavs)}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — OmniVoice Engine
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("🔊 Loading OmniVoice (k2-fsa/OmniVoice)...")
print("=" * 60)

from omnivoice import OmniVoice, OmniVoiceGenerationConfig

OMNI_CHECKPOINT = "k2-fsa/OmniVoice"
omni_model = OmniVoice.from_pretrained(
    OMNI_CHECKPOINT,
    device_map="cuda",
    dtype=torch.float16,
    load_asr=True,
    token=False,
)
omni_sampling_rate = omni_model.sampling_rate
print(f"[OmniVoice] Loaded. Sampling rate: {omni_sampling_rate} Hz")
print(f"[OmniVoice] Shared voice library: {len(get_characters())} voices found")

import shutil

def _backup_cache():
    src = Path.home() / ".cache" / "huggingface"
    dst = Path(_CACHE_BACKUP)
    if src.exists() and not dst.exists():
        print("[Cache] Backing up HF cache for fast restarts...")
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print("[Cache] ✓ Backup done.")

threading.Thread(target=_backup_cache, daemon=True).start()


# ── OmniVoice: Generation ──────────────────────────────────────────────────────
LANGUAGES = [
    "Auto", "English (en)", "Chinese (zh)", "Japanese (ja)", "Korean (ko)",
    "French (fr)", "German (de)", "Spanish (es)", "Portuguese (pt)",
    "Russian (ru)", "Arabic (ar)", "Hindi (hi)", "Italian (it)",
    "Dutch (nl)", "Turkish (tr)", "Polish (pl)", "Swedish (sv)",
    "Thai (th)", "Vietnamese (vi)", "Indonesian (id)", "Malay (ms)",
]

CATEGORIES = {
    "Gender": ["male", "female"],
    "Age": ["child", "teenager", "young adult", "middle-aged", "elderly"],
    "Pitch": ["very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"],
    "Style": ["whisper"],
    "English Accent": [
        "american accent", "british accent", "australian accent",
        "canadian accent", "indian accent", "chinese accent",
        "japanese accent", "korean accent", "portuguese accent", "russian accent",
    ],
    "Chinese Dialect": ["四川话", "陕西话", "广东话", "东北话", "河南话", "云南话", "贵州话", "桂林话", "济南话"],
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


def generate_speech(text, language, ref_audio, instruct,
                    num_step, guidance_scale, denoise,
                    speed, duration, preprocess_prompt, postprocess_output,
                    mode="clone", ref_text=None):
    global _last_activity_time
    _last_activity_time = time.time()

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
        kw["voice_clone_prompt"] = omni_model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
        )

    if mode == "design" and instruct and instruct.strip():
        kw["instruct"] = instruct.strip()

    try:
        audio = omni_model.generate(**kw)
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"

    waveform = audio[0].squeeze()
    if hasattr(waveform, "numpy"):
        waveform = waveform.numpy()
    waveform = (waveform * 32767).astype(np.int16)
    return (omni_sampling_rate, waveform), f"Generated {waveform.shape[-1] / omni_sampling_rate:.1f}s audio at {omni_sampling_rate}Hz"


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


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Qwen3-TTS Engine
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("🎙️  Loading Qwen3-TTS (lazy — loads on first use to save VRAM)...")
print("=" * 60)

from qwen_tts import Qwen3TTSModel

QWEN_MODEL_MAP = {
    "base":   "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "custom": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}

# Qwen preset characters (built-in to the CustomVoice model)
QWEN_PRESET_CHARACTERS = [
    "Serena", "Vivian", "Ono_Anna", "Sohee",        # female
    "Aiden", "Dylan", "Eric", "Ryan", "Uncle_Fu",   # male
]

_qwen_current_model = None
_qwen_current_type = None


def _qwen_load_model(model_type: str):
    """Lazy-load Qwen model; swap if a different type is needed."""
    global _qwen_current_model, _qwen_current_type

    if _qwen_current_type == model_type:
        print(f"[Qwen3] Using cached '{model_type}' model")
        return _qwen_current_model

    if _qwen_current_model is not None:
        print(f"[Qwen3] Unloading '{_qwen_current_type}' model to free VRAM...")
        del _qwen_current_model
        _qwen_current_model = None
        gc.collect()
        torch.cuda.empty_cache()

    model_name = QWEN_MODEL_MAP[model_type]
    print(f"[Qwen3] Loading {model_name}...")
    t0 = time.time()
    try:
        _qwen_current_model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map="cuda:0",
            dtype=torch.float16,
            attn_implementation="sdpa",
        )
        _qwen_current_type = model_type
        allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
        print(f"[Qwen3] Loaded in {time.time() - t0:.1f}s | GPU: {allocated:.2f} GB")
        return _qwen_current_model
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Qwen3] ❌ Failed to load: {e}")
        return None


def _qwen_write_wav(wavs, sr) -> str:
    """Write Qwen output to a temp WAV file, return path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, wavs[0], sr)
    return tmp.name


def qwen_clone(text: str, character_name: str, ref_audio_path: str,
               ref_transcript: str, fast_mode: bool):
    """
    Qwen3 voice clone.
    - If character_name is set (one of the shared library voices), uses that WAV.
    - If ref_audio_path is provided directly (custom upload), uses that instead.
    - fast_mode = True skips transcript for speed.
    """
    global _last_activity_time
    _last_activity_time = time.time()

    if not text or not text.strip():
        return None, "Please enter text."

    # Resolve reference audio: shared library takes priority over dropdown
    ref_path = ref_audio_path  # custom upload
    if character_name:
        chars = get_characters()
        if character_name in chars:
            ref_path = chars[character_name]
            print(f"[Qwen3-Clone] Using shared voice: {character_name} → {ref_path}")

    if not ref_path:
        return None, "Please select a shared voice or upload a reference audio."

    try:
        t0 = time.time()
        model = _qwen_load_model("base")
        if model is None:
            return None, "Qwen3 base model failed to load."

        if fast_mode or not ref_transcript or not ref_transcript.strip():
            prompt_items = model.create_voice_clone_prompt(
                ref_audio=ref_path, x_vector_only_mode=True
            )
        else:
            prompt_items = model.create_voice_clone_prompt(
                ref_audio=ref_path,
                ref_text=ref_transcript.strip(),
                x_vector_only_mode=False,
            )

        with torch.inference_mode():
            wavs, sr = model.generate_voice_clone(text=text.strip(), voice_clone_prompt=prompt_items)

        out_path = _qwen_write_wav(wavs, sr)
        dur = len(wavs[0]) / sr
        gc.collect(); torch.cuda.empty_cache()
        return out_path, f"✅ Qwen3 clone — {dur:.1f}s audio in {time.time() - t0:.1f}s"

    except Exception as e:
        import traceback; traceback.print_exc()
        return None, f"❌ Qwen3 clone error: {e}"


def qwen_custom(text: str, voice_name: str, instruction: str):
    """Qwen3 preset character voice (9 built-in voices)."""
    global _last_activity_time
    _last_activity_time = time.time()

    if not text or not text.strip():
        return None, "Please enter text."

    try:
        t0 = time.time()
        model = _qwen_load_model("custom")
        if model is None:
            return None, "Qwen3 custom model failed to load."

        with torch.inference_mode():
            if instruction and instruction.strip():
                wavs, sr = model.generate_custom_voice(
                    text=text.strip(), speaker=voice_name, instruct=instruction.strip()
                )
            else:
                wavs, sr = model.generate_custom_voice(text=text.strip(), speaker=voice_name)

        out_path = _qwen_write_wav(wavs, sr)
        dur = len(wavs[0]) / sr
        gc.collect(); torch.cuda.empty_cache()
        return out_path, f"✅ Qwen3 custom ({voice_name}) — {dur:.1f}s in {time.time() - t0:.1f}s"

    except Exception as e:
        import traceback; traceback.print_exc()
        return None, f"❌ Qwen3 custom error: {e}"


def qwen_design(text: str, voice_description: str):
    """Qwen3 voice design from text description."""
    global _last_activity_time
    _last_activity_time = time.time()

    if not text or not text.strip():
        return None, "Please enter text."
    if not voice_description or not voice_description.strip():
        return None, "Please describe the voice."

    try:
        t0 = time.time()
        model = _qwen_load_model("design")
        if model is None:
            return None, "Qwen3 design model failed to load."

        with torch.inference_mode():
            wavs, sr = model.generate_voice_design(
                text=text.strip(), instruct=voice_description.strip()
            )

        out_path = _qwen_write_wav(wavs, sr)
        dur = len(wavs[0]) / sr
        gc.collect(); torch.cuda.empty_cache()
        return out_path, f"✅ Qwen3 design — {dur:.1f}s in {time.time() - t0:.1f}s"

    except Exception as e:
        import traceback; traceback.print_exc()
        return None, f"❌ Qwen3 design error: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MFA Forced Alignment (unchanged from OmniVoice notebook)
# ═══════════════════════════════════════════════════════════════════════════════
_ALIGNER_MODEL = None
_TARGET_SR = 16000


def _resample_to_16k(data: np.ndarray, sr: int) -> np.ndarray:
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
    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(data) // frame_len
    silences, in_silence, gap_start = [], False, 0.0
    for i in range(n_frames):
        chunk = data[i * frame_len: (i + 1) * frame_len]
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        t = i * frame_ms / 1000.0
        if rms < rms_threshold:
            if not in_silence:
                in_silence, gap_start = True, t
        else:
            if in_silence:
                in_silence = False
                if t - gap_start >= min_silence_sec:
                    silences.append({"start": round(gap_start, 3), "end": round(t, 3)})
    if in_silence:
        t_end = n_frames * frame_ms / 1000.0
        if t_end - gap_start >= min_silence_sec:
            silences.append({"start": round(gap_start, 3), "end": round(t_end, 3)})
    return silences


_JP_PARTICLES = {
    'が', 'は', 'を', 'に', 'で', 'と', 'も', 'か', 'や', 'な', 'の',
    'へ', 'より', 'から', 'まで', 'て', 'で', 'ば', 'し',
    'た', 'だ', 'な', 'ね', 'よ', 'わ', 'さ', 'ぞ', 'ぜ',
}
_KATAKANA_RE = re.compile(r'^[\u30A0-\u30FF\u30FC]+$')


def _katakana_to_romaji(surface: str, katsu) -> str:
    r = katsu.romaji(surface)
    r_clean = re.sub(r"[^A-Za-z\'\-]", " ", r).strip()
    kana_chars = len([c for c in surface if ord(c) >= 0x30A0 and ord(c) <= 0x30FF])
    if kana_chars > 0 and len(r_clean.replace(" ", "")) < kana_chars:
        r = katsu.romaji(surface, ensure_ascii=False)
    return r


def _tokenize_with_index(text: str, cutlet_module, fugashi_module):
    tagger = fugashi_module.Tagger()
    katsu = cutlet_module.Cutlet()
    raw_tokens = []
    for word in tagger(text.strip()):
        surface = word.surface
        if not surface.strip():
            continue
        if _KATAKANA_RE.match(surface):
            r = katsu.romaji(surface)
            r_words = re.sub(r"[^A-Za-z\'\-]", " ", r).split()
            if any(len(w) > 6 for w in r_words) and len(surface) <= 4:
                kana_map_digraph = {
                    "キャ": "kya", "キュ": "kyu", "キョ": "kyo",
                    "シャ": "sha", "シュ": "shu", "ショ": "sho",
                    "チャ": "cha", "チュ": "chu", "チョ": "cho",
                    "ニャ": "nya", "ニュ": "nyu", "ニョ": "nyo",
                    "ヒャ": "hya", "ヒュ": "hyu", "ヒョ": "hyo",
                    "ミャ": "mya", "ミュ": "myu", "ミョ": "myo",
                    "リャ": "rya", "リュ": "ryu", "リョ": "ryo",
                    "ギャ": "gya", "ギュ": "gyu", "ギョ": "gyo",
                    "ジャ": "ja",  "ジュ": "ju",  "ジョ": "jo",
                    "ビャ": "bya", "ビュ": "byu", "ビョ": "byo",
                    "ピャ": "pya", "ピュ": "pyu", "ピョ": "pyo",
                    "ファ": "fa",  "フィ": "fi",  "フェ": "fe",  "フォ": "fo",
                    "ティ": "ti",  "ディ": "di",  "トゥ": "tu",  "ドゥ": "du",
                    "ウィ": "wi",  "ウェ": "we",  "ウォ": "wo",
                    "ヴァ": "va",  "ヴィ": "vi",  "ヴ": "vu",   "ヴェ": "ve",  "ヴォ": "vo",
                }
                kana_map = {
                    "ア": "a", "イ": "i", "ウ": "u", "エ": "e", "オ": "o",
                    "カ": "ka", "キ": "ki", "ク": "ku", "ケ": "ke", "コ": "ko",
                    "サ": "sa", "シ": "shi", "ス": "su", "セ": "se", "ソ": "so",
                    "タ": "ta", "チ": "chi", "ツ": "tsu", "テ": "te", "ト": "to",
                    "ナ": "na", "ニ": "ni", "ヌ": "nu", "ネ": "ne", "ノ": "no",
                    "ハ": "ha", "ヒ": "hi", "フ": "fu", "ヘ": "he", "ホ": "ho",
                    "マ": "ma", "ミ": "mi", "ム": "mu", "メ": "me", "モ": "mo",
                    "ヤ": "ya", "ユ": "yu", "ヨ": "yo",
                    "ラ": "ra", "リ": "ri", "ル": "ru", "レ": "re", "ロ": "ro",
                    "ワ": "wa", "ヲ": "wo", "ン": "n",
                    "ガ": "ga", "ギ": "gi", "グ": "gu", "ゲ": "ge", "ゴ": "go",
                    "ザ": "za", "ジ": "ji", "ズ": "zu", "ゼ": "ze", "ゾ": "zo",
                    "ダ": "da", "デ": "de", "ド": "do",
                    "バ": "ba", "ビ": "bi", "ブ": "bu", "ベ": "be", "ボ": "bo",
                    "パ": "pa", "ピ": "pi", "プ": "pu", "ペ": "pe", "ポ": "po",
                    "ッ": "t",
                }
                result_r = ""
                i_k = 0
                while i_k < len(surface):
                    two = surface[i_k:i_k + 2]
                    if two in kana_map_digraph:
                        result_r += kana_map_digraph[two]
                        i_k += 2
                    elif surface[i_k] == "ー":
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
            continue
        raw_tokens.append((surface, words))

    if not raw_tokens:
        return [], []

    merged = []
    for surface, words in raw_tokens:
        if surface in _JP_PARTICLES and merged:
            prev_surface, prev_words = merged[-1]
            merged[-1] = (prev_surface + surface, prev_words + words)
        else:
            merged.append((surface, words))

    token_pairs = [(s, len(w)) for s, w in merged]
    romaji_words = [w for _, words in merged for w in words]
    return token_pairs, romaji_words


def align_japanese(audio_path: str, japanese_text: str, translation_map_json: str = "") -> str:
    global _last_activity_time
    _last_activity_time = time.time()

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
        token_pairs, romaji_words = _tokenize_with_index(japanese_text, cutlet, fugashi)
        if not romaji_words:
            return _json.dumps({"error": "Japanese Script produced no romanizable tokens."}, ensure_ascii=False)

        jp_tokens = [jp for jp, _ in token_pairs]
        jp_counts = [cnt for _, cnt in token_pairs]
        romaji_transcript = " ".join(romaji_words)
        print(f"[Align] JP tokens : {jp_tokens}")
        print(f"[Align] jp_counts : {jp_counts}  (sum={sum(jp_counts)}, words={len(romaji_words)})")
        print(f"[Align] Romaji    : {romaji_transcript}")

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_data, raw_sr = sf.read(audio_path)
            if raw_data.ndim > 1:
                raw_data = raw_data.mean(axis=1)
            audio_16k = _resample_to_16k(raw_data.astype(np.float32), raw_sr)
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

        count_sum = sum(jp_counts)
        tolerance = max(2, round(count_sum * 0.10))
        count_ok = abs(count_sum - len(stamps)) <= tolerance
        errors = None if count_ok else (
            f"jp_counts sum ({count_sum}) != romaji stamp count ({len(stamps)}) "
            f"(delta={abs(count_sum - len(stamps))}, tolerance={tolerance}). "
            "The app should fall back to raw stamps."
        )
        if errors:
            print(f"[Align] WARNING: {errors}")
        elif count_sum != len(stamps):
            delta = len(stamps) - count_sum
            print(f"[Align] INFO: Adjusting jp_counts by {delta} (within tolerance)")
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


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Shared Session State
# ═══════════════════════════════════════════════════════════════════════════════
SESSION_START_TIME = int(time.time() * 1000)   # ms, set once at boot
_last_activity_time = time.time()
IDLE_TIMEOUT_SEC = 10 * 60   # 10 min auto-kill
IDLE_WARN_SEC    =  8 * 60   # warn at 8 min

RELAY_URL = "https://omnivoice-relay.zsage84869.workers.dev/register"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Gradio UI
# ═══════════════════════════════════════════════════════════════════════════════
import gradio as gr
import requests

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.gradio-container { max-width: 1060px !important; margin: auto !important; }
.brand-header { text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 28px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 10px 25px rgba(102,126,234,0.3); }
.brand-title { color: white; font-size: 2em; font-weight: 700; margin: 0 0 6px 0; }
.brand-subtitle { color: rgba(255,255,255,0.88); font-size: 1em; margin-bottom: 12px; }
.hint { color: rgba(255,255,255,0.84); font-size: 0.92em; }
button.primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; color: white !important; font-weight: 600 !important; border-radius: 12px !important; }
.engine-tag { display:inline-block; padding:2px 8px; border-radius:6px; font-size:0.8em; font-weight:600; margin-left:6px; }
.tag-omni { background:#e8f4fd; color:#1a7bbf; }
.tag-qwen { background:#fde8f4; color:#bf1a7b; }
"""

BRAND_HTML = """
<div class="brand-header">
  <div class="brand-title">🎙️ cinEZma TTS Studio</div>
  <div class="brand-subtitle">OmniVoice + Qwen3-TTS · Unified Voice Library · MFA Alignment</div>
  <div class="hint">
    jp_charon and all shared voices work across both engines &nbsp;·&nbsp;
    Same relay, same API surface as before
  </div>
</div>
"""


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


def refresh_characters():
    chars = get_characters()
    choices = list(chars.keys())
    return gr.update(choices=choices, value=choices[0] if choices else None)


with gr.Blocks(title="cinEZma TTS Studio", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.HTML(BRAND_HTML)

    with gr.Tabs():

        # ── Tab: OmniVoice — Preset Voices ──────────────────────────────────
        with gr.TabItem("🎭 OmniVoice — Preset Voices"):
            gr.Markdown(
                "Clone one of your **shared voice library** files (jp_charon, etc.).\n\n"
                "<span class='engine-tag tag-omni'>OmniVoice</span> Japanese-optimised, great for anime narration."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    characters = get_characters()
                    char_choices = list(characters.keys())
                    preset_name = gr.Dropdown(
                        label="Preset Character",
                        choices=char_choices,
                        value=char_choices[0] if char_choices else None,
                        info="Pulled from /voice-assets/voices/",
                    )
                    gr.Button("Refresh Voices", size="sm").click(refresh_characters, outputs=[preset_name])
                    pv_text = gr.Textbox(label="Text to Synthesize", lines=4, placeholder="Enter text here...")
                    pv_ref_text = gr.Textbox(label="Reference Text (optional)", lines=2,
                                             placeholder="Transcript of preset clip. Leave blank for auto.")
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
                api_name="generate_preset",   # ← cinEZma uses this
            )

        # ── Tab: OmniVoice — Custom Clone ───────────────────────────────────
        with gr.TabItem("🔬 OmniVoice — Custom Clone"):
            gr.Markdown(
                "Upload any reference clip to clone.\n\n"
                "<span class='engine-tag tag-omni'>OmniVoice</span>"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    vc_text = gr.Textbox(label="Text to Synthesize", lines=4,
                                         placeholder="You can use tags like [laughter], [sigh], etc.")
                    vc_ref_audio = gr.Audio(label="Reference Audio (3-10s recommended)", type="filepath")
                    vc_ref_text = gr.Textbox(label="Reference Text (optional)", lines=2,
                                             placeholder="Transcript of ref audio. Leave blank for auto.")
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
                api_name="generate_custom",   # ← cinEZma uses this
            )

        # ── Tab: OmniVoice — Voice Design ───────────────────────────────────
        with gr.TabItem("🎨 OmniVoice — Voice Design"):
            gr.Markdown("<span class='engine-tag tag-omni'>OmniVoice</span> Describe the voice you want.")
            with gr.Row():
                with gr.Column(scale=1):
                    vd_text = gr.Textbox(label="Text to Synthesize", lines=4, placeholder="Enter text here...")
                    vd_lang = lang_dropdown()
                    vd_groups = []
                    for cat, choices in CATEGORIES.items():
                        vd_groups.append(
                            gr.Dropdown(label=cat, choices=["Auto"] + choices, value="Auto",
                                        info=ATTR_INFO.get(cat))
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

        # ── Tab: Qwen3 — Voice Clone (shared library + upload) ──────────────
        with gr.TabItem("🎤 Qwen3 — Voice Clone"):
            gr.Markdown(
                "Clone any voice using **Qwen3-TTS Base**.\n\n"
                "<span class='engine-tag tag-qwen'>Qwen3-TTS</span> "
                "Select a shared voice (jp_charon, etc.) **or** upload a reference clip. "
                "Shared voice takes priority when both are provided."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    qc_text = gr.Textbox(label="Text to Synthesize", lines=4, placeholder="Enter text...")
                    characters2 = get_characters()
                    char_choices2 = ["(none — use upload below)"] + list(characters2.keys())
                    qc_char = gr.Dropdown(
                        label="Shared Voice Library",
                        choices=char_choices2,
                        value=char_choices2[0],
                        info="Shared with OmniVoice — jp_charon, etc.",
                    )
                    gr.Button("Refresh Voices", size="sm").click(
                        lambda: gr.update(choices=["(none — use upload below)"] + list(get_characters().keys())),
                        outputs=[qc_char],
                    )
                    qc_upload = gr.Audio(label="Or Upload Reference Audio (3+ sec)", type="filepath")
                    qc_transcript = gr.Textbox(label="Transcript of Reference (optional)", lines=2,
                                               placeholder="What's said in the reference clip...")
                    qc_fast = gr.Checkbox(label="Fast Mode (skip transcript)", value=True)
                    qc_btn = gr.Button("Generate", variant="primary", size="lg")
                with gr.Column(scale=1):
                    qc_audio = gr.Audio(label="Generated Audio")
                    qc_status = gr.Textbox(label="Status", lines=2)
                    gr.Markdown("**First run:** ~2-3 min (model load) · RTF ~3.5-5x")

            def _qc_resolve_char(char_val):
                """Return None if placeholder selected, else the voice name."""
                return None if char_val == "(none — use upload below)" else char_val

            qc_btn.click(
                lambda text, char, upload, transcript, fast: qwen_clone(
                    text, _qc_resolve_char(char), upload, transcript, fast
                ),
                inputs=[qc_text, qc_char, qc_upload, qc_transcript, qc_fast],
                outputs=[qc_audio, qc_status],
                api_name="qwen_clone",   # ← cinEZma can use this
            )

        # ── Tab: Qwen3 — Built-in Character Voices ──────────────────────────
        with gr.TabItem("🎭 Qwen3 — Character Voices"):
            gr.Markdown(
                "9 built-in preset characters with optional style control.\n\n"
                "<span class='engine-tag tag-qwen'>Qwen3-TTS CustomVoice</span> "
                "Female: Serena, Vivian, Ono_Anna, Sohee · Male: Aiden, Dylan, Eric, Ryan, Uncle_Fu"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    qq_text = gr.Textbox(label="Text to Synthesize", lines=4, placeholder="Enter text...")
                    qq_voice = gr.Dropdown(
                        choices=QWEN_PRESET_CHARACTERS,
                        label="Voice Character",
                        value="Serena",
                    )
                    qq_instruction = gr.Textbox(
                        label="Style Instruction (optional)",
                        placeholder="e.g. 'speak slowly and cheerfully'",
                        lines=2,
                    )
                    qq_btn = gr.Button("Generate", variant="primary", size="lg")
                with gr.Column(scale=1):
                    qq_audio = gr.Audio(label="Generated Audio")
                    qq_status = gr.Textbox(label="Status", lines=2)

            qq_btn.click(
                qwen_custom,
                inputs=[qq_text, qq_voice, qq_instruction],
                outputs=[qq_audio, qq_status],
                api_name="qwen_custom",   # ← cinEZma can use this
            )

        # ── Tab: Qwen3 — Voice Design ────────────────────────────────────────
        with gr.TabItem("✏️ Qwen3 — Voice Design"):
            gr.Markdown(
                "Design a voice from a text description.\n\n"
                "<span class='engine-tag tag-qwen'>Qwen3-TTS VoiceDesign</span> "
                "Describe age, gender, emotion, accent, speed."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    qd_text = gr.Textbox(label="Text to Synthesize", lines=4, placeholder="Enter text...")
                    qd_desc = gr.Textbox(
                        label="Voice Description",
                        lines=3,
                        placeholder="A young female, cheerful, speaking clearly with British accent",
                    )
                    qd_btn = gr.Button("Generate", variant="primary", size="lg")
                with gr.Column(scale=1):
                    qd_audio = gr.Audio(label="Generated Audio")
                    qd_status = gr.Textbox(label="Status", lines=2)

            qd_btn.click(
                qwen_design,
                inputs=[qd_text, qd_desc],
                outputs=[qd_audio, qd_status],
                api_name="qwen_design",   # ← cinEZma can use this
            )

        # ── Tab: MFA Align ───────────────────────────────────────────────────
        with gr.TabItem("📐 MFA Align"):
            gr.Markdown(
                "CTC forced alignment — Japanese audio → romaji timestamps → subtitle cards.\n\n"
                "Works with audio from **either engine**."
            )
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
                api_name="align_japanese",   # ← cinEZma uses this
            )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Launch + Relay + Watchdog + Cache Warm
# ═══════════════════════════════════════════════════════════════════════════════

def _play_ready_chime():
    try:
        from IPython.display import display, Javascript
        display(Javascript("""
        (function() {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
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
    for _ in range(90):
        url = getattr(demo, "share_url", None)
        if url and "gradio.live" in str(url):
            url = str(url).strip()
            print(f"[Relay] Got URL: {url}")
            for attempt in range(1, 4):
                try:
                    r = requests.post(
                        RELAY_URL,
                        json={"url": url, "started_at": SESSION_START_TIME},
                        timeout=15,
                    )
                    if r.status_code == 200:
                        print(f"[Relay] ✓ Registered (attempt {attempt})")
                        _play_ready_chime()
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


def _idle_watcher():
    warned = False
    while True:
        time.sleep(60)
        idle = time.time() - _last_activity_time
        remaining = IDLE_TIMEOUT_SEC - idle
        if idle >= IDLE_TIMEOUT_SEC:
            print(f"[IdleTimer] ⏹  No activity for {IDLE_TIMEOUT_SEC // 60} min — shutting down.")
            demo.close()
            os._exit(0)
        elif idle >= IDLE_WARN_SEC and not warned:
            warned = True
            print(f"[IdleTimer] ⚠  Idle for {int(idle // 60)} min. "
                  f"Auto-kill in ~{int(remaining // 60)} min if no requests arrive.")
        elif idle < IDLE_WARN_SEC:
            warned = False


threading.Thread(target=_idle_watcher, daemon=True).start()


def _warm_cache():
    print("[Cache] Waiting for server before warming cache...")
    for _ in range(90):
        if getattr(demo, "share_url", None):
            break
        time.sleep(2)
    else:
        print("[Cache] ⚠ Server never came up — skipping cache warm.")
        return

    print("[Cache] Server is up. Starting background cache warm-up...")
    try:
        global _ALIGNER_MODEL
        if _ALIGNER_MODEL is None:
            from ctc_forced_aligner import get_word_stamps
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

        print("[Cache] ✓ Cache warm-up complete.")
    except Exception as e:
        print(f"[Cache] ⚠ Warm-up error (non-fatal): {e}")


threading.Thread(target=_warm_cache, daemon=True).start()

demo.queue()
print("=" * 60)
print("🚀 Launching cinEZma TTS Studio (OmniVoice + Qwen3-TTS)...")
print("   Relay:", RELAY_URL)
print("=" * 60)
demo.launch(share=True, debug=True, theme=gr.themes.Soft(), css=CSS)
