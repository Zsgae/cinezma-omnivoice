#!/usr/bin/env python3
"""
OmniVoice + WriterBot - cinEZma Edition
Auto-pulled from GitHub by the Kaggle launcher notebook.
Edit this file and push to GitHub — changes take effect on next Kaggle restart.

WriterBot uses Qwen3-14B (4-bit) loaded lazily on first use.
Recommended accelerator: T4x2 (32GB VRAM) for both models running together.
P100 (16GB) will OOM — if you're on P100, don't click "Load WriterBot" until
you're done with TTS for the session.
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


# ── WriterBot: Qwen3-14B (lazy load) ─────────────────────────────────────────
# Loaded on first use so OmniVoice startup never OOMs.
# On T4x2 (32GB) both models run together fine.
# On P100 (16GB) only use one at a time per session.

_qwen_model     = None
_qwen_tokenizer = None
_qwen_loading   = False
QWEN_CHECKPOINT = "Qwen/Qwen3-14B"

def _load_qwen():
    """Lazy-load Qwen3-14B in 4-bit. Called once on first WriterBot generate."""
    global _qwen_model, _qwen_tokenizer, _qwen_loading
    if _qwen_model is not None:
        return True, "Already loaded."
    if _qwen_loading:
        return False, "Still loading, please wait..."

    _qwen_loading = True
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        print("[WriterBot] Loading Qwen3-14B (4-bit)...")

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        _qwen_tokenizer = AutoTokenizer.from_pretrained(
            QWEN_CHECKPOINT,
            trust_remote_code=True,
        )
        _qwen_model = AutoModelForCausalLM.from_pretrained(
            QWEN_CHECKPOINT,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
        )
        _qwen_model.eval()
        print("[WriterBot] ✓ Qwen3-14B loaded.")
        return True, "✓ WriterBot model loaded. Ready to generate."
    except Exception as e:
        import traceback
        msg = f"Failed to load Qwen: {e}\n{traceback.format_exc()[-800:]}"
        print(f"[WriterBot] ✗ {msg}")
        return False, msg
    finally:
        _qwen_loading = False


# ── WriterBot: Prompt System ───────────────────────────────────────────────────
# 6-layer cinEZma system built on top of the Manhwa Fresh formula.
# Layers stack in order: 0 → 1 → 2 → 3 → 4 → 5 → [your script]

_PROMPT_0_THEMATIC = """
Before you write anything, silently identify three anchors from the script:
1. The core emotional wound of the MC in this chapter (what are they really losing or gaining?)
2. The single line of dialogue that carries the most weight
3. The one image or moment that would stick in someone's mind after watching

Hold these three anchors in mind throughout everything. They are what you cannot cut,
no matter how aggressive the compression. Do not name or label these anchors in your output.
They are invisible scaffolding only.
""".strip()

_PROMPT_1_COMPRESSION = """
You are rewriting a manhwa recap script for a YouTube channel. The original script is too long,
too descriptive, and reads like a novel instead of someone telling you what happened.

Your job is to turn it into something that sounds like a friend recapping the chapter to you.
Casual. Fast. Easy to listen to. Every plot beat from the original stays. Nothing new gets added.

Here is how this voice tends to behave:

It does not narrate what is already happening on screen. If the viewer can see something happen,
the voiceover moves the plot forward instead of describing the visual.

It cuts atmosphere. Weather, lighting, smells, ambient mood — all of that goes away unless
it actually matters for the plot.

It compresses dialogue. Back-and-forth quoted conversations get summarized into a sentence
or two of natural narration. No "he said," no "she replied."

It compresses overall length. The final script should land around a third of the original
word count. The point is information per second, not word count for its own sake.

It refers to characters in a relaxed way. The protagonist's full name shows up once at the
start and then almost never again. After that, casual stand-ins that feel natural.

It leans on comparison instead of description. When something evokes a familiar feeling,
point at something the audience already knows rather than describing it from scratch.

It carries personality through small asides, not big jokes. Ironic understatement, dry
observation — sprinkled in naturally, never forced.

It uses transitions that flow. No mechanical "and then" / "suddenly" / "meanwhile" repetition.

It sounds spoken, not written. Short clean sentences that flow easily read aloud.
No em-dashes. No semicolons. No nested clauses.

THE HARD RULES — cannot be broken:
Never invent plot points. Every event has to exist in the original script.
Never delete plot points to hit a word count. Compress them instead.
Output as a single flowing block of prose. No headers, bullets, or markdown formatting.
""".strip()

_GENRE_MODULES = {
    "Isekai / Power Fantasy": """
The story is an isekai or power fantasy manhwa. The audience is young adult anime fans who
know the genre conventions inside out. Lean into the genre vocabulary: Truck-kun, system
notifications, OP characters, regression, leveling up. The energy is hyped, casual, slightly
chaotic. The MC is the hero in the audience's eyes — root for him quietly in the narration.
Comparisons come from action movies, superhero franchises, gaming references: Marvel, Superman,
Dark Souls, Naruto, Dragon Ball, Solo Leveling. Swearing stays PG-13 — "damn," "effing" can
land for emphasis on big moments, sparingly.
""".strip(),

    "Romance / Emotional Drama": """
The story is a romance or emotional drama manhwa. The energy is warmer and more invested
than a power fantasy recap. Sound like a friend recapping a show they're emotionally caught
up in. Refer to main characters by their real names more often. Asides tend to be relatable
reactions: "you can tell she's about to spiral," "this is the moment everything shifts."
Comparisons come from other romance media, K-drama tropes, TikTok relationship language:
"this is giving second-lead syndrome," "classic enemies-to-lovers arc." Swearing is rare.
The voice stays warm and accessible.
""".strip(),

    "Dark Action / Revenge": """
The story is a dark action or revenge manhwa. The tone is heavier, the stakes are serious.
Still casual, still conversational, but with more restraint. Asides land harder when they're
rare — silence and short sentences carry weight here. In the heavy beats — a death, a betrayal,
a long-awaited revenge — use the real name instead of "bro." It hits harder. Asides lean toward
ominous understatement: "which is the last time we see him doing that," "nobody walks away
from this one clean." Comparisons match the tone: John Wick, Berserk, The Boys, Squid Game.
Swearing slightly stronger here, still PG-13, still sparingly.
""".strip(),

    "Comedy / Slice of Life": """
The story is a comedy or slice-of-life manhwa. The energy is the lightest of all the niches.
Match the source — playful, casual, leaning into the comedy beats already in the story.
Use "bro," "my guy," or real names freely. Asides tend to be observational, slightly absurd,
genuinely amused: "this is somehow the seventh time this has happened to him." Let the humor
of the source carry it. Comparisons can be playful: sitcoms, comedic anime tropes, social
media humor. Swearing is rare and only for comedic emphasis.
""".strip(),
}

_ARC_MODULES = {
    "Chapter 1 — Establish (hook needed)": """
This is the first chapter of a new manhwa — the opening line carries more weight than any
other line in the video. Write the opening sentence so the viewer knows within ten words
exactly what kind of story they're about to hear: the protagonist, the situation, the genre,
the vibe. The opening should feel like the punchline you'd give if a friend asked "so what's
this manhwa about?" and you only had one breath to answer. Not a setup. Not atmosphere.
Just the core of the story landing immediately. Avoid: scenery, philosophical observations,
slow buildups, anything that delays naming what's actually happening.
""".strip(),

    "Mid-arc — Tension (keep momentum)": """
The audience already knows the premise and the characters. Skip re-establishing anything
they know. Jump straight into what changed THIS chapter. The energy should feel like
escalation — something shifted, something was revealed, something got worse or more
complicated. Keep the tension alive. End on a beat that makes the next chapter feel necessary.
""".strip(),

    "Climax — Payoff (deliver the hit)": """
This is a climax or payoff chapter. The audience has been waiting for this. Every word of
the narration should feel like it earns the wait. Don't rush the big moment — give it room
to land. Short sentences. White space in the pacing. Let the weight hit before moving on.
After the peak, land the consequences clearly so the emotional payoff registers fully.
""".strip(),
}

_PROMPT_HOOK = """
After the script ends, add ONE final sentence — a cliffhanger or retention hook.
This is NOT a sign-off or outro. It's a single punchy sentence that makes the viewer
feel physically unable to not watch the next chapter. It should feel like the story
interrupted itself at the worst possible moment. Do not explain it. Do not frame it.
Just deliver it and stop.
""".strip()


def _build_writerbot_prompt(
    script: str,
    genre: str,
    arc_position: str,
    style_dna: str,
) -> str:
    """Stack all 6 layers into one prompt in the correct order."""
    parts = []

    # Layer 0 — Thematic anchor (invisible scaffolding)
    parts.append(_PROMPT_0_THEMATIC)

    # Layer 1 — Universal compression engine
    parts.append(_PROMPT_1_COMPRESSION)

    # Layer 2 — Genre module
    genre_text = _GENRE_MODULES.get(genre, _GENRE_MODULES["Isekai / Power Fantasy"])
    parts.append(genre_text)

    # Layer 3 — Arc/stakes position module
    arc_text = _ARC_MODULES.get(arc_position, _ARC_MODULES["Mid-arc — Tension (keep momentum)"])
    parts.append(arc_text)

    # Layer 4 — Style DNA (user's own prose voice)
    if style_dna and style_dna.strip():
        parts.append(
            f"One more thing before the script. Here are examples of the exact voice "
            f"and writing style you must mirror. Study the rhythm, the sentence length, "
            f"the personality. This IS the voice of the channel:\n\n{style_dna.strip()}"
        )

    # Layer 5 — Cliffhanger/retention hook
    parts.append(_PROMPT_HOOK)

    # The script itself
    parts.append(f"Now here is the script. Go.\n\n---\n\n{script.strip()}")

    return "\n\n---\n\n".join(parts)


def generate_writerbot(script, genre, arc_position, style_dna, max_tokens, temperature):
    """Main WriterBot generate function called by Gradio."""
    global _last_activity_time
    _last_activity_time = time.time()

    if not script or not script.strip():
        return "", "⚠ Please paste your draft script first."

    # Lazy-load Qwen if not already loaded
    if _qwen_model is None:
        ok, msg = _load_qwen()
        if not ok:
            return "", f"✗ Model load failed: {msg}"

    try:
        full_prompt = _build_writerbot_prompt(script, genre, arc_position, style_dna)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional manhwa recap writer for a YouTube channel. "
                    "You follow instructions precisely and output only the final script prose — "
                    "no preamble, no labels, no markdown, no commentary about what you're doing. "
                    "Just the finished script, flowing as a single block of spoken prose."
                ),
            },
            {"role": "user", "content": full_prompt},
        ]

        text = _qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = _qwen_tokenizer([text], return_tensors="pt").to(_qwen_model.device)

        with torch.no_grad():
            output_ids = _qwen_model.generate(
                **inputs,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                do_sample=float(temperature) > 0,
                top_p=0.92,
                repetition_penalty=1.1,
                pad_token_id=_qwen_tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        result = _qwen_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        word_count = len(result.split())
        orig_count = len(script.split())
        ratio = round(orig_count / max(word_count, 1), 1)

        status = (
            f"✓ Generated {word_count} words "
            f"(original: {orig_count} words, compression: {ratio}x)"
        )
        return result, status

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "", (
            "✗ CUDA OOM — you may be on P100 (16GB) with OmniVoice already loaded. "
            "Either switch to T4x2 accelerator or restart and load WriterBot first."
        )
    except Exception as e:
        import traceback
        return "", f"✗ Error: {e}\n{traceback.format_exc()[-600:]}"


def load_writerbot_model():
    """Called by the Load Model button in the UI."""
    ok, msg = _load_qwen()
    return msg


# ── Gradio UI + Relay ─────────────────────────────────────────────────────────
import gradio as gr
import requests
import time

SESSION_START_TIME = int(time.time() * 1000)

IDLE_TIMEOUT_SEC = 10 * 60
_last_activity_time = time.time()

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
.writerbot-note { background: rgba(102,126,234,0.08); border-left: 3px solid #667eea; padding: 10px 14px; border-radius: 6px; font-size: 0.88em; color: #555; margin-bottom: 8px; }
"""

BRAND_HTML = """
<div class="brand-header">
  <div class="brand-title">OmniVoice + WriterBot</div>
  <div class="brand-subtitle">cinEZma Edition</div>
  <div class="hint">TTS · Alignment · Script Generation — all on one Kaggle session</div>
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
    _last_activity_time = time.time()

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
            return _json.dumps({
                "error": "Japanese Script produced no romanizable tokens."
            }, ensure_ascii=False)

        jp_tokens  = [jp  for jp,  _ in token_pairs]
        jp_counts  = [cnt for _,  cnt in token_pairs]
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
        count_ok  = abs(count_sum - len(stamps)) <= tolerance
        errors    = None if count_ok else (
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
    katsu  = cutlet_module.Cutlet()

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
                    "ッ":"t",
                }
                result_r = ""
                i_k = 0
                while i_k < len(surface):
                    two = surface[i_k:i_k+2]
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

    token_pairs  = [(s, len(w)) for s, w in merged]
    romaji_words = [w for _, words in merged for w in words]

    return token_pairs, romaji_words


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="cinEZma — OmniVoice + WriterBot") as demo:
    gr.HTML(BRAND_HTML)

    with gr.Tabs():

        # ── Tab 1: Character Voices ──────────────────────────────────────────
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
                    pv_text = gr.Textbox(label="Text to Synthesize", lines=4, placeholder="Enter text here...")
                    pv_ref_text = gr.Textbox(
                        label="Reference Text (optional)", lines=2,
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

        # ── Tab 2: Custom Voice Clone ────────────────────────────────────────
        with gr.TabItem("Custom Voice Clone"):
            with gr.Row():
                with gr.Column(scale=1):
                    vc_text = gr.Textbox(
                        label="Text to Synthesize", lines=4,
                        placeholder="Enter text here... You can use tags like [laughter], [sigh], etc.",
                    )
                    vc_ref_audio = gr.Audio(label="Reference Audio (3-10s recommended)", type="filepath")
                    vc_ref_text = gr.Textbox(
                        label="Reference Text (optional)", lines=2,
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

        # ── Tab 3: Voice Design ──────────────────────────────────────────────
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

        # ── Tab 4: MFA Align ─────────────────────────────────────────────────
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

        # ── Tab 5: WriterBot ─────────────────────────────────────────────────
        with gr.TabItem("✍️ WriterBot"):
            gr.HTML("""
            <div class="writerbot-note">
              <strong>Powered by Qwen3-14B (4-bit)</strong> — loads separately from OmniVoice.<br>
              <strong>T4x2 accelerator recommended</strong> (32GB VRAM handles both models).
              On P100 (16GB), only load one model per session.
            </div>
            """)

            with gr.Row():
                wb_load_btn = gr.Button("⚡ Load WriterBot Model", variant="primary", size="sm")
                wb_load_status = gr.Textbox(label="Model Status", lines=1, value="Not loaded — click to load.")
            wb_load_btn.click(load_writerbot_model, outputs=[wb_load_status])

            gr.Markdown("---")

            with gr.Row():
                # ── Left column: controls ────────────────────────────────────
                with gr.Column(scale=1):
                    wb_genre = gr.Dropdown(
                        label="Genre",
                        choices=list(_GENRE_MODULES.keys()),
                        value="Isekai / Power Fantasy",
                        info="Sets the tone and reference vocabulary for the rewrite.",
                    )
                    wb_arc = gr.Dropdown(
                        label="Chapter Position",
                        choices=list(_ARC_MODULES.keys()),
                        value="Mid-arc — Tension (keep momentum)",
                        info="Controls the energy and hook strategy for this chapter.",
                    )
                    wb_style_dna = gr.Textbox(
                        label="Style DNA — Your Voice (optional but powerful)",
                        lines=5,
                        placeholder=(
                            "Paste 3-5 sentences written in YOUR exact voice here.\n"
                            "Not instructions — actual prose. The model will mirror it.\n\n"
                            "Example:\n"
                            "Bro wakes up dead. Not metaphorically dead — actually in a coffin, "
                            "underground, with like thirty seconds of air left. Classic Tuesday."
                        ),
                    )
                    with gr.Accordion("Generation Settings", open=False):
                        wb_max_tokens = gr.Slider(
                            256, 2048, value=800, step=64,
                            label="Max Output Tokens",
                            info="800 is good for a 5-8 min chapter recap.",
                        )
                        wb_temperature = gr.Slider(
                            0.1, 1.2, value=0.72, step=0.01,
                            label="Temperature",
                            info="Higher = more personality. Lower = more controlled.",
                        )

                # ── Right column: script in / out ────────────────────────────
                with gr.Column(scale=1):
                    wb_script_in = gr.Textbox(
                        label="Draft Script — paste your raw chapter here",
                        lines=12,
                        placeholder="Paste your full draft script here. Every plot beat stays. The WriterBot compresses, voices, and sharpens it.",
                    )
                    wb_generate_btn = gr.Button("🚀 Generate Script", variant="primary", size="lg")
                    wb_output = gr.Textbox(
                        label="Rewritten Script",
                        lines=12,
                        placeholder="Your cinEZma-style rewrite will appear here...",
                        show_copy_button=True,
                    )
                    wb_status = gr.Textbox(label="Status", lines=1)

            wb_generate_btn.click(
                generate_writerbot,
                inputs=[wb_script_in, wb_genre, wb_arc, wb_style_dna, wb_max_tokens, wb_temperature],
                outputs=[wb_output, wb_status],
                api_name="generate_writerbot",
            )


# ── Launch Gradio + relay registration ───────────────────────────────────────
import threading, time

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
                    r = requests.post(RELAY_URL, json={"url": url, "started_at": SESSION_START_TIME}, timeout=15)
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
        global _ALIGNER_MODEL
        if _ALIGNER_MODEL is None:
            from ctc_forced_aligner import get_word_stamps
            import tempfile, soundfile as sf
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

IDLE_WARN_SEC = 8 * 60

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

demo.queue()
demo.launch(share=True, debug=True, theme=gr.themes.Soft(), css=CSS)
