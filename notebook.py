"""
Fish Audio S2 Pro — cinEZma notebook.py
Push this to cinezma-omnivoice/notebook.py on GitHub.
All logic lives here. The launcher just exec()'s this.
"""

import os, sys, glob, json, time, subprocess, tempfile, requests
import gradio as gr
import soundfile as sf
import numpy as np
import torch

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR  = os.environ.get('FISH_S2_MODEL_DIR', '/kaggle/working/models/s2-pro')
VOICE_DIR  = '/kaggle/working/voice-assets/voices'
OUTPUT_DIR = '/kaggle/working/outputs'
RELAY_URL  = 'https://cinezma-relay.vercel.app/api/register'   # same relay as before
SESSION_ID = 'cinezma-fish-s2'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VOICE_DIR,  exist_ok=True)

print(f'[S2 Pro] MODEL_DIR  : {MODEL_DIR}')
print(f'[S2 Pro] VOICE_DIR  : {VOICE_DIR}')
print(f'[S2 Pro] CUDA       : {torch.cuda.is_available()}')

# ── Load model once at startup ────────────────────────────────────────────────
print('[S2 Pro] Loading inference engine...')
try:
    from fish_speech.inference_engine import TTSInferenceEngine
    engine = TTSInferenceEngine(
        checkpoint_path=MODEL_DIR,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        precision=torch.bfloat16,
    )
    print('[S2 Pro] Engine loaded ✓')
    USE_ENGINE = True
except Exception as e:
    print(f'[S2 Pro] Engine import failed ({e}), falling back to CLI mode.')
    USE_ENGINE = False

# ── Voice helpers ─────────────────────────────────────────────────────────────
def get_voices():
    wavs = glob.glob(os.path.join(VOICE_DIR, '**/*.wav'), recursive=True)
    wavs += glob.glob(os.path.join(VOICE_DIR, '**/*.mp3'), recursive=True)
    names = sorted(set(
        os.path.splitext(os.path.relpath(w, VOICE_DIR))[0].replace(os.sep, '/')
        for w in wavs
    ))
    return names if names else ['(no voices found — add .wav files to voices/)']

def voice_path(name):
    for ext in ('.wav', '.mp3'):
        p = os.path.join(VOICE_DIR, name + ext)
        if os.path.exists(p):
            return p
    return None

# ── Core TTS function ─────────────────────────────────────────────────────────
def generate_audio(text: str, voice_name: str, temperature: float, top_p: float):
    if not text.strip():
        raise gr.Error('Text is empty.')

    ref_audio = voice_path(voice_name) if voice_name else None
    out_path   = os.path.join(OUTPUT_DIR, f'out_{int(time.time()*1000)}.wav')

    if USE_ENGINE:
        # Python API path
        try:
            result = engine.generate(
                text=text,
                reference_audio=ref_audio,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=2048,
            )
            # result is typically (sample_rate, audio_array) or an object
            if hasattr(result, 'audio'):
                sr, audio = result.sample_rate, result.audio
            else:
                sr, audio = result
            sf.write(out_path, audio, sr)
        except Exception as e:
            raise gr.Error(f'Generation failed: {e}')
    else:
        # CLI fallback path
        cmd = [
            sys.executable, '-m', 'fish_speech.tts',
            '--text', text,
            '--checkpoint-path', MODEL_DIR,
            '--output', out_path,
            '--temperature', str(temperature),
            '--top-p', str(top_p),
            '--max-new-tokens', '2048',
        ]
        if ref_audio:
            cmd += ['--reference-audio', ref_audio]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.exists(out_path):
            raise gr.Error(f'CLI failed:\n{proc.stderr[-800:]}')

    return out_path


# ── Gradio UI ─────────────────────────────────────────────────────────────────
CSS = """
#gen-btn { background: linear-gradient(135deg,#667eea,#764ba2); border:none; color:white; font-weight:700; }
#gen-btn:hover { opacity:.9; }
.tag-hint { font-size:.8em; color:#888; margin-top:4px; }
"""

with gr.Blocks(title='Fish S2 Pro — cinEZma') as demo:

    gr.HTML("""
    <div style="text-align:center;padding:24px;background:linear-gradient(135deg,#667eea,#764ba2);
         border-radius:14px;margin-bottom:8px">
      <h1 style="color:white;margin:0;font-family:Inter,sans-serif;font-weight:700">
        🐟 Fish Audio S2 Pro
      </h1>
      <p style="color:rgba(255,255,255,.85);margin:6px 0 0;font-family:Inter,sans-serif;font-size:.9em">
        cinEZma Edition — type anything in [brackets] to direct the performance
      </p>
    </div>
    """)

    with gr.Row():
        voice_dd = gr.Dropdown(
            choices=get_voices(),
            value=get_voices()[0],
            label='🎙️ Voice',
            scale=2,
        )
        refresh_btn = gr.Button('↻', scale=0, min_width=50)

    text_in = gr.Textbox(
        label='Script',
        placeholder=(
            'She set the folder down. [long pause] Then she looked up.\n'
            '[laughing] I have absolutely no idea what just happened.\n\n'
            '— Use [any description you want] anywhere inline.'
        ),
        lines=7,
    )

    gr.HTML('<div class="tag-hint">💡 Examples: [voice breaking] [overly cheerful, clearly forcing it] '
            '[whispering, scared] [the calm tone of someone who has done this a thousand times]</div>')

    with gr.Accordion('⚙️ Advanced', open=False):
        temperature = gr.Slider(0.1, 1.0, value=0.7, step=0.05, label='Temperature')
        top_p       = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label='Top-p')

    gen_btn   = gr.Button('Generate ▶', variant='primary', elem_id='gen-btn')
    audio_out = gr.Audio(label='Output Audio', type='filepath')

    def refresh_voices():
        voices = get_voices()
        return gr.Dropdown(choices=voices, value=voices[0])

    gen_btn.click(
        fn=generate_audio,
        inputs=[text_in, voice_dd, temperature, top_p],
        outputs=audio_out,
    )
    refresh_btn.click(fn=refresh_voices, outputs=voice_dd)


# ── Launch + relay registration ───────────────────────────────────────────────
print('[S2 Pro] Launching Gradio...')
_, _, share_url = demo.launch(
    share=True,
    quiet=True,
    prevent_thread_lock=True,
    css=CSS,
)

print(f'[S2 Pro] Public URL: {share_url}')

# Register with relay so cinEZma auto-discovers this session
if share_url:
    try:
        r = requests.post(RELAY_URL, json={
            'id':  SESSION_ID,
            'url': share_url,
        }, timeout=10)
        print(f'[S2 Pro] Relay registered: {r.status_code}')
    except Exception as e:
        print(f'[S2 Pro] Relay registration skipped: {e}')

# Keep session alive
print('[S2 Pro] Ready. cinEZma can now connect.')
import threading
threading.Event().wait()
