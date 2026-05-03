"""
Fish Audio S2 Pro — cinEZma notebook.py
Uses fish-speech's built-in HTTP API server (most stable approach).
Push to cinezma-omnivoice/notebook.py on GitHub.
"""

import os, sys, glob, time, subprocess, threading, requests, tempfile
import gradio as gr
import soundfile as sf
import torch

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR  = os.environ.get('FISH_S2_MODEL_DIR', '/kaggle/working/models/s2-pro')
VOICE_DIR  = '/kaggle/working/voice-assets/voices'
OUTPUT_DIR = '/kaggle/working/outputs'
RELAY_URL  = 'https://cinezma-relay.vercel.app/api/register'
SESSION_ID = 'cinezma-fish-s2'
API_PORT   = 8080
API_BASE   = f'http://127.0.0.1:{API_PORT}'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VOICE_DIR,  exist_ok=True)

print(f'[S2 Pro] MODEL_DIR : {MODEL_DIR}')
print(f'[S2 Pro] VOICE_DIR : {VOICE_DIR}')
print(f'[S2 Pro] CUDA      : {torch.cuda.is_available()}')

# ── Start fish-speech HTTP API server ─────────────────────────────────────────
print('[S2 Pro] Starting fish-speech API server...')
server_proc = subprocess.Popen(
    [
        sys.executable, '-m', 'fish_speech.inference_engine',
        '--checkpoint-path', MODEL_DIR,
        '--device', 'cuda' if torch.cuda.is_available() else 'cpu',
        '--half',
        '--listen', f'0.0.0.0:{API_PORT}',
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

# Stream server logs in background
def _stream_logs(proc):
    for line in proc.stdout:
        print(f'[fish-server] {line}', end='')
threading.Thread(target=_stream_logs, args=(server_proc,), daemon=True).start()

# Wait for server to be ready
print('[S2 Pro] Waiting for API server to come online...')
for attempt in range(60):
    try:
        r = requests.get(f'{API_BASE}/v1/models', timeout=2)
        if r.status_code == 200:
            print(f'[S2 Pro] API server ready after {attempt+1}s ✓')
            break
    except Exception:
        pass
    time.sleep(1)
else:
    print('[S2 Pro] WARNING: Server did not respond in 60s — will try anyway.')

# ── Voice helpers ─────────────────────────────────────────────────────────────
def get_voices():
    wavs = glob.glob(os.path.join(VOICE_DIR, '**/*.wav'), recursive=True)
    wavs += glob.glob(os.path.join(VOICE_DIR, '**/*.mp3'), recursive=True)
    names = sorted(set(
        os.path.splitext(os.path.relpath(w, VOICE_DIR))[0].replace(os.sep, '/')
        for w in wavs
    ))
    return names if names else ['(no voices — add .wav to voices/ folder)']

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
    out_path  = os.path.join(OUTPUT_DIR, f'out_{int(time.time()*1000)}.wav')

    files = {'text': (None, text)}
    if ref_audio and os.path.exists(ref_audio):
        files['reference_audio'] = (os.path.basename(ref_audio), open(ref_audio, 'rb'), 'audio/wav')

    params = {
        'temperature': temperature,
        'top_p': top_p,
        'max_new_tokens': 2048,
        'format': 'wav',
    }

    try:
        resp = requests.post(
            f'{API_BASE}/v1/tts',
            files=files,
            params=params,
            timeout=120,
        )
        if resp.status_code != 200:
            raise gr.Error(f'API error {resp.status_code}: {resp.text[:400]}')
        with open(out_path, 'wb') as f:
            f.write(resp.content)
    except requests.exceptions.Timeout:
        raise gr.Error('Generation timed out. Try shorter text.')
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f'Request failed: {e}')

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
        voice_dd    = gr.Dropdown(choices=get_voices(), value=get_voices()[0], label='🎙️ Voice', scale=2)
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
        v = get_voices()
        return gr.Dropdown(choices=v, value=v[0])

    gen_btn.click(fn=generate_audio, inputs=[text_in, voice_dd, temperature, top_p], outputs=audio_out)
    refresh_btn.click(fn=refresh_voices, outputs=voice_dd)

# ── Launch + relay ────────────────────────────────────────────────────────────
print('[S2 Pro] Launching Gradio...')
_, _, share_url = demo.launch(
    share=True,
    quiet=True,
    prevent_thread_lock=True,
    css=CSS,
)

print(f'[S2 Pro] Public URL: {share_url}')

if share_url:
    try:
        r = requests.post(RELAY_URL, json={'id': SESSION_ID, 'url': share_url}, timeout=10)
        print(f'[S2 Pro] Relay registered: {r.status_code}')
    except Exception as e:
        print(f'[S2 Pro] Relay skipped: {e}')

print('[S2 Pro] Ready.')
threading.Event().wait()
