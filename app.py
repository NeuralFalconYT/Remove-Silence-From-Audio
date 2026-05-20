import gradio as gr
import os
import uuid
from pydub import AudioSegment
from pydub.silence import split_on_silence
import re
import time
import subprocess
import threading
import torch
import torchaudio
import numpy as np

# ─── Load Silero VAD model globally ───────────────────────────────────────────
SILERO_MODEL = None
SILERO_UTILS = None

def load_silero_model():
    global SILERO_MODEL, SILERO_UTILS
    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        SILERO_MODEL = model
        SILERO_UTILS = utils
        print("Silero VAD model loaded successfully!")
    except Exception as e:
        print(f"Failed to load Silero VAD: {e}")

load_silero_model()

# ─── File Helpers ─────────────────────────────────────────────────────────────

def clean_file_name(file_path):
    file_name = os.path.basename(file_path)
    file_name, file_extension = os.path.splitext(file_name)
    cleaned = re.sub(r'[^a-zA-Z\d]+', '_', file_name)
    clean_name = re.sub(r'_+', '_', cleaned).strip('_')
    if clean_name.endswith('_tmp'):
        clean_name = clean_name[:-4]
    random_uuid = uuid.uuid4().hex[:6]
    clean_file_path = os.path.join(
        os.path.dirname(file_path),
        f"{clean_name}_{random_uuid}{file_extension}"
    )
    return clean_file_path

def calculate_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    return len(audio) / 1000.0


# ─── File Tracking & Cleanup ──────────────────────────────────────────────────

FILE_TIMESTAMPS = {}

def track_file(file_path):
    FILE_TIMESTAMPS[file_path] = time.time()

def cleanup_tracked_files(max_age_seconds=3600):
    now = time.time()
    to_delete = []
    for file_path, created_time in list(FILE_TIMESTAMPS.items()):
        if now - created_time > max_age_seconds:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    pass
            to_delete.append(file_path)
    for f in to_delete:
        FILE_TIMESTAMPS.pop(f, None)

_CLEANUP_STARTED = False

def start_cleanup_worker(interval=3600):
    global _CLEANUP_STARTED
    if _CLEANUP_STARTED:
        return
    _CLEANUP_STARTED = True
    def worker():
        while True:
            cleanup_tracked_files()
            time.sleep(interval)
    threading.Thread(target=worker, daemon=True).start()


# ─── Audio Conversion ─────────────────────────────────────────────────────────

def convert_to_wav(audio_path):
    if not os.path.isfile(audio_path):
        return None

    # Get extension
    ext = os.path.splitext(audio_path)[1].lower()

    # If already mp3 or wav, return original file
    if ext in [".mp3", ".wav"]:
        return audio_path

    # Clean filename
    file_name = os.path.splitext(os.path.basename(audio_path))[0]
    clean_name = re.sub(r'[^a-zA-Z0-9]+', '_', file_name)
    clean_name = re.sub(r'_+', '_', clean_name).strip('_')

    # Output wav path
    wav_path = os.path.join(
        os.path.dirname(audio_path),
        f"{clean_name}_tmp.wav"
    )

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", audio_path,
                "-ar", "16000",
                "-ac", "1",
                wav_path
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        if os.path.isfile(wav_path) and os.path.getsize(wav_path) > 0:
            return wav_path

    except Exception:
        pass

    return None


# ─── Method 1: Super Strict (PyDub) ──────────────────────────────────────────

def remove_silence_pydub(file_path, minimum_silence=50):
    sound = AudioSegment.from_file(file_path)
    audio_chunks = split_on_silence(
        sound, min_silence_len=100, silence_thresh=-45, keep_silence=minimum_silence
    )
    if not audio_chunks:
        dynamic_thresh = sound.dBFS - 16
        audio_chunks = split_on_silence(
            sound, min_silence_len=100, silence_thresh=dynamic_thresh, keep_silence=minimum_silence
        )
    combined = AudioSegment.empty()
    for chunk in audio_chunks:
        combined += chunk
    if len(combined) == 0:
        combined = sound
    output_path = clean_file_name(file_path)
    ext = os.path.splitext(output_path)[1].lower().replace('.', '')
    if not ext:
        ext = "wav"
        output_path += ".wav"
    combined.export(output_path, format=ext)
    return output_path


# ─── Method 2: Human Speech Only (Silero VAD) ────────────────────────────────

def remove_silence_silero(file_path, min_silence_duration_ms=100, padding_ms=30):
    global SILERO_MODEL, SILERO_UTILS
    if SILERO_MODEL is None:
        load_silero_model()
        if SILERO_MODEL is None:
            raise RuntimeError("AI model could not be loaded. Please try Super Strict mode.")

    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = SILERO_UTILS
    wav = read_audio(file_path, sampling_rate=16000)

    speech_timestamps = get_speech_timestamps(
        wav, SILERO_MODEL, sampling_rate=16000,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=padding_ms,
        min_speech_duration_ms=50,
        threshold=0.35
    )

    if not speech_timestamps:
        output_path = clean_file_name(file_path)
        sound = AudioSegment.from_file(file_path)
        sound.export(output_path, format="wav")
        return output_path

    original_audio = AudioSegment.from_file(file_path)
    original_sr = original_audio.frame_rate
    sr_ratio = original_sr / 16000.0

    combined = AudioSegment.empty()
    for ts in speech_timestamps:
        start_ms = int((ts['start'] * sr_ratio) / original_sr * 1000)
        end_ms = int((ts['end'] * sr_ratio) / original_sr * 1000)
        start_ms = max(0, start_ms)
        end_ms = min(len(original_audio), end_ms)
        if end_ms > start_ms:
            combined += original_audio[start_ms:end_ms]

    if len(combined) == 0:
        combined = original_audio

    output_path = clean_file_name(file_path)
    ext = os.path.splitext(output_path)[1].lower().replace('.', '')
    if not ext:
        ext = "wav"
        output_path += ".wav"
    combined.export(output_path, format=ext)
    return output_path


# ─── Main Processing ──────────────────────────────────────────────────────────

def process_audio(audio_file, seconds_str, method):
    if audio_file is None:
        return None, None, ""
    if not os.path.exists(audio_file):
        return None, None, ""

    try:
        seconds = float(seconds_str)
    except ValueError:
        seconds = 0.05

    track_file(audio_file)
    converted_audio = convert_to_wav(audio_file)

    if converted_audio:
        track_file(converted_audio)
        audio_file = converted_audio
    else:
        return None, None, "Invalid file format or conversion failed."

    keep_silence = int(seconds * 1000)

    try:
        before = calculate_duration(audio_file)

        if method == "Human Speech Only (AI)":
            output_audio_file = remove_silence_silero(
                audio_file,
                min_silence_duration_ms=max(keep_silence, 100),
                padding_ms=keep_silence
            )
        else:
            output_audio_file = remove_silence_pydub(
                audio_file,
                minimum_silence=keep_silence
            )

        track_file(output_audio_file)
        after = calculate_duration(output_audio_file)

        removed = before - after
        percent = (removed / before * 100) if before > 0 else 0

        def fmt(s):
            m = int(s // 60)
            sec = s % 60
            if m > 0:
                return f"{m}m {sec:.1f}s"
            return f"{sec:.2f}s"

        mode_label = "SUPER STRICT" if method == "Super Strict" else "HUMAN SPEECH (AI)"

        # Sleek, Dark-Themed Result Card
        result_html = f"""
        <div style="margin-top: 15px; background: #1c1c1c; border: 1px solid #333; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 10px rgba(0,0,0,0.5);">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #333;">
                
                <div style="flex: 1; padding: 18px 10px; text-align: center; border-right: 1px solid #333;">
                    <div style="font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;">Original</div>
                    <div style="font-size: 22px; font-weight: 700; color: #f4f4f5; font-family: 'Inter', sans-serif;">{fmt(before)}</div>
                </div>
                
                <div style="flex: 1; padding: 18px 10px; text-align: center; border-right: 1px solid #333; background: #222;">
                    <div style="font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;">New</div>
                    <div style="font-size: 22px; font-weight: 700; color: #3b82f6; font-family: 'Inter', sans-serif;">{fmt(after)}</div>
                </div>
                
                <div style="flex: 1; padding: 18px 10px; text-align: center; background: #1c1c1c;">
                    <div style="font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;">Removed</div>
                    <div style="font-size: 22px; font-weight: 700; color: #ef4444; font-family: 'Inter', sans-serif;">{percent:.1f}%</div>
                    <div style="font-size: 12px; color: #666; margin-top: 4px; font-weight: 500;">{fmt(removed)}</div>
                </div>

            </div>
            <div style="background: #141414; padding: 10px 16px; text-align: right; font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">
                Mode: {mode_label}
            </div>
        </div>
        """
        return output_audio_file, output_audio_file, result_html

    except Exception as e:
        return None, None, f"<p style='color:#ef4444; font-family:Inter,sans-serif; font-size:13px; margin-top:10px; padding:12px; background:#2a1111; border-radius:8px; border:1px solid #5a1a1a;'>Error: {str(e)}</p>"


# -----------------------------
# CSS FOR LAYOUT & THEMING
# -----------------------------

css = """
/* APP & MAIN */
body, html {
    background: #171717 !important;
    font-family: 'Inter', sans-serif !important;
    color: white !important;
    margin: 0;
    padding: 0;
}

/* 🟢 FORCING THE WIDE LAYOUT 🟢 */
.gradio-container {
    max-width: 1500px !important; /* Forces container to stretch out */
    width: 95% !important;        /* Uses 95% of the screen width */
    margin: auto !important;
    background: #171717 !important;
    padding: 20px 24px !important;
}

/* REMOVE DEFAULT FOOTER */
footer { display: none !important; }

.dark {
    --body-background-fill: #171717 !important;
    --block-background-fill: #262626 !important;
    --block-border-color: #333 !important;
}

/* TOPBAR */
.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 25px; 
}

.logo {
    display: flex;
    align-items: center;
    gap: 12px;
}

.logo-icon {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: white;
    color: black;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 15px;
}

.logo-text {
    font-size: 18px;
    font-weight: 600;
}

.nav {
    display: flex;
    gap: 22px;
    align-items: center;
}

.nav a {
    color: #a1a1aa;
    text-decoration: none;
    cursor: pointer;
    transition: 0.2s;
    font-size: 13px;
    font-weight: 500;
}

.nav a:hover {
    color: white;
}

.nav-yt { color: #ff4e4e !important; }
.nav-kofi { color: #29abe0 !important; }

/* HERO */
.hero {
    text-align: center;
    margin-bottom: 35px; 
}

.hero h1 {
    font-size: 42px; 
    font-weight: 800;
    letter-spacing: -0.05em;
    margin-bottom: 6px;
}

/* BLUE HERO TEXT */
.hero span {
    color: #3b82f6; 
}

.hero p {
    color: #a1a1aa;
    font-size: 15px; 
    margin: 4px 0 16px 0;
}

.badges {
    display: flex;
    justify-content: center;
    gap: 10px;
}

.badge {
    background: #262626;
    border: 1px solid #333;
    padding: 6px 14px;
    border-radius: 999px;
    color: #d4d4d8;
    font-size: 11px; 
    font-weight: 500;
}

/* CARDS / PANELS */
.panel {
    background: #262626 !important;
    border: 1px solid #333 !important;
    border-radius: 14px !important;
    padding: 24px !important; 
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
}

/* FORCE SIDE BY SIDE ON DESKTOP */
@media (min-width: 768px) {
    .side-by-side {
        flex-wrap: nowrap !important;
    }
}

/* TITLES */
.section-title {
    font-size: 12px;
    text-transform: uppercase;
    color: #888;
    letter-spacing: 1px;
    margin-bottom: 14px;
    font-weight: 600;
}

/* AUDIO COMPONENT */
audio {
    border-radius: 8px !important;
    background: #1f1f1f !important;
}

/* CUSTOM BLUE BUTTON */
#process-btn {
    background: #3b82f6 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    height: 48px !important; 
    font-size: 15px !important;
    font-weight: 600 !important;
    margin-top: 14px !important;
    transition: all 0.2s ease-in-out !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
}

#process-btn:hover {
    background: #2563eb !important;
    box-shadow: 0 4px 16px rgba(59, 130, 246, 0.4) !important;
    transform: translateY(-1px);
}

/* INPUTS */
.gr-textbox, .gr-dropdown {
    background: #1f1f1f !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 8px !important;
}

/* MOBILE RESPONSIVENESS */
@media(max-width: 900px) {
    .topbar { flex-direction: column; gap: 16px; }
    .nav { flex-wrap: wrap; justify-content: center; }
}
"""

# -----------------------------
# START BACKGROUND WORKERS
# -----------------------------
start_cleanup_worker()

# -----------------------------
# UI
# -----------------------------

with gr.Blocks(
    theme=gr.themes.Base(),
    css=css,
    title="Remove Silence"
) as demo:

    # TOPBAR
    gr.HTML("""
    <div class="topbar">
        <div class="logo">
            <div class="logo-icon">R</div>
            <div class="logo-text">Remove Silence</div>
        </div>
        <div class="nav">
            <a href="https://www.youtube.com/@neuralfalcon/" target="_blank" class="nav-yt">Subscribe YouTube</a>
            <a href="https://ko-fi.com/neuralfalcon" target="_blank" class="nav-kofi">Donate</a>
            <a href="https://github.com/NeuralFalconYT/Remove-Silence-From-Audio" target="_blank">GitHub</a>
            <a href="mailto:NeuralFalcon@proton.me" target="_blank">Mail</a>
            <a href="https://x.com/NeuralFalcon" target="_blank">X</a>
        </div>
    </div>
    """)

    # HERO
    gr.HTML("""
    <div class="hero">
        <h1>REMOVE <span>SILENCE</span></h1>
        <p>Drop your audio and instantly clean pauses for Shorts, TikTok & Reels</p>
        <div class="badges">
            <div class="badge">100% Free</div>
            <div class="badge">No Sign-Up</div>
            <div class="badge">AI Powered</div>
        </div>
    </div>
    """)

    with gr.Row(equal_height=False, elem_classes="side-by-side"):

        # LEFT PANEL (UPLOAD) - Adjusted min_width to stretch beautifully in the new wide container
        with gr.Column(scale=1, min_width=450):
            with gr.Group(elem_classes="panel"):
                gr.HTML('<div class="section-title">Upload & Settings</div>')

                input_audio = gr.Audio(
                    type="filepath",
                    label="",
                    show_label=False
                )

                with gr.Row():
                    mode = gr.Dropdown(choices=["Super Strict", "Human Speech Only (AI)"], value="Super Strict", label="Mode")
                    keep_silence = gr.Textbox(
                        value="0.05",
                        label="Keep Silence (sec)"
                    )

                # Blue submit button
                process_btn = gr.Button("Remove Silence", elem_id="process-btn")

        # RIGHT PANEL (RESULT)
        with gr.Column(scale=1, min_width=450):
            with gr.Group(elem_classes="panel"):
                gr.HTML('<div class="section-title">Result</div>')

                output_audio = gr.Audio(
                    label="Play Audio",
                    show_label=False
                )
                
                download_audio = gr.File(
                    label="Download Audio",
                    show_label=False
                )
                
                stats_html = gr.HTML()

    # PROCESS
    process_btn.click(
        fn=process_audio,
        inputs=[input_audio, keep_silence, mode],
        outputs=[output_audio, download_audio, stats_html]
    )

demo.launch()
