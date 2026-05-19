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
    file_name = os.path.splitext(os.path.basename(audio_path))[0]
    clean_name = re.sub(r'[^a-zA-Z0-9]+', '_', file_name)
    clean_name = re.sub(r'_+', '_', clean_name).strip('_')
    wav_path = os.path.join(
        os.path.dirname(audio_path),
        f"{clean_name}_tmp.wav"
    )
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        if os.path.isfile(wav_path) and os.path.getsize(wav_path) > 0:
            return wav_path
    except Exception as e:
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

def process_audio(audio_file, seconds, method):
    if audio_file is None:
        return None, None, ""

    if not os.path.exists(audio_file):
        return None, None, ""

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

        # Format durations nicely
        def fmt(s):
            m = int(s // 60)
            sec = s % 60
            if m > 0:
                return f"{m}m {sec:.1f}s"
            return f"{sec:.2f}s"

        result_html = f"""
<div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin-top:8px;">
    <div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); border-radius:12px; padding:16px; text-align:center; color:white;">
        <div style="font-size:0.75em; opacity:0.85; text-transform:uppercase; letter-spacing:1px;">Original</div>
        <div style="font-size:1.5em; font-weight:700; margin-top:4px;">{fmt(before)}</div>
    </div>
    <div style="background:linear-gradient(135deg,#11998e 0%,#38ef7d 100%); border-radius:12px; padding:16px; text-align:center; color:white;">
        <div style="font-size:0.75em; opacity:0.85; text-transform:uppercase; letter-spacing:1px;">New</div>
        <div style="font-size:1.5em; font-weight:700; margin-top:4px;">{fmt(after)}</div>
    </div>
    <div style="background:linear-gradient(135deg,#eb3349 0%,#f45c43 100%); border-radius:12px; padding:16px; text-align:center; color:white;">
        <div style="font-size:0.75em; opacity:0.85; letter-spacing:1px;">REMOVED</div>
        <div style="font-size:1.5em; font-weight:700; margin-top:4px;">{percent:.1f}%</div>
        <div style="font-size:0.7em; opacity:0.8; margin-top:2px;">{fmt(removed)}</div>
    </div>
</div>
<div style="text-align:center; margin-top:10px; font-size:0.8em; color:#888;">
    Mode: {method}
</div>
"""
        return output_audio_file, output_audio_file, result_html

    except Exception as e:
        return None, None, f"<p style='color:#ef4444;'>Error: {str(e)}</p>"


# ─── UI ───────────────────────────────────────────────────────────────────────

def ui():
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(
        body_background_fill="linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%)",
        body_background_fill_dark="linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%)",
        block_background_fill="rgba(255,255,255,0.03)",
        block_background_fill_dark="rgba(255,255,255,0.03)",
        block_border_width="1px",
        block_border_color="rgba(255,255,255,0.08)",
        block_border_color_dark="rgba(255,255,255,0.08)",
        block_radius="16px",
        block_label_text_color="#e2e8f0",
        block_label_text_color_dark="#e2e8f0",
        block_title_text_color="#f1f5f9",
        block_title_text_color_dark="#f1f5f9",
        body_text_color="#e2e8f0",
        body_text_color_dark="#e2e8f0",
        body_text_color_subdued="#94a3b8",
        body_text_color_subdued_dark="#94a3b8",
        input_background_fill="rgba(255,255,255,0.06)",
        input_background_fill_dark="rgba(255,255,255,0.06)",
        input_border_color="rgba(255,255,255,0.12)",
        input_border_color_dark="rgba(255,255,255,0.12)",
        button_primary_background_fill="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        button_primary_background_fill_dark="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        button_primary_background_fill_hover="linear-gradient(135deg, #5a6fd6 0%, #6a4292 100%)",
        button_primary_background_fill_hover_dark="linear-gradient(135deg, #5a6fd6 0%, #6a4292 100%)",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_primary_border_color="rgba(255,255,255,0.15)",
        button_primary_border_color_dark="rgba(255,255,255,0.15)",
    )

    css = """
    /* Overall page */
    .gradio-container {
        max-width: 1000px !important;
        margin: auto !important;
    }

    /* Glass card effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 20px !important;
        padding: 24px !important;
    }

    /* Primary button */
    button.primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 12px !important;
        padding: 14px 24px !important;
        font-size: 1.1em !important;
        font-weight: 600 !important;
        color: white !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
    }

    button.primary:hover {
        box-shadow: 0 6px 30px rgba(102, 126, 234, 0.6) !important;
        transform: translateY(-1px) !important;
    }

    /* Radio buttons styling */
    .gr-radio {
        border-radius: 12px !important;
    }

    /* Mode selector cards */
    .mode-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 14px 18px;
        margin: 6px 0;
        transition: all 0.2s ease;
    }

    .mode-card:hover {
        border-color: rgba(102, 126, 234, 0.5);
        background: rgba(102, 126, 234, 0.08);
    }

    /* Footer */
    .footer-text {
        text-align: center;
        color: rgba(255,255,255,0.4);
        font-size: 0.8em;
        margin-top: 20px;
        padding: 16px;
    }

    .footer-text a {
        color: rgba(102, 126, 234, 0.8);
        text-decoration: none;
    }

    .footer-text a:hover {
        color: #667eea;
    }

    /* Hide dark mode toggle */
    .dark-mode-toggle { display: none !important; }

    /* Audio component */
    audio { border-radius: 10px !important; }
    """

    with gr.Blocks(theme=theme, css=css, title="Remove Silence From Audio") as demo:

        # ─── Header ──────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center; padding:40px 20px 20px;">
            <div style="font-size:3.5em; margin-bottom:8px;">
                <span style="background: linear-gradient(135deg, #667eea, #764ba2, #f093fb); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight:800;">
                    Remove Silence
                </span>
            </div>
            <p style="font-size:1.15em; color:#94a3b8; margin:0 0 8px; font-weight:300;">
                Drop your audio. Get it tight. Perfect for Shorts, TikTok & Reels.
            </p>
            <div style="display:inline-flex; gap:8px; margin-top:12px; flex-wrap:wrap; justify-content:center;">
                <span style="background:rgba(102,126,234,0.15); border:1px solid rgba(102,126,234,0.3); color:#a5b4fc; padding:4px 12px; border-radius:20px; font-size:0.8em;">100% Free</span>
                <span style="background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.3); color:#6ee7b7; padding:4px 12px; border-radius:20px; font-size:0.8em;">No Sign-up</span>
                <span style="background:rgba(244,63,94,0.15); border:1px solid rgba(244,63,94,0.3); color:#fda4af; padding:4px 12px; border-radius:20px; font-size:0.8em;">AI Powered</span>
            </div>
        </div>
        """)

        # ─── Main Content ────────────────────────────────────────────────
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                gr.HTML('<div style="color:#e2e8f0; font-weight:600; font-size:1em; margin-bottom:8px;">Upload Audio</div>')
                audio_input = gr.Audio(
                    label="",
                    type="filepath",
                    sources=["upload", "microphone"],
                    show_label=False
                )

                gr.HTML("""
                <div style="color:#e2e8f0; font-weight:600; font-size:1em; margin:16px 0 10px;">Choose Mode</div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:16px;">
                    <div style="background:rgba(102,126,234,0.08); border:1px solid rgba(102,126,234,0.25); border-radius:12px; padding:14px; text-align:center;">
                        <div style="font-size:1.4em;">&#9889;</div>
                        <div style="color:#a5b4fc; font-weight:600; font-size:0.85em; margin-top:4px;">Super Strict</div>
                        <div style="color:#64748b; font-size:0.72em; margin-top:3px;">Removes ALL quiet parts<br>breaths, noise, everything</div>
                    </div>
                    <div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.25); border-radius:12px; padding:14px; text-align:center;">
                        <div style="font-size:1.4em;">&#129504;</div>
                        <div style="color:#6ee7b7; font-weight:600; font-size:0.85em; margin-top:4px;">Human Speech Only</div>
                        <div style="color:#64748b; font-size:0.72em; margin-top:3px;">AI keeps only voice<br>removes breaths & noise</div>
                    </div>
                </div>
                """)

                method_choice = gr.Radio(
                    choices=["Super Strict", "Human Speech Only (AI)"],
                    value="Super Strict",
                    label="Mode",
                    show_label=False
                )

                silence_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=0.5,
                    step=0.01,
                    value=0.05,
                    label="Keep Silence (seconds)",
                    info="Lower = more trimmed. For Shorts/TikTok try 0.03-0.05"
                )

                submit_btn = gr.Button(
                    "Remove Silence",
                    variant="primary",
                    size="lg"
                )

            with gr.Column(scale=1):
                gr.HTML('<div style="color:#e2e8f0; font-weight:600; font-size:1em; margin-bottom:8px;">Result</div>')
                result_stats = gr.HTML(
                    value='<div style="text-align:center; padding:40px 20px; color:#64748b; font-size:0.9em;">Upload audio and click Remove Silence to see results here.</div>'
                )
                audio_output = gr.Audio(label="Play Processed Audio", show_label=True)
                file_output = gr.File(label="Download", show_label=True)

        # ─── Footer ──────────────────────────────────────────────────────
        gr.HTML("""
        <div class="footer-text">
            <p>Works with MP3, WAV, OGG, FLAC, M4A and more</p>
            <p style="margin-top:8px;">
                <a href="https://github.com/NeuralFalconYT/Remove-Silence-From-Audio" target="_blank">
                    Install locally for unlimited use
                </a>
                &nbsp;&bull;&nbsp; No copyrighted content please
            </p>
        </div>
        """)

        submit_btn.click(
            fn=process_audio,
            inputs=[audio_input, silence_threshold, method_choice],
            outputs=[audio_output, file_output, result_stats]
        )

    return demo


# ─── Launch ───────────────────────────────────────────────────────────────────

start_cleanup_worker()
demo = ui()
demo.queue().launch()
