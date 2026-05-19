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
    return os.path.join(os.path.dirname(file_path), f"{clean_name}_{random_uuid}{file_extension}")


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
                except:
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
    wav_path = os.path.join(os.path.dirname(audio_path), f"{clean_name}_tmp.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", wav_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        if os.path.isfile(wav_path) and os.path.getsize(wav_path) > 0:
            return wav_path
    except:
        pass
    return None


# ─── Method 1: Super Strict (PyDub) ──────────────────────────────────────────

def remove_silence_pydub(file_path, minimum_silence=50):
    sound = AudioSegment.from_file(file_path)
    audio_chunks = split_on_silence(sound, min_silence_len=100, silence_thresh=-45, keep_silence=minimum_silence)
    if not audio_chunks:
        dynamic_thresh = sound.dBFS - 16
        audio_chunks = split_on_silence(sound, min_silence_len=100, silence_thresh=dynamic_thresh, keep_silence=minimum_silence)
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
        min_speech_duration_ms=50, threshold=0.35
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
        return None, None, "<p style='color:#ef4444;'>Invalid file format or conversion failed.</p>"

    keep_silence = int(seconds * 1000)

    try:
        before = calculate_duration(audio_file)

        if method == "Human Speech Only (AI)":
            output_audio_file = remove_silence_silero(
                audio_file, min_silence_duration_ms=max(keep_silence, 100), padding_ms=keep_silence
            )
        else:
            output_audio_file = remove_silence_pydub(audio_file, minimum_silence=keep_silence)

        track_file(output_audio_file)
        after = calculate_duration(output_audio_file)
        removed = before - after
        percent = (removed / before * 100) if before > 0 else 0

        def fmt(s):
            m = int(s // 60)
            sec = s % 60
            return f"{m}m {sec:.1f}s" if m > 0 else f"{sec:.2f}s"

        result_html = f"""
<div style="display:flex; justify-content:center; gap:24px; margin:12px 0; flex-wrap:wrap;">
    <div style="text-align:center;">
        <div style="font-size:0.7em; text-transform:uppercase; letter-spacing:2px; color:#94a3b8;">Original</div>
        <div style="font-size:1.8em; font-weight:300; color:#e2e8f0; margin-top:2px;">{fmt(before)}</div>
    </div>
    <div style="text-align:center; color:#475569; font-size:1.5em; align-self:center;">&#8594;</div>
    <div style="text-align:center;">
        <div style="font-size:0.7em; text-transform:uppercase; letter-spacing:2px; color:#94a3b8;">New</div>
        <div style="font-size:1.8em; font-weight:300; color:#e2e8f0; margin-top:2px;">{fmt(after)}</div>
    </div>
    <div style="text-align:center; color:#475569; font-size:1.5em; align-self:center;">&#8226;</div>
    <div style="text-align:center;">
        <div style="font-size:0.7em; text-transform:uppercase; letter-spacing:2px; color:#94a3b8;">Removed</div>
        <div style="font-size:1.8em; font-weight:300; color:#f87171; margin-top:2px;">{percent:.1f}%</div>
    </div>
</div>
<div style="text-align:center; font-size:0.72em; color:#64748b; margin-top:4px;">Mode: {method}</div>
"""
        return output_audio_file, output_audio_file, result_html
    except Exception as e:
        return None, None, f"<p style='color:#ef4444;'>Error: {str(e)}</p>"


# ─── UI ───────────────────────────────────────────────────────────────────────

def ui():
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.slate,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#0f172a",
        body_background_fill_dark="#0f172a",
        block_background_fill="rgba(255,255,255,0.02)",
        block_background_fill_dark="rgba(255,255,255,0.02)",
        block_border_width="1px",
        block_border_color="rgba(255,255,255,0.06)",
        block_border_color_dark="rgba(255,255,255,0.06)",
        block_radius="12px",
        block_label_text_color="#94a3b8",
        block_label_text_color_dark="#94a3b8",
        block_title_text_color="#e2e8f0",
        block_title_text_color_dark="#e2e8f0",
        body_text_color="#e2e8f0",
        body_text_color_dark="#e2e8f0",
        body_text_color_subdued="#64748b",
        body_text_color_subdued_dark="#64748b",
        input_background_fill="rgba(255,255,255,0.04)",
        input_background_fill_dark="rgba(255,255,255,0.04)",
        input_border_color="rgba(255,255,255,0.08)",
        input_border_color_dark="rgba(255,255,255,0.08)",
        button_primary_background_fill="#e2e8f0",
        button_primary_background_fill_dark="#e2e8f0",
        button_primary_background_fill_hover="#f8fafc",
        button_primary_background_fill_hover_dark="#f8fafc",
        button_primary_text_color="#0f172a",
        button_primary_text_color_dark="#0f172a",
        button_primary_border_color="transparent",
        button_primary_border_color_dark="transparent",
    )

    css = """
    .gradio-container {
        max-width: 860px !important;
        margin: auto !important;
        padding: 0 20px !important;
    }

    button.primary {
        border-radius: 8px !important;
        padding: 12px 32px !important;
        font-size: 0.95em !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.2s ease !important;
    }

    button.primary:hover {
        opacity: 0.9 !important;
    }

    audio { border-radius: 8px !important; }

    .footer-link {
        text-align: center;
        margin-top: 48px;
        padding: 24px 0;
        border-top: 1px solid rgba(255,255,255,0.04);
        color: #475569;
        font-size: 0.78em;
    }

    .footer-link a {
        color: #64748b;
        text-decoration: none;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 1px;
    }

    .footer-link a:hover { color: #94a3b8; }
    """

    with gr.Blocks(theme=theme, css=css, title="Remove Silence") as demo:

        # ─── Header ──────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center; padding:60px 0 40px;">
            <h1 style="font-size:2.8em; font-weight:200; color:#f1f5f9; margin:0; letter-spacing:-1px;">
                Remove Silence
            </h1>
            <p style="font-size:0.95em; color:#64748b; margin:8px 0 0; font-weight:300;">
                Drop your audio. Get it tight. Perfect for Shorts, TikTok & Reels.
            </p>
        </div>
        """)

        # ─── Upload + Controls (centered, single column) ─────────────────
        audio_input = gr.Audio(
            label="Upload Audio",
            type="filepath",
            sources=["upload", "microphone"]
        )

        with gr.Row():
            method_choice = gr.Radio(
                choices=["Super Strict", "Human Speech Only (AI)"],
                value="Super Strict",
                label="Mode",
            )
            silence_threshold = gr.Slider(
                minimum=0.0, maximum=0.5, step=0.01, value=0.05,
                label="Keep Silence (seconds)",
                info="Lower = more trimmed. For Shorts/TikTok try 0.03-0.05"
            )

        submit_btn = gr.Button("Remove Silence", variant="primary", size="lg")

        # ─── Results ─────────────────────────────────────────────────────
        gr.HTML('<div style="height:32px;"></div>')

        result_stats = gr.HTML(value="")
        audio_output = gr.Audio(label="Processed Audio", show_label=True)
        file_output = gr.File(label="Download", show_label=True)

        # ─── Mode explanation (below, subtle) ────────────────────────────
        gr.HTML("""
        <div style="display:flex; justify-content:center; gap:48px; margin:48px 0 0; flex-wrap:wrap;">
            <div style="text-align:center; max-width:200px;">
                <div style="font-size:1.2em; margin-bottom:6px;">&#9889;</div>
                <div style="font-size:0.8em; font-weight:500; color:#cbd5e1; margin-bottom:4px;">Super Strict</div>
                <div style="font-size:0.72em; color:#475569; line-height:1.5;">Removes ALL quiet parts &mdash; breaths, noise, everything</div>
            </div>
            <div style="text-align:center; max-width:200px;">
                <div style="font-size:1.2em; margin-bottom:6px;">&#129504;</div>
                <div style="font-size:0.8em; font-weight:500; color:#cbd5e1; margin-bottom:4px;">Human Speech Only</div>
                <div style="font-size:0.72em; color:#475569; line-height:1.5;">AI keeps only voice, removes breaths & noise</div>
            </div>
        </div>
        """)

        # ─── Footer ──────────────────────────────────────────────────────
        gr.HTML("""
        <div class="footer-link">
            Works with MP3, WAV, OGG, FLAC, M4A and more<br><br>
            <a href="https://github.com/NeuralFalconYT/Remove-Silence-From-Audio" target="_blank">Install locally for unlimited use</a>
            &nbsp;&middot;&nbsp; No copyrighted content please
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
