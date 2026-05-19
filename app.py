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
        print("✅ Silero VAD model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Failed to load Silero VAD: {e}")

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
    duration_seconds = len(audio) / 1000.0
    return duration_seconds


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
                    print(f"🗑️ Deleted: {file_path}")
                except Exception as e:
                    print(f"⚠️ Error deleting {file_path}: {e}")
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
        print(f"⚠️ FFmpeg conversion error: {e}")

    return None


# ─── Method 1: Super Strict (PyDub) ──────────────────────────────────────────

def remove_silence_pydub(file_path, minimum_silence=50):
    sound = AudioSegment.from_file(file_path)

    audio_chunks = split_on_silence(
        sound,
        min_silence_len=100,
        silence_thresh=-45,
        keep_silence=minimum_silence
    )

    if not audio_chunks:
        dynamic_thresh = sound.dBFS - 16
        audio_chunks = split_on_silence(
            sound,
            min_silence_len=100,
            silence_thresh=dynamic_thresh,
            keep_silence=minimum_silence
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
    """
    Uses Silero VAD to detect human speech only.
    Removes everything that is not speech — breaths, noise, dead air.
    """
    global SILERO_MODEL, SILERO_UTILS

    if SILERO_MODEL is None:
        load_silero_model()
        if SILERO_MODEL is None:
            raise RuntimeError("AI model could not be loaded. Please try Super Strict mode.")

    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = SILERO_UTILS

    wav = read_audio(file_path, sampling_rate=16000)

    speech_timestamps = get_speech_timestamps(
        wav,
        SILERO_MODEL,
        sampling_rate=16000,
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
        return None, None, "No file uploaded"

    if not os.path.exists(audio_file):
        return None, None, "File not found"

    track_file(audio_file)

    converted_audio = convert_to_wav(audio_file)

    if converted_audio:
        track_file(converted_audio)
        audio_file = converted_audio
    else:
        return None, None, "Invalid file format or conversion failed"

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

        text = (
            f"Original Duration: {before:.2f}s\n"
            f"New Duration: {after:.2f}s\n"
            f"Silence Removed: {removed:.2f}s ({percent:.1f}%)\n"
            f"Method: {method}"
        )

        return output_audio_file, output_audio_file, text

    except Exception as e:
        print(f"⚠️ Error processing audio: {e}")
        return None, None, f"An error occurred during processing: {str(e)}"


# ─── UI ───────────────────────────────────────────────────────────────────────

def ui():
    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"]
    )

    css = """
    .gradio-container {max-width: none !important;}

    button.primary {
        background-color: #2563eb !important;
        color: white !important;
        font-weight: 600;
        border: none !important;
        border-radius: 10px;
        padding: 12px 18px;
        font-size: 1.05em;
    }

    button.primary:hover {
        background-color: #1e40af !important;
    }
    """

    with gr.Blocks(theme=theme, css=css) as demo:

        gr.HTML("""
        <div style="text-align:center; margin:20px auto; max-width:800px;">
            <h1 style="font-size:2.4em; margin-bottom:6px;">
                🔇 Remove Silence From Audio
            </h1>
            <p style="font-size:1.05em; color:#555; margin:0 0 10px;">
                Upload an audio file to remove silent parts — perfect for YT Shorts, TikTok & Reels.
            </p>
            <p style="font-size:0.8em; color:#999;">
                ⚠️ Please don't upload copyrighted content — it can take this Space offline.
            </p>
            <p style="font-size:0.9em; color:#777;">
                Install locally on your computer, enjoy unlimited runs with no waiting queue  
                <a href="https://github.com/NeuralFalconYT/Remove-Silence-From-Audio" target="_blank" style="text-decoration:none;">
                    Download Link
                </a>
            </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Upload Audio",
                    type="filepath",
                    sources=["upload", "microphone"]
                )

                method_choice = gr.Radio(
                    choices=["Super Strict", "Human Speech Only (AI)"],
                    value="Super Strict",
                    label="Mode",
                    info="Super Strict = removes ALL quiet parts | Human Speech Only = AI keeps only voice, removes breaths & noise"
                )

                silence_threshold = gr.Number(
                    label="Keep Silence Upto (in seconds)",
                    value=0.05
                )

                submit_btn = gr.Button(
                    "🔇 Remove Silence",
                    variant="primary"
                )

            with gr.Column(scale=1):
                audio_output = gr.Audio(label="Play Audio")
                file_output = gr.File(label="Download Audio File")
                duration_output = gr.Textbox(label="Result", lines=4)

        submit_btn.click(
            fn=process_audio,
            inputs=[audio_input, silence_threshold, method_choice],
            outputs=[audio_output, file_output, duration_output]
        )

    return demo


# ─── Launch ───────────────────────────────────────────────────────────────────

start_cleanup_worker()
demo = ui()
demo.queue().launch()
