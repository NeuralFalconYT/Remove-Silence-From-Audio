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

        def fmt(s):
            m = int(s // 60)
            sec = s % 60
            if m > 0:
                return f"{m}m {sec:.1f}s"
            return f"{sec:.2f}s"

        mode_label = "SUPER STRICT" if method == "Super Strict" else "HUMAN SPEECH (AI)"

        result_html = f"""
<div style="margin-top:14px; border:1px solid #e8eaed; border-radius:12px; overflow:hidden; background:#ffffff; box-shadow:0 1px 3px rgba(0,0,0,0.08);">
    <div style="display:grid; grid-template-columns:1fr 1fr 1fr;">
        <div style="padding:22px 16px; text-align:center; border-right:1px solid #f1f3f4;">
            <div style="font-size:11px; letter-spacing:0.05em; color:#5f6368; text-transform:uppercase; margin-bottom:10px; font-family:'Inter',sans-serif; font-weight:500;">Original</div>
            <div style="font-size:22px; font-weight:600; color:#202124; letter-spacing:0.01em; font-family:'Inter',sans-serif;">{fmt(before)}</div>
        </div>
        <div style="padding:22px 16px; text-align:center; border-right:1px solid #f1f3f4; background:#f8f9fe;">
            <div style="font-size:11px; letter-spacing:0.05em; color:#5f6368; text-transform:uppercase; margin-bottom:10px; font-family:'Inter',sans-serif; font-weight:500;">New</div>
            <div style="font-size:22px; font-weight:600; color:#1a73e8; letter-spacing:0.01em; font-family:'Inter',sans-serif;">{fmt(after)}</div>
        </div>
        <div style="padding:22px 16px; text-align:center;">
            <div style="font-size:11px; letter-spacing:0.05em; color:#5f6368; text-transform:uppercase; margin-bottom:10px; font-family:'Inter',sans-serif; font-weight:500;">Removed</div>
            <div style="font-size:22px; font-weight:600; color:#ea4335; letter-spacing:0.01em; font-family:'Inter',sans-serif;">{percent:.1f}%</div>
            <div style="font-size:12px; color:#80868b; margin-top:4px; font-family:'Inter',sans-serif;">{fmt(removed)}</div>
        </div>
    </div>
    <div style="padding:10px 16px; background:#f8f9fa; border-top:1px solid #f1f3f4; text-align:right;">
        <span style="font-family:'Inter',sans-serif; font-size:11px; color:#80868b; letter-spacing:0.02em;">Mode: {mode_label}</span>
    </div>
</div>
"""
        return output_audio_file, output_audio_file, result_html

    except Exception as e:
        return None, None, f"<p style='color:#ea4335; font-family:Inter,sans-serif; font-size:13px; margin-top:12px; padding:12px 16px; background:#fce8e6; border-radius:8px; border:1px solid #f5c6cb;'>Error: {str(e)}</p>"


# ─── UI ───────────────────────────────────────────────────────────────────────

def ui():
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.gray,
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        font_mono=[gr.themes.GoogleFont("Inter"), "monospace"],
    ).set(
        body_background_fill="#ffffff",
        body_background_fill_dark="#ffffff",
        block_background_fill="#ffffff",
        block_background_fill_dark="#ffffff",
        block_border_width="1px",
        block_border_color="#e8eaed",
        block_border_color_dark="#e8eaed",
        block_radius="12px",
        block_shadow="0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)",
        block_shadow_dark="0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)",
        block_label_background_fill="#ffffff",
        block_label_background_fill_dark="#ffffff",
        block_label_border_width="0px",
        block_label_text_color="#5f6368",
        block_label_text_color_dark="#5f6368",
        block_label_text_size="13px",
        block_title_text_color="#202124",
        block_title_text_color_dark="#202124",
        block_title_text_size="14px",
        body_text_color="#202124",
        body_text_color_dark="#202124",
        body_text_color_subdued="#5f6368",
        body_text_color_subdued_dark="#5f6368",
        body_text_size="14px",
        input_background_fill="#f8f9fa",
        input_background_fill_dark="#f8f9fa",
        input_background_fill_focus="#ffffff",
        input_background_fill_focus_dark="#ffffff",
        input_border_color="#e8eaed",
        input_border_color_dark="#e8eaed",
        input_border_color_focus="#1a73e8",
        input_border_color_focus_dark="#1a73e8",
        input_border_width="1px",
        input_radius="8px",
        input_shadow="none",
        input_shadow_dark="none",
        input_text_size="14px",
        input_placeholder_color="#9aa0a6",
        input_placeholder_color_dark="#9aa0a6",
        button_primary_background_fill="linear-gradient(135deg, #1a73e8 0%, #6c63ff 100%)",
        button_primary_background_fill_dark="linear-gradient(135deg, #1a73e8 0%, #6c63ff 100%)",
        button_primary_background_fill_hover="linear-gradient(135deg, #1557b0 0%, #5a52d5 100%)",
        button_primary_background_fill_hover_dark="linear-gradient(135deg, #1557b0 0%, #5a52d5 100%)",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_primary_border_color="transparent",
        button_primary_border_color_dark="transparent",
        button_secondary_background_fill="#f8f9fa",
        button_secondary_background_fill_dark="#f8f9fa",
        button_secondary_background_fill_hover="#f1f3f4",
        button_secondary_background_fill_hover_dark="#f1f3f4",
        button_secondary_text_color="#1a73e8",
        button_secondary_text_color_dark="#1a73e8",
        button_secondary_border_color="#e8eaed",
        button_secondary_border_color_dark="#e8eaed",
        button_large_radius="24px",
        button_large_text_size="14px",
        button_large_padding="12px 32px",
        slider_color="#1a73e8",
        slider_color_dark="#1a73e8",
        checkbox_background_color="#ffffff",
        checkbox_background_color_dark="#ffffff",
        checkbox_border_color="#dadce0",
        checkbox_border_color_dark="#dadce0",
        checkbox_border_color_selected="#1a73e8",
        checkbox_border_color_selected_dark="#1a73e8",
        checkbox_label_background_fill="#ffffff",
        checkbox_label_background_fill_dark="#ffffff",
        checkbox_label_background_fill_selected="#e8f0fe",
        checkbox_label_background_fill_selected_dark="#e8f0fe",
        checkbox_label_border_color="#e8eaed",
        checkbox_label_border_color_dark="#e8eaed",
        checkbox_label_border_color_hover="#1a73e8",
        checkbox_label_border_color_hover_dark="#1a73e8",
        checkbox_label_text_color="#3c4043",
        checkbox_label_text_color_dark="#3c4043",
        checkbox_label_text_color_selected="#1a73e8",
        checkbox_label_text_color_selected_dark="#1a73e8",
    )

    css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .gradio-container {
        max-width: 900px !important;
        margin: 0 auto !important;
        padding: 0 24px !important;
        box-sizing: border-box !important;
        background: #ffffff !important;
    }

    body {
        background: #f8f9fa !important;
    }

    /* ── YouTube Banner ── */
    .yt-banner {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        padding: 10px 20px;
        background: #f0f4ff;
        border: 1px solid #d2e3fc;
        border-radius: 10px;
        margin-bottom: 32px;
        margin-top: 16px;
    }

    .yt-banner svg {
        flex-shrink: 0;
    }

    .yt-banner p {
        margin: 0;
        font-family: 'Inter', sans-serif;
        font-size: 13px;
        color: #3c4043;
        font-weight: 400;
    }

    .yt-banner a {
        color: #1a73e8;
        font-weight: 600;
        text-decoration: none;
    }

    .yt-banner a:hover {
        text-decoration: underline;
    }

    /* ── Header ── */
    .site-header {
        padding: 40px 0 32px;
        text-align: center;
    }

    .header-title {
        font-family: 'Inter', sans-serif;
        font-size: clamp(28px, 5vw, 42px);
        font-weight: 700;
        color: #202124;
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin-bottom: 12px;
    }

    .header-title span {
        background: linear-gradient(135deg, #1a73e8, #6c63ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .header-sub {
        font-family: 'Inter', sans-serif;
        font-size: 15px;
        color: #5f6368;
        margin-bottom: 20px;
        font-weight: 400;
    }

    .header-badges {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        justify-content: center;
    }

    .hbadge {
        font-family: 'Inter', sans-serif;
        font-size: 12px;
        font-weight: 500;
        padding: 4px 14px;
        border: 1px solid #e8eaed;
        color: #5f6368;
        border-radius: 20px;
        background: #f8f9fa;
    }

    /* ── Section labels ── */
    .section-tag {
        font-family: 'Inter', sans-serif;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.02em;
        color: #5f6368;
        text-transform: uppercase;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #f1f3f4;
    }

    /* ── Radio ── */
    .gr-radio-group .wrap { gap: 8px !important; }
    .gr-radio-group label {
        border-radius: 20px !important;
        padding: 8px 18px !important;
        font-size: 13px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        transition: all 0.15s !important;
    }

    /* ── Submit button ── */
    .submit-row button {
        width: 100% !important;
        height: 48px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        border-radius: 24px !important;
        font-family: 'Inter', sans-serif !important;
        box-shadow: 0 2px 8px rgba(26, 115, 232, 0.2) !important;
        transition: all 0.2s ease !important;
    }

    .submit-row button:hover {
        box-shadow: 0 4px 16px rgba(26, 115, 232, 0.3) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Divider ── */
    .hdivider {
        height: 1px;
        background: #f1f3f4;
        margin: 20px 0;
    }

    /* ── Result placeholder ── */
    .result-empty {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 80px;
        border: 1px dashed #dadce0;
        border-radius: 12px;
        margin-top: 14px;
        background: #f8f9fa;
    }

    .result-empty p {
        font-family: 'Inter', sans-serif;
        font-size: 13px;
        color: #9aa0a6;
        text-align: center;
        font-weight: 400;
    }

    /* ── Footer / Contact Section ── */
    .site-footer {
        border-top: 1px solid #f1f3f4;
        padding: 28px 0 40px;
        margin-top: 40px;
    }

    .footer-content {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        flex-wrap: wrap;
        gap: 20px;
    }

    .footer-formats {
        font-family: 'Inter', sans-serif;
        font-size: 12px;
        color: #9aa0a6;
        font-weight: 400;
    }

    .footer-contact {
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        align-items: center;
    }

    .footer-contact a {
        font-family: 'Inter', sans-serif;
        font-size: 12px;
        color: #5f6368;
        text-decoration: none;
        display: flex;
        align-items: center;
        gap: 5px;
        transition: color 0.15s;
        font-weight: 500;
    }

    .footer-contact a:hover {
        color: #1a73e8;
    }

    .footer-divider {
        width: 1px;
        height: 14px;
        background: #e8eaed;
    }

    .gr-row { gap: 24px !important; }
    #result-html > div { margin: 0 !important; }
    """

    EMPTY_RESULT = """
<div class="result-empty">
    <p>Upload audio and click process to see results</p>
</div>
"""

    with gr.Blocks(theme=theme, css=css, title="Remove Silence") as demo:

        gr.HTML("""
        <div class="yt-banner">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="#ea4335">
                <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/>
            </svg>
            <p>Subscribe to <a href="https://www.youtube.com/@neuralfalcon/" target="_blank">Neural Falcon on YouTube</a> for more AI tools and tutorials</p>
        </div>
        """)

        gr.HTML("""
        <div class="site-header">
            <div class="header-title">REMOVE <span>SILENCE</span></div>
            <div class="header-sub">Drop your audio, get it tight - perfect for Shorts, TikTok &amp; Reels</div>
            <div class="header-badges">
                <span class="hbadge">100% Free</span>
                <span class="hbadge">No Sign-up</span>
                <span class="hbadge">AI Powered</span>
            </div>
        </div>
        """)

        with gr.Row(equal_height=False):

            with gr.Column(scale=1):
                gr.HTML('<div class="section-tag">Upload</div>')
                audio_input = gr.Audio(
                    label="",
                    type="filepath",
                    sources=["upload", "microphone"],
                    show_label=False,
                )

                gr.HTML('<div class="hdivider"></div>')
                gr.HTML('<div class="section-tag">Mode</div>')

                method_choice = gr.Radio(
                    choices=["Super Strict", "Human Speech Only (AI)"],
                    value="Super Strict",
                    label="",
                    show_label=False,
                    elem_classes=["gr-radio-group"]
                )

                gr.HTML('<div class="hdivider"></div>')
                silence_threshold = gr.Number(
                                    label="Keep Silence (seconds)",
                                    value=0.05,
                                    info="Lower = tighter cut. For Shorts/TikTok try 0.03-0.05"
                                )
  
                gr.HTML('<div style="height:12px;"></div>')

                with gr.Row(elem_classes=["submit-row"]):
                    submit_btn = gr.Button(
                        "Remove Silence",
                        variant="primary",
                        size="lg"
                    )

            with gr.Column(scale=1):
                gr.HTML('<div class="section-tag">Result</div>')

                audio_output = gr.Audio(
                    label="Processed Audio",
                    show_label=True,
                )

                file_output = gr.File(
                    label="Download",
                    show_label=True,
                )

                result_stats = gr.HTML(
                    value=EMPTY_RESULT,
                    elem_id="result-html"
                )

        gr.HTML("""
        <div class="site-footer">
            <div class="footer-content">
                <span class="footer-formats">Supported: MP3, WAV, OGG, FLAC, M4A, and more</span>
                <div class="footer-contact">
                    <a href="mailto:NeuralFalcon@proton.me" title="Email">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/></svg>
                        Email
                    </a>
                    <div class="footer-divider"></div>
                    <a href="https://x.com/NeuralFalcon" target="_blank" title="X / Twitter">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
                        Twitter
                    </a>
                    <div class="footer-divider"></div>
                    <a href="https://github.com/NeuralFalconYT/Remove-Silence-From-Audio" target="_blank" title="Install Locally">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                        GitHub
                    </a>
                </div>
            </div>
        </div>
        """)

        def process_wrapper(audio_file, seconds, method):
            return process_audio(audio_file, seconds, method)

        submit_btn.click(
            fn=process_wrapper,
            inputs=[audio_input, silence_threshold, method_choice],
            outputs=[audio_output, file_output, result_stats]
        )

    return demo


# ─── Launch ───────────────────────────────────────────────────────────────────

start_cleanup_worker()
demo = ui()
demo.queue().launch()
