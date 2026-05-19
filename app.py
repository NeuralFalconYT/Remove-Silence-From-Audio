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
<div style="margin-top:14px; border:1px solid #00ffe720; border-radius:10px; overflow:hidden; background:#050d14; position:relative;">
    <div style="position:absolute; top:0; left:0; right:0; height:1px; background:linear-gradient(90deg,transparent,#00ffe7,transparent);"></div>
    <div style="display:grid; grid-template-columns:1fr 1fr 1fr;">
        <div style="padding:22px 16px; text-align:center; border-right:1px solid #00ffe715;">
            <div style="font-size:9px; letter-spacing:0.22em; color:#1e5550; text-transform:uppercase; margin-bottom:10px; font-family:'Courier New',monospace;">Original</div>
            <div style="font-size:22px; font-weight:400; color:#4a9e9a; letter-spacing:0.04em; font-family:'Courier New',monospace;">{fmt(before)}</div>
        </div>
        <div style="padding:22px 16px; text-align:center; border-right:1px solid #00ffe715; background:#081820;">
            <div style="font-size:9px; letter-spacing:0.22em; color:#1e5550; text-transform:uppercase; margin-bottom:10px; font-family:'Courier New',monospace;">New</div>
            <div style="font-size:22px; font-weight:400; color:#00ffe7; letter-spacing:0.04em; font-family:'Courier New',monospace;">{fmt(after)}</div>
        </div>
        <div style="padding:22px 16px; text-align:center;">
            <div style="font-size:9px; letter-spacing:0.22em; color:#1e5550; text-transform:uppercase; margin-bottom:10px; font-family:'Courier New',monospace;">Removed</div>
            <div style="font-size:22px; font-weight:400; color:#ff4d6d; letter-spacing:0.04em; font-family:'Courier New',monospace;">{percent:.1f}%</div>
            <div style="font-size:11px; color:#1e4a48; margin-top:4px; font-family:'Courier New',monospace;">{fmt(removed)}</div>
        </div>
    </div>
    <div style="padding:8px 16px; background:#020a10; border-top:1px solid #00ffe710; text-align:right;">
        <span style="font-family:'Courier New',monospace; font-size:9px; color:#1a4040; letter-spacing:0.18em; text-transform:uppercase;">MODE // {mode_label}</span>
    </div>
</div>
"""
        return output_audio_file, output_audio_file, result_html

    except Exception as e:
        return None, None, f"<p style='color:#ff4d6d; font-family:monospace; font-size:12px; margin-top:12px;'>ERROR: {str(e)}</p>"


# ─── UI ───────────────────────────────────────────────────────────────────────

def ui():
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.slate,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Share Tech Mono"), "monospace"],
        font_mono=[gr.themes.GoogleFont("Share Tech Mono"), "monospace"],
    ).set(
        body_background_fill="#030b11",
        body_background_fill_dark="#030b11",
        block_background_fill="#050d16",
        block_background_fill_dark="#050d16",
        block_border_width="1px",
        block_border_color="#00ffe720",
        block_border_color_dark="#00ffe720",
        block_radius="10px",
        block_shadow="none",
        block_shadow_dark="none",
        block_label_background_fill="#050d16",
        block_label_background_fill_dark="#050d16",
        block_label_border_width="0px",
        block_label_text_color="#2a6e6a",
        block_label_text_color_dark="#2a6e6a",
        block_label_text_size="11px",
        block_title_text_color="#3a8e8a",
        block_title_text_color_dark="#3a8e8a",
        block_title_text_size="11px",
        body_text_color="#7adbd6",
        body_text_color_dark="#7adbd6",
        body_text_color_subdued="#2a6e6a",
        body_text_color_subdued_dark="#2a6e6a",
        body_text_size="13px",
        input_background_fill="#020a10",
        input_background_fill_dark="#020a10",
        input_background_fill_focus="#050f18",
        input_background_fill_focus_dark="#050f18",
        input_border_color="#00ffe725",
        input_border_color_dark="#00ffe725",
        input_border_color_focus="#00ffe760",
        input_border_color_focus_dark="#00ffe760",
        input_border_width="1px",
        input_radius="8px",
        input_shadow="none",
        input_shadow_dark="none",
        input_text_size="13px",
        input_placeholder_color="#1e5550",
        input_placeholder_color_dark="#1e5550",
        button_primary_background_fill="#00ffe7",
        button_primary_background_fill_dark="#00ffe7",
        button_primary_background_fill_hover="#33fff0",
        button_primary_background_fill_hover_dark="#33fff0",
        button_primary_text_color="#020a10",
        button_primary_text_color_dark="#020a10",
        button_primary_border_color="#00ffe7",
        button_primary_border_color_dark="#00ffe7",
        button_secondary_background_fill="#050d16",
        button_secondary_background_fill_dark="#050d16",
        button_secondary_background_fill_hover="#081820",
        button_secondary_background_fill_hover_dark="#081820",
        button_secondary_text_color="#3a8e8a",
        button_secondary_text_color_dark="#3a8e8a",
        button_secondary_border_color="#00ffe720",
        button_secondary_border_color_dark="#00ffe720",
        button_large_radius="8px",
        button_large_text_size="12px",
        button_large_padding="14px 28px",
        slider_color="#00ffe7",
        slider_color_dark="#00ffe7",
        checkbox_background_color="#020a10",
        checkbox_background_color_dark="#020a10",
        checkbox_border_color="#00ffe725",
        checkbox_border_color_dark="#00ffe725",
        checkbox_border_color_selected="#00ffe7",
        checkbox_border_color_selected_dark="#00ffe7",
        checkbox_label_background_fill="#020a10",
        checkbox_label_background_fill_dark="#020a10",
        checkbox_label_background_fill_selected="#061418",
        checkbox_label_background_fill_selected_dark="#061418",
        checkbox_label_border_color="#00ffe720",
        checkbox_label_border_color_dark="#00ffe720",
        checkbox_label_border_color_hover="#00ffe750",
        checkbox_label_border_color_hover_dark="#00ffe750",
        checkbox_label_text_color="#2a6e6a",
        checkbox_label_text_color_dark="#2a6e6a",
        checkbox_label_text_color_selected="#00ffe7",
        checkbox_label_text_color_selected_dark="#00ffe7",
    )

    css = """
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

    .gradio-container {
        max-width: 100% !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 28px !important;
        box-sizing: border-box !important;
        background: #030b11 !important;
    }

    /* Scanline effect */
    body {
        background: #030b11 !important;
    }
    body::after {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: repeating-linear-gradient(
            0deg,
            transparent,
            transparent 3px,
            rgba(0,255,231,0.012) 3px,
            rgba(0,255,231,0.012) 4px
        );
        pointer-events: none;
        z-index: 9999;
    }

    /* ── Header — centered ── */
    .site-header {
        padding: 64px 0 52px;
        border-bottom: 1px solid #00ffe715;
        margin-bottom: 48px;
        text-align: center;
    }

    .header-eyebrow {
        font-family: 'Share Tech Mono', monospace;
        font-size: 10px;
        letter-spacing: 0.35em;
        color: #1a4a48;
        text-transform: uppercase;
        margin-bottom: 20px;
    }

    .header-title {
        font-family: 'Orbitron', monospace;
        font-size: clamp(30px, 5vw, 60px);
        font-weight: 900;
        color: #cef5f3;
        letter-spacing: 0.08em;
        line-height: 1.0;
        margin-bottom: 16px;
    }

    .header-title span {
        color: #00ffe7;
    }

    .header-sub {
        font-family: 'Share Tech Mono', monospace;
        font-size: 13px;
        color: #2a7a74;
        margin-bottom: 28px;
        letter-spacing: 0.08em;
    }

    .header-badges {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        justify-content: center;
    }

    .hbadge {
        font-family: 'Share Tech Mono', monospace;
        font-size: 10px;
        letter-spacing: 0.18em;
        padding: 5px 16px;
        border: 1px solid #00ffe730;
        color: #00ffe790;
        border-radius: 2px;
        background: #00ffe708;
        text-transform: uppercase;
    }

    /* ── Section labels ── */
    .section-tag {
        font-family: 'Share Tech Mono', monospace;
        font-size: 9px;
        letter-spacing: 0.28em;
        color: #1a4a48;
        text-transform: uppercase;
        margin-bottom: 14px;
        padding-bottom: 10px;
        border-bottom: 1px solid #00ffe710;
    }

    /* ── Radio ── */
    .gr-radio-group .wrap { gap: 8px !important; }
    .gr-radio-group label {
        border-radius: 6px !important;
        padding: 10px 18px !important;
        font-size: 12px !important;
        letter-spacing: 0.1em !important;
        font-family: 'Share Tech Mono', monospace !important;
        transition: all 0.15s !important;
        text-transform: uppercase !important;
    }

    /* ── Submit button ── */
    .submit-row button {
        width: 100% !important;
        height: 54px !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        letter-spacing: 0.28em !important;
        text-transform: uppercase !important;
        border-radius: 6px !important;
        font-family: 'Orbitron', monospace !important;
    }

    /* ── Divider ── */
    .hdivider {
        height: 1px;
        background: #00ffe710;
        margin: 26px 0;
    }

    /* ── Result placeholder ── */
    .result-empty {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 80px;
        border: 1px dashed #00ffe718;
        border-radius: 8px;
        margin-top: 14px;
    }

    .result-empty p {
        font-family: 'Share Tech Mono', monospace;
        font-size: 11px;
        color: #174040;
        letter-spacing: 0.14em;
        text-align: center;
        text-transform: uppercase;
    }

    /* ── Footer ── */
    .site-footer {
        border-top: 1px solid #00ffe710;
        padding: 22px 0 44px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 44px;
    }

    .footer-l {
        font-family: 'Share Tech Mono', monospace;
        font-size: 10px;
        color: #1a4040;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .footer-r a {
        font-family: 'Share Tech Mono', monospace;
        font-size: 10px;
        color: #2a6060;
        letter-spacing: 0.1em;
        text-decoration: underline;
        text-underline-offset: 3px;
        text-transform: uppercase;
    }

    .gr-row { gap: 24px !important; }
    #result-html > div { margin: 0 !important; }
    """

    EMPTY_RESULT = """
<div class="result-empty">
    <p>// upload audio &rarr; execute</p>
</div>
"""

    with gr.Blocks(theme=theme, css=css, title="Remove Silence") as demo:

        gr.HTML("""
        <div class="site-header">
            <div class="header-eyebrow">// audio processing tool v2.0 //</div>
            <div class="header-title">REMOVE <span>SILENCE</span></div>
            <div class="header-sub">Drop your audio &nbsp;&bull;&nbsp; get it tight &nbsp;&bull;&nbsp; perfect for Shorts, TikTok &amp; Reels</div>
            <div class="header-badges">
                <span class="hbadge">100% free</span>
                <span class="hbadge">no sign-up</span>
                <span class="hbadge">ai powered</span>
            </div>
        </div>
        """)

        with gr.Row(equal_height=False):

            with gr.Column(scale=1):
                gr.HTML('<div class="section-tag">[ 01 ] &nbsp; upload</div>')
                audio_input = gr.Audio(
                    label="",
                    type="filepath",
                    sources=["upload", "microphone"],
                    show_label=False,
                )

                gr.HTML('<div class="hdivider"></div>')
                gr.HTML('<div class="section-tag">[ 02 ] &nbsp; mode</div>')

                method_choice = gr.Radio(
                    choices=["⚡ Super Strict", "🧠 Human Speech Only (AI)"],
                    value="⚡ Super Strict",
                    label="",
                    show_label=False,
                    elem_classes=["gr-radio-group"]
                )

                gr.HTML('<div class="hdivider"></div>')
                silence_threshold = gr.Number(
                                    label="Keep Silence (seconds)",
                                    value=0.05,
                                    info="lower = tighter cut · for shorts / TikTok try 0.03–0.05"
                                )
  
                gr.HTML('<div style="height:12px;"></div>')

                with gr.Row(elem_classes=["submit-row"]):
                    submit_btn = gr.Button(
                        "▶  REMOVE SILENCE",
                        variant="primary",
                        size="lg"
                    )

            with gr.Column(scale=1):
                gr.HTML('<div class="section-tag">[ 03 ] &nbsp; result</div>')

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
            <span class="footer-l">mp3 &bull; wav &bull; ogg &bull; flac &bull; m4a &bull; and more</span>
            <span class="footer-r">
                <a href="https://github.com/NeuralFalconYT/Remove-Silence-From-Audio" target="_blank">install locally &rarr;</a>
                &nbsp;&nbsp;
                <a href="#">no copyrighted content</a>
            </span>
        </div>
        """)

        def process_wrapper(audio_file, seconds, method):
            clean_method = method.replace("⚡ ", "").replace("🧠 ", "")
            return process_audio(audio_file, seconds, clean_method)

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
