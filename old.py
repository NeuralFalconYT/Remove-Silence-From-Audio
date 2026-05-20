import gradio as gr
import os
import uuid
from pydub import AudioSegment
from pydub.silence import split_on_silence
import re 
import time
import subprocess
import threading

# gr.close_all()

def clean_file_name(file_path):
    file_name = os.path.basename(file_path)
    file_name, file_extension = os.path.splitext(file_name)

    cleaned = re.sub(r'[^a-zA-Z\d]+', '_', file_name)

    clean_file_name = re.sub(r'_+', '_', cleaned).strip('_')

    if clean_file_name.endswith('_tmp'):
        clean_file_name = clean_file_name[:-4]

    random_uuid = uuid.uuid4().hex[:6]

    clean_file_path = os.path.join(
        os.path.dirname(file_path), 
        f"{clean_file_name}_{random_uuid}{file_extension}"
    )

    return clean_file_path


# def remove_silence(file_path, minimum_silence=50):
#     sound = AudioSegment.from_file(file_path)  # auto-detects format
#     audio_chunks = split_on_silence(sound,
#                                     min_silence_len=100,
#                                     silence_thresh=-45,
#                                     keep_silence=minimum_silence) 
#     combined = AudioSegment.empty()
#     for chunk in audio_chunks:
#         combined += chunk
#     output_path=clean_file_name(file_path)        
#     combined.export(output_path)  
#     return output_path


def remove_silence(file_path, minimum_silence=50):
    sound = AudioSegment.from_file(file_path)
    
    # Try splitting with default -45 dBFS
    audio_chunks = split_on_silence(
        sound,
        min_silence_len=100,
        silence_thresh=-45,
        keep_silence=minimum_silence
    ) 
    
    # If no chunks were extracted (e.g. whole file is quieter than -45dBFS)
    # retry with a dynamic threshold relative to the audio's actual loudness
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

def calculate_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    duration_seconds = len(audio) / 1000.0  
    return duration_seconds

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
            [
                "ffmpeg",
                "-y",
                "-i", audio_path,
                wav_path
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        if os.path.isfile(wav_path) and os.path.getsize(wav_path) > 0:
            return wav_path

    except Exception as e:
        print(f"⚠️ FFmpeg conversion error: {e}")
        pass

    return None

def process_audio(audio_file, seconds=0.05):
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
        
        output_audio_file = remove_silence(
            audio_file,
            minimum_silence=keep_silence
        )
        
        track_file(output_audio_file)

        after = calculate_duration(output_audio_file)
        text = f"Old Duration: {before:.3f} Seconds \nNew Duration: {after:.3f} Seconds"

        return output_audio_file, output_audio_file, text
        
    except Exception as e:
        print(f"⚠️ Error processing audio: {e}")
        return None, None, f"An error occurred during processing: {str(e)}"

def ui():
    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"]
    )

    css = """
    .gradio-container {max-width: none !important;}
    .tab-content {padding: 20px;}

    /* Primary button - BLUE by default */
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
                Upload an Audio file, and it will remove silent parts from it.
            </p>
            <p style="font-size:0.8em; color:#999;">
            ⚠️ Please don’t upload copyrighted content — it can take this Space offline.
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

                silence_threshold = gr.Number(
                    label="Keep Silence Upto (In seconds)",
                    value=0.05
                )

                submit_btn = gr.Button(
                    "🔇 Remove Silence",
                    variant="primary"
                )

            with gr.Column(scale=1):
                audio_output = gr.Audio(label="Play Audio")
                file_output = gr.File(label="Download Audio File")
                duration_output = gr.Textbox(label="Duration", lines=2)

        submit_btn.click(
            fn=process_audio,   
            inputs=[audio_input, silence_threshold],
            outputs=[audio_output, file_output, duration_output]
        )

    return demo

start_cleanup_worker()

demo= ui()
demo.queue().launch()
# demo.queue().launch(debug=True, share=True)
