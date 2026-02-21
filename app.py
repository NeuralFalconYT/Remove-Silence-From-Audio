
import gradio as gr
import os
import uuid
from pydub import AudioSegment
from pydub.silence import split_on_silence
import re 

def clean_file_name(file_path):
    # Get the base file name and extension
    file_name = os.path.basename(file_path)
    file_name, file_extension = os.path.splitext(file_name)

    # Replace non-alphanumeric characters with an underscore
    cleaned = re.sub(r'[^a-zA-Z\d]+', '_', file_name)

    # Remove any multiple underscores
    clean_file_name = re.sub(r'_+', '_', cleaned).strip('_')

    # Generate a random UUID for uniqueness
    random_uuid = uuid.uuid4().hex[:6]

    # Combine cleaned file name with the original extension
    clean_file_path = os.path.join(os.path.dirname(file_path), clean_file_name + f"_{random_uuid}" + file_extension)

    return clean_file_path



def remove_silence(file_path, minimum_silence=50):
    sound = AudioSegment.from_file(file_path)  # auto-detects format
    audio_chunks = split_on_silence(sound,
                                    min_silence_len=100,
                                    silence_thresh=-45,
                                    keep_silence=minimum_silence) 
    combined = AudioSegment.empty()
    for chunk in audio_chunks:
        combined += chunk
    output_path=clean_file_name(file_path)        
    combined.export(output_path)  # format inferred from output file extension
    return output_path



def calculate_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    duration_seconds = len(audio) / 1000.0  # pydub uses milliseconds
    return duration_seconds


def process_audio(audio_file, seconds=0.05):
    keep_silence = int(seconds * 1000)
    output_audio_file = remove_silence(audio_file, minimum_silence=keep_silence)
    before = calculate_duration(audio_file)
    after = calculate_duration(output_audio_file)
    text = f"Old Duration: {before:.3f} seconds \nNew Duration: {after:.3f} seconds"
    return output_audio_file, output_audio_file, text

# def ui():
#     theme = gr.themes.Soft(font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"])
#     css = ".gradio-container {max-width: none !important;} .tab-content {padding: 20px;}"
#     demo = gr.Interface(
#     fn=process_audio,
#     inputs=[
#         gr.Audio(label="Upload Audio", type="filepath", sources=['upload', 'microphone']),
#         gr.Number(label="Keep Silence Upto (In seconds)", value=0.05)
#     ],
#     outputs=[
#         gr.Audio(label="Play Audio"),
#         gr.File(label="Download Audio File"),
#         gr.Textbox(label="Duration")
#     ],
#     title="Remove Silence From Audio",
#     description="Upload an MP3 or WAV file, and it will remove silent parts from it.",
#     theme=theme,
#     css=css,
#     )
#     return demo


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

        # Header
        gr.HTML("""
        <div style="text-align:center; margin:20px auto; max-width:800px;">
            <h1 style="font-size:2.4em; margin-bottom:6px;">
                🔇 Remove Silence From Audio
            </h1>

            <p style="font-size:1.05em; color:#555; margin:0 0 10px;">
                Upload an MP3 or WAV file, and it will remove silent parts from it.
            </p>

            <p style="font-size:0.9em; color:#777;">
                Made by 
                <a href="https://github.com/NeuralFalconYT" target="_blank" style="text-decoration:none;">
                    NeuralFalconYT
                </a>
            </p>
        </div>
        """)

        with gr.Row():
            # LEFT: Inputs
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

            # RIGHT: Outputs
            with gr.Column(scale=1):
                audio_output = gr.Audio(label="Play Audio")
                file_output = gr.File(label="Download Audio File")
                duration_output = gr.Textbox(label="Duration")

        submit_btn.click(
            fn=process_audio,   # <-- your function
            inputs=[audio_input, silence_threshold],
            outputs=[audio_output, file_output, duration_output]
        )

    return demo

import click
@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def main(debug, share):
    demo=ui()
    demo.queue().launch(debug=debug, share=share)
if __name__ == "__main__":
    main()    
