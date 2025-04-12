
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

def ui():
    demo = gr.Interface(
    fn=process_audio,
    inputs=[
        gr.Audio(label="Upload Audio", type="filepath", sources=['upload', 'microphone']),
        gr.Number(label="Keep Silence Upto (In seconds)", value=0.05)
    ],
    outputs=[
        gr.Audio(label="Play Audio"),
        gr.File(label="Download Audio File"),
        gr.Textbox(label="Duration")
    ],
    title="Remove Silence From Audio",
    description="Upload an MP3 or WAV file, and it will remove silent parts from it.",
    #theme='NoCrypt/miku'
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
