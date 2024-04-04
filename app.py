import gradio as gr
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import os
import tempfile
import soundfile as sf
from tqdm import tqdm
import shutil
import re
from sys import platform

SAMPLING_RATE = 16000

model_name = 'ivrit-ai/whisper-large-v2-tuned'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
processor = WhisperProcessor.from_pretrained(model_name)

def find_silent_sections(audio_data, sr, min_silence_length=0.5, silence_threshold=-40):
    """
    Find silent sections in an audio file.
    
    :param audio_data: Numpy array of the audio data.
    :param sr: Sampling rate of the audio data.
    :param min_silence_length: Minimum length of silence to detect (in seconds).
    :param silence_threshold: The threshold (in dB) below which, the segment is considered silent.
    :return: A list of tuples indicating the start and end samples of silent sections.
    """
    # Identify silent sections
    silent_sections = librosa.effects.split(audio_data, top_db=-silence_threshold)
    # Filter out short silent sections
    silent_sections = [s for s in silent_sections if (s[1] - s[0]) >= min_silence_length * sr]
    
    return silent_sections

def split_into_paragraphs(text, min_words_per_paragraph=20):
    """
    Split the text into paragraphs based on sentence endings and paragraph length.
    
    :param text: The input text to be split into paragraphs.
    :param min_words_per_paragraph: The minimum number of words required in a paragraph.
    :return: A list of paragraphs.
    """
    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    paragraphs = []
    current_paragraph = ""
    
    for sentence in sentences:
        current_paragraph += sentence + " "
        
        if len(current_paragraph.split()) >= min_words_per_paragraph:
            paragraphs.append(current_paragraph.strip())
            current_paragraph = ""
    
    if current_paragraph:
        paragraphs.append(current_paragraph.strip())
    
    return paragraphs

def transcribe_and_translate(audio_file, source_language):
    if not source_language or (isinstance(source_language, list) and not source_language[0]):
        return "Source language was not selected. Please choose a language."

    if isinstance(source_language, list):
        source_language = source_language[0]
    
    audio, rate = librosa.load(audio_file, sr=None)

    if rate != SAMPLING_RATE:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=SAMPLING_RATE)
    

    if platform == "linux" or platform == "linux2":
        temp_directory="/tmp/hebrew_wispher/"
    else:
        temp_directory="D:\\hebrew wispher\\"

    temp_dir = tempfile.mkdtemp(dir=temp_directory)
    
    chunk_duration = 30  # Duration in seconds
    chunks = []

    silent_sections = find_silent_sections(audio, SAMPLING_RATE, min_silence_length=0.3, silence_threshold=-40)
    start_idx = 0
    for end_idx in [s[0] for s in silent_sections]:
        if end_idx - start_idx > SAMPLING_RATE * chunk_duration:
            chunks.append(audio[start_idx:end_idx])
            start_idx = end_idx

    # Handling the last chunk
    if len(audio) - start_idx > 0:
        chunks.append(audio[start_idx:])

    transcribed_text = ""
    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")
        sf.write(chunk_path, chunk, samplerate=SAMPLING_RATE)
        
        chunk_audio, _ = librosa.load(chunk_path, sr=SAMPLING_RATE)
        
        input_features = processor(chunk_audio, sampling_rate=SAMPLING_RATE, return_tensors="pt").input_features.to(device)
        
        predicted_ids = model.generate(input_features, language=source_language, num_beams=5)
        chunk_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        transcribed_text += chunk_text + " "
        print(f"Processed chunk {i+1}/{len(chunks)}")
    
    output_dir = temp_directory + "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Split the transcribed text into paragraphs
    paragraphs = split_into_paragraphs(transcribed_text)
    
    text_path = os.path.join(output_dir, "transcribed_text.txt")
    with open(text_path, "w", encoding="utf-8") as file:
        for paragraph in paragraphs:
            file.write(paragraph + "\n\n")
    
    # Delete temporary WAV files
    for file in os.listdir(temp_dir):
        if file.endswith(".wav"):
            os.remove(os.path.join(temp_dir, file))
    
    # Delete the temporary directory
    shutil.rmtree(temp_dir)
    
    return "\n\n".join(paragraphs)

title = "Unlimited Length Transcription and Translation"
description = "With ivrit-ai/whisper-large-v2-tuned | GUI by Shmuel Ronen"

interface = gr.Interface(
    fn=transcribe_and_translate,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio File"),
        gr.Dropdown(choices=['Hebrew', 'English', 'Spanish', 'French'], label="Source Language")
    ],
    outputs=gr.Textbox(label="Transcription / Translation"),
    title=title,
    description=description
)

is_share_link = os.environ.get("IS_SHARE_LINK", "False")
print(f"is_share_link={is_share_link}")

if is_share_link == "True":
    interface.launch(share=True)
else:
    interface.launch()
