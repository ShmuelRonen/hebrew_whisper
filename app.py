from googletrans import Translator
import gradio as gr
import librosa
import numpy as np
import os
import shutil
import soundfile as sf
import tempfile
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

SAMPLING_RATE = 16000
model_name = 'ivrit-ai/whisper-large-v2-tuned'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
processor = WhisperProcessor.from_pretrained(model_name)
translator = Translator()

def transcribe(audio_file):
    audio, rate = librosa.load(audio_file, sr=None)
    if rate != SAMPLING_RATE:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=SAMPLING_RATE)

    temp_dir = tempfile.mkdtemp()
    chunks = np.array_split(audio, indices_or_sections=int(np.ceil(len(audio) / (SAMPLING_RATE * 30))))  # 30s chunks
    transcribed_text = ""

    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")
        sf.write(chunk_path, chunk, samplerate=SAMPLING_RATE)
        chunk_audio, _ = librosa.load(chunk_path, sr=SAMPLING_RATE)
        input_features = processor(chunk_audio, sampling_rate=SAMPLING_RATE, return_tensors="pt").input_features.to(device)
        predicted_ids = model.generate(input_features, num_beams=5)
        chunk_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        transcribed_text += chunk_text + " "

    shutil.rmtree(temp_dir)
    return transcribed_text

def translate_text(text, target_lang):
    # Google Translate's 'he' is for Hebrew and 'en' for English.
    translations = {'Hebrew': 'he', 'English': 'en', 'Spanish': 'es', 'French': 'fr'}
    translated_text = translator.translate(text, dest=translations[target_lang]).text
    return translated_text

def transcribe_and_translate(audio_file, target_language):
    transcribed_text = transcribe(audio_file)
    detected_language = translator.detect(transcribed_text).lang

    # Mapping of target language choices to Google Translate's language codes
    translations = {'Hebrew': 'he', 'English': 'en', 'Spanish': 'es', 'French': 'fr'}

    # If the detected language is Hebrew and the target language is also Hebrew, return the transcribed text as is
    if detected_language == 'he' and target_language == 'Hebrew':
        return transcribed_text
    # If the detected language is Hebrew but the target language is not Hebrew, translate the text
    elif detected_language == 'he' and target_language != 'Hebrew':
        transcribed_text = translate_text(transcribed_text, target_language)
    # If the detected language is not Hebrew and the target language is not the same as the detected language,
    # translate the text. This is for English transcriptions that need to be translated to another language.
    elif detected_language != 'he' and translations[target_language] != detected_language:
        transcribed_text = translate_text(transcribed_text, target_language)

    return transcribed_text


title = "Unlimited Length Transcription and Translation"
description = "With ivrit-ai/whisper-large-v2-tuned | GUI by Shmuel Ronen"

interface = gr.Interface(
    fn=transcribe_and_translate,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio File"),
        gr.Dropdown(choices=['Hebrew', 'English', 'Spanish', 'French'], label="Target Language")
    ],
    outputs=gr.Textbox(label="Transcription / Translation"),
    title=title,
    description=description
)

if __name__ == "__main__":
    interface.launch()
