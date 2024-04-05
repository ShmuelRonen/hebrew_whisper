from googletrans import Translator
import gradio as gr
import librosa
import numpy as np
import os
import re  # Import the missing module
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
    translations = {'Hebrew': 'he', 'English': 'en', 'Spanish': 'es', 'French': 'fr'}
    translated_text = translator.translate(text, dest=translations[target_lang]).text
    return translated_text
    
def split_into_paragraphs(text, min_words_per_paragraph=20):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    paragraphs = []
    current_paragraph = []  # This is correctly a list, to accumulate sentences.

    for sentence in sentences:
        words_in_sentence = sentence.split()  # Split the sentence into words to count them.
        current_paragraph.extend(words_in_sentence)  # Extend the list of words in the current paragraph.
        # Check if the current paragraph has reached the minimum word count to form a paragraph.
        if len(current_paragraph) >= min_words_per_paragraph:
            paragraphs.append(' '.join(current_paragraph))  # Join words to form the paragraph text.
            current_paragraph = []  # Reset for the next paragraph.

    # After the loop, if there are any remaining words, form the final paragraph.
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))  # This joins the remaining words into a paragraph.

    return '\n\n'.join(paragraphs)  # Join paragraphs with two newlines.


def transcribe_and_translate(audio_file, target_language):
    translations = {'Hebrew': 'he', 'English': 'en', 'Spanish': 'es', 'French': 'fr'}

    transcribed_text = transcribe(audio_file)
    # Detect the primary language of the transcribed text
    detected_language_code = translator.detect(transcribed_text).lang

    # Account for both 'iw' and 'he' as codes for Hebrew
    if detected_language_code == 'iw' or detected_language_code == 'he':
        detected_language = 'Hebrew'
    else:
        detected_language = 'English' if detected_language_code == 'en' else None

    print(f"Detected Language: {detected_language}")  # Debug for verification

    if translations.get(target_language) != detected_language_code:
        transcribed_text = translate_text(transcribed_text, target_language)
    
    # Apply paragraph splitting based on detected language
    if detected_language == 'Hebrew':
        final_text = split_into_paragraphs(transcribed_text)
    elif detected_language == 'English' and target_language == 'English':
        final_text = split_into_paragraphs(transcribed_text)
    else:
        final_text = transcribed_text

    return final_text


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
