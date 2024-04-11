from googletrans import Translator
import gradio as gr
import librosa
import numpy as np
import os
import re
import shutil
import soundfile as sf
import tempfile
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import whisper
import datetime

SAMPLING_RATE = 16000
model_name = 'ivrit-ai/whisper-large-v2-tuned'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
processor = WhisperProcessor.from_pretrained(model_name)
translator = Translator()

def transcribe(audio_numpy, sampling_rate=16000):
    if audio_numpy.ndim > 1:
        audio_numpy = audio_numpy.mean(axis=1)

    temp_dir = tempfile.mkdtemp()
    chunks = np.array_split(audio_numpy, indices_or_sections=int(np.ceil(len(audio_numpy) / (sampling_rate * 30))))  # 30s chunks
    transcribed_text = ""

    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")
        sf.write(chunk_path, chunk, samplerate=sampling_rate)
        input_features = processor(chunk, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)
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
    current_paragraph = []

    for sentence in sentences:
        words_in_sentence = sentence.split()
        current_paragraph.extend(words_in_sentence)
        if len(current_paragraph) >= min_words_per_paragraph:
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []

    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))

    return '\n\n'.join(paragraphs)

from pydub import AudioSegment

def generate_srt_content(audio_file_path, target_language='Hebrew', max_line_length=50):
    print("Starting transcription and translation process...")

    audio = AudioSegment.from_file(audio_file_path)
    audio_numpy = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    audio_numpy = librosa.resample(audio_numpy, orig_sr=audio.frame_rate, target_sr=16000)

    temp_file_name = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            temp_file_name = tmpfile.name
            sf.write(tmpfile.name, audio_numpy, 16000)

        transcription_result = whisper.load_model("large").transcribe(audio=temp_file_name)

        srt_content = ""
        for segment in transcription_result['segments']:
            start_time = str(datetime.timedelta(seconds=int(segment['start']))) + ',000'
            end_time = str(datetime.timedelta(seconds=int(segment['end']))) + ',000'
            text = segment['text']
            segment_id = segment['id'] + 1

            lines = []
            while len(text) > max_line_length:
                split_index = text.rfind(' ', 0, max_line_length)
                if split_index == -1:
                    split_index = max_line_length
                lines.append(text[:split_index].strip())
                text = text[split_index:].strip()
            lines.append(text)

            srt_entry = f"{segment_id}\n{start_time} --> {end_time}\n"
            
            translated_lines = []
            for line in lines:
                for attempt in range(3):  # Retry translation up to 3 times
                    try:
                        translated_line = translator.translate(line, dest='he').text
                        translated_lines.append(translated_line)
                        break
                    except Exception as e:
                        print(f"Translation failed (attempt {attempt+1}): {str(e)}")
                        if attempt < 2:
                            time.sleep(1)  # Delay before retrying
                        else:
                            translated_lines.append(line)  # Use original English line if translation fails

            srt_entry += "\n".join(translated_lines) + "\n\n"
            srt_content += srt_entry

        os.makedirs("output", exist_ok=True)
        srt_file_path = os.path.join("output", "output.srt")
        with open(srt_file_path, "w", encoding="utf-8") as srt_file:
            srt_file.write(srt_content)

        return srt_content

    finally:
        if temp_file_name:
            os.remove(temp_file_name)

def transcribe_and_translate(audio_file, target_language, generate_srt_checkbox):
    if not target_language:
        return "Please choose a Target Language"
    translations = {'Hebrew': 'he', 'English': 'en', 'Spanish': 'es', 'French': 'fr'}
    
    audio = AudioSegment.from_file(audio_file)
    audio_numpy = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    audio_numpy = librosa.resample(audio_numpy, orig_sr=audio.frame_rate, target_sr=16000)
    
    transcribed_text = transcribe(audio_numpy)
    detected_language_code = translator.detect(transcribed_text).lang

    if generate_srt_checkbox:
        srt_result = generate_srt_content(audio_file, 'Hebrew')
        return srt_result
    else:
        if isinstance(target_language, list):
            target_language = target_language[0]

        if translations.get(target_language) != detected_language_code:
            translated_text = translate_text(transcribed_text, target_language)
        else:
            translated_text = transcribed_text

        final_text = split_into_paragraphs(translated_text)

        os.makedirs("output", exist_ok=True)
        result_file_path = os.path.join("output", "result.txt")
        with open(result_file_path, "w", encoding="utf-8") as result_file:
            result_file.write(final_text)

        return final_text

title = "Unlimited Length Transcription and Translation"
description = "With ivrit-ai/whisper-large-v2-tuned | GUI by Shmuel Ronen"

interface = gr.Interface(
    fn=transcribe_and_translate,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio File"),
        gr.Dropdown(choices=['Hebrew', 'English', 'Spanish', 'French'], label="Target Language"),
        gr.Checkbox(label="Generate Hebrew SRT File")
    ],
    outputs=gr.Textbox(label="Transcription / Translation / SRT Result"),
    title=title,
    description=description
)

if __name__ == "__main__":
    interface.launch()