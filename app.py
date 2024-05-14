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
import time
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import whisper
import datetime
from pydub import AudioSegment

SAMPLING_RATE = 16000
english_model_name = 'large-v2'
hebrew_model_name = 'ivrit-ai/whisper-v2-d3-e3'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
english_model = whisper.load_model(english_model_name, device=device)
hebrew_processor = WhisperProcessor.from_pretrained(hebrew_model_name)
hebrew_model = WhisperForConditionalGeneration.from_pretrained(hebrew_model_name).to(device)
translator = Translator()

def is_hebrew(text):
    return bool(re.search(r'[\u0590-\u05FF]', text))

def format_text(text):
    if is_hebrew(text):
        return f'<div style="text-align: right; direction: rtl;">{text}</div>'
    else:
        return f'<div style="text-align: left; direction: ltr;">{text}</div>'

def get_audio_length(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path)
    return len(audio) / 1000.0

def transcribe(audio_numpy, sampling_rate=16000):
    if audio_numpy.ndim > 1:
        audio_numpy = audio_numpy.mean(axis=1)

    temp_dir = tempfile.mkdtemp()
    chunks = np.array_split(audio_numpy, indices_or_sections=int(np.ceil(len(audio_numpy) / (sampling_rate * 10))))
    transcribed_text = ""

    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")
        sf.write(chunk_path, chunk, samplerate=sampling_rate)
        input_features = hebrew_processor(chunk, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)
        predicted_ids = hebrew_model.generate(input_features, num_beams=5)
        chunk_text = hebrew_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        transcribed_text += chunk_text + " "

    shutil.rmtree(temp_dir)
    return transcribed_text

def translate_text(text, target_lang):
    translations = {'Hebrew': 'he', 'English': 'en', 'Spanish': 'es', 'French': 'fr'}
    translated_text = translator.translate(text, dest=translations[target_lang]).text
    return translated_text

def split_lines(text, max_line_length=40):
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        if len(' '.join(current_line + [word])) <= max_line_length:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

def format_srt_entry(segment_id, start_time, end_time, lines):
    if len(lines) > 2:
        lines = [' '.join(lines[i:i + 2]) for i in range(0, len(lines), 2)]
    srt_entry = f"{segment_id}\n{start_time} --> {end_time}\n" + "\n".join(lines) + "\n\n"
    return srt_entry

def generate_srt_content(audio_file_path, target_language='Hebrew', max_line_length=40):
    audio_length = get_audio_length(audio_file_path)
    audio = AudioSegment.from_file(audio_file_path)
    audio_numpy = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    audio_numpy = librosa.resample(audio_numpy, orig_sr=audio.frame_rate, target_sr=16000)

    temp_file_name = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            temp_file_name = tmpfile.name
            sf.write(tmpfile.name, audio_numpy, 16000)

        transcription_result = english_model.transcribe(temp_file_name)

        srt_content = ""
        previous_end_time = 0
        for segment in transcription_result['segments']:
            start_time_seconds = max(segment['start'], previous_end_time)
            end_time_seconds = min(segment['end'], audio_length)

            if end_time_seconds <= start_time_seconds:
                continue

            start_time = str(datetime.timedelta(seconds=start_time_seconds)).split(".")[0] + ',000'
            end_time = str(datetime.timedelta(seconds=end_time_seconds)).split(".")[0] + ',000'
            text = segment['text']
            segment_id = segment['id'] + 1
            previous_end_time = end_time_seconds

            lines = split_lines(text, max_line_length=max_line_length)

            translated_lines = []
            for line in lines:
                for attempt in range(3):
                    try:
                        translated_line = translate_text(line, 'Hebrew')
                        translated_lines.append(translated_line)
                        break
                    except Exception as e:
                        print(f"Translation failed (attempt {attempt+1}): {str(e)}")
                        if attempt < 2:
                            time.sleep(1)
                        else:
                            translated_lines.append(line)

            srt_entry = format_srt_entry(segment_id, start_time, end_time, translated_lines)
            srt_content += srt_entry

        os.makedirs("output", exist_ok=True)
        srt_file_path = os.path.join("output", "output.srt")
        with open(srt_file_path, "w", encoding="utf-8") as srt_file:
            srt_file.write(srt_content)

        srt_html_content = ""
        for line in srt_content.split('\n'):
            srt_html_content += f"<div>{line}</div>"

        return format_text(srt_html_content)

    finally:
        if temp_file_name:
            os.remove(temp_file_name)

def transcribe_and_translate(audio_file, target_language, generate_srt_checkbox):
    if not target_language:
        return format_text("Please choose a Target Language")

    translations = {'Hebrew': 'he', 'English': 'en', 'Spanish': 'es', 'French': 'fr'}
    audio_length = get_audio_length(audio_file)

    audio = AudioSegment.from_file(audio_file)
    audio_numpy = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    audio_numpy = librosa.resample(audio_numpy, orig_sr=audio.frame_rate, target_sr=16000)

    if torch.cuda.is_available():
        print("GPU is available")
        gpu_info = torch.cuda.get_device_properties(0)
        print(f"GPU name: {gpu_info.name}")
        print(f"GPU memory usage before transcription:")
        print(torch.cuda.memory_summary(device=None, abbreviated=False))

    transcribed_text = transcribe(audio_numpy)

    if torch.cuda.is_available():
        print(f"GPU memory usage after transcription:")
        print(torch.cuda.memory_summary(device=None, abbreviated=False))

    if generate_srt_checkbox:
        srt_result = generate_srt_content(audio_file, target_language)
        return f"Audio Length: {audio_length} seconds\n\n" + srt_result
    else:
        if isinstance(target_language, list):
            target_language = target_language[0]

        translated_text = transcribed_text
        if translations.get(target_language) != 'he':
            translated_text = translate_text(transcribed_text, target_language)
        else:
            translated_text = translate_text(transcribed_text, target_language)

        final_text = split_lines(translated_text)
        os.makedirs("output", exist_ok=True)
        result_file_path = os.path.join("output", "result.txt")
        with open(result_file_path, "w", encoding="utf-8") as result_file:
            result_file.write('\n\n'.join(final_text))

        return f"Audio Length: {audio_length} seconds\n\n" + format_text('\n\n'.join(final_text))

title = "Unlimited Length Transcription and Translation"
description = "With: ivrit-ai/whisper-v2-d3-e3 | GUI by Shmuel Ronen"

interface = gr.Interface(
    fn=transcribe_and_translate,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio File"),
        gr.Dropdown(choices=['Hebrew', 'English', 'Spanish', 'French'], label="Target Language"),
        gr.Checkbox(label="Generate Hebrew SRT File")
    ],
    outputs=gr.HTML(label="Transcription / Translation / SRT Result"),
    title=title,
    description=description
)

interface.css = """
    #output_text, #output_text * {
        text-align: right !important;
        direction: rtl !important;
    }
"""

if __name__ == "__main__":
    interface.launch()
