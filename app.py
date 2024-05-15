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
import whisper
import datetime
from pydub import AudioSegment
from transformers import WhisperProcessor, WhisperForConditionalGeneration

SAMPLING_RATE = 16000
general_model_name = 'large-v2'
hebrew_model_name = 'ivrit-ai/whisper-v2-d3-e3'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
general_model = whisper.load_model(general_model_name, device=device)
hebrew_processor = WhisperProcessor.from_pretrained(hebrew_model_name)
hebrew_model = WhisperForConditionalGeneration.from_pretrained(hebrew_model_name).to(device)
translator = Translator()

def is_hebrew(text):
    return bool(re.search(r'[\u0590-\u05FF]', text))

def is_arabic(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))

def format_text(text):
    if is_hebrew(text):
        return f'<div style="text-align: right; direction: rtl;">{text}</div>'
    elif is_arabic(text):
        return f'<div style="text-align: left; direction: rtl;">{text}</div>'
    else:
        return f'<div style="text-align: left; direction: ltr;">{text}</div>'

def get_audio_length(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path)
    return len(audio) / 1000.0

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
    srt_entry = f"{segment_id}\n{start_time} --> {end_time}\n" + "\n".join(lines) + "\n\n"
    return srt_entry

def transcribe_with_model(audio_file_path, model_choice):
    if model_choice == 'General Model':
        return transcribe_with_general_model(audio_file_path)
    else:
        return transcribe_with_hebrew_model(audio_file_path)

def transcribe_with_general_model(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path)
    audio_numpy = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    audio_numpy = librosa.resample(audio_numpy, orig_sr=audio.frame_rate, target_sr=16000)

    temp_file_name = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            temp_file_name = tmpfile.name
            sf.write(temp_file_name, audio_numpy, 16000)

        transcription_result = general_model.transcribe(temp_file_name, language="he")
        return transcription_result
    finally:
        if temp_file_name:
            os.remove(temp_file_name)

def transcribe_with_hebrew_model(audio_file_path):
    audio, sr = librosa.load(audio_file_path, sr=SAMPLING_RATE)
    audio_numpy = np.array(audio)
    temp_dir = tempfile.mkdtemp()
    transcribed_segments = []

    for i in range(0, len(audio_numpy), SAMPLING_RATE * 30):
        chunk = audio_numpy[i:i + SAMPLING_RATE * 30]
        chunk_path = os.path.join(temp_dir, f"chunk_{i // (SAMPLING_RATE * 30)}.wav")
        sf.write(chunk_path, chunk, samplerate=SAMPLING_RATE)
        input_features = hebrew_processor(chunk, sampling_rate=SAMPLING_RATE, return_tensors="pt").input_features.to(device)
        predicted_ids = hebrew_model.generate(input_features, num_beams=5)
        chunk_text = hebrew_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        transcribed_segments.append({
            "start": i / SAMPLING_RATE,
            "end": min((i + SAMPLING_RATE * 30) / SAMPLING_RATE, len(audio_numpy) / SAMPLING_RATE),
            "text": chunk_text,
            "id": i // (SAMPLING_RATE * 30)
        })

    shutil.rmtree(temp_dir)
    return {"segments": transcribed_segments}

def translate_text(text, target_lang):
    translations = {'Hebrew': 'he', 'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de', 'Portuguese': 'pt', 'Arabic': 'ar'}
    translated_text = translator.translate(text, dest=translations[target_lang]).text
    return translated_text

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

def transcribe_and_translate(audio_file, target_language, model_choice, generate_srt_checkbox):
    if not target_language:
        return format_text("Please choose a Target Language")

    audio_length = get_audio_length(audio_file)

    if torch.cuda.is_available():
        print("GPU is available")
        gpu_info = torch.cuda.get_device_properties(0)
        print(f"GPU name: {gpu_info.name}")
        print(f"GPU memory usage before transcription:")
        print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # Always transcribe with the general model to get accurate timestamps
    general_transcription_result = transcribe_with_general_model(audio_file)
    general_segments = general_transcription_result['segments']
    total_general_duration = sum([segment['end'] - segment['start'] for segment in general_segments])

    # Accumulator for proportional timings
    proportional_timings = []
    cumulative_duration = 0.0
    for general_segment in general_segments:
        general_duration = general_segment['end'] - general_segment['start']
        proportional_duration = (general_duration / total_general_duration) * audio_length
        proportional_timings.append((cumulative_duration, cumulative_duration + proportional_duration))
        cumulative_duration += proportional_duration

    if model_choice == 'General Model':
        return process_general_model(general_segments, target_language, generate_srt_checkbox, audio_length, proportional_timings)
    else:
        return process_hebrew_model(audio_file, target_language, generate_srt_checkbox, audio_length, proportional_timings)


def process_general_model(general_segments, target_language, generate_srt_checkbox, audio_length, proportional_timings):
    transcribed_text = ' '.join([segment['text'] for segment in general_segments])

    if generate_srt_checkbox:
        srt_content = ""
        segment_id = 1
        max_line_length = 40

        for i, (start_time_seconds, end_time_seconds) in enumerate(proportional_timings):
            if i < len(general_segments):
                text = general_segments[i]['text']
            else:
                text = ""

            lines = split_lines(text, max_line_length=max_line_length)
            while lines:
                current_lines = lines[:2]
                lines = lines[2:]

                start_time = str(datetime.timedelta(seconds=start_time_seconds)).split(".")[0] + ',000'
                end_time = str(datetime.timedelta(seconds=end_time_seconds)).split(".")[0] + ',000'
                translated_lines = [translate_text(line, target_language) for line in current_lines]

                srt_entry = format_srt_entry(segment_id, start_time, end_time, translated_lines)
                srt_content += srt_entry
                segment_id += 1

        os.makedirs("output", exist_ok=True)
        srt_file_path = os.path.join("output", "output.srt")
        with open(srt_file_path, "w", encoding="utf-8") as srt_file:
            srt_file.write(srt_content)

        srt_html_content = ""
        for line in srt_content.split('\n'):
            srt_html_content += f"<div>{line}</div>"

        return f"Audio Length: {audio_length} seconds\n\n" + format_text(srt_html_content)
    else:
        paragraphs = split_text_into_paragraphs_by_sentence(transcribed_text)
        translated_paragraphs = [translate_text(paragraph, target_language) for paragraph in paragraphs]
        final_text = '\n\n'.join(translated_paragraphs)
        html_paragraphs = ''.join([f'<p>{paragraph}</p>' for paragraph in translated_paragraphs])
        os.makedirs("output", exist_ok=True)
        result_file_path = os.path.join("output", "result.txt")
        with open(result_file_path, "w", encoding="utf-8") as result_file:
            result_file.write(final_text)
        return f"Audio Length: {audio_length} seconds\n\n" + format_text(html_paragraphs)

def split_text_into_paragraphs_by_sentence(text, max_sentences_per_paragraph=5):
    # Split the text into sentences using a regular expression
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    paragraphs = []
    current_paragraph = []

    for sentence in sentences:
        current_paragraph.append(sentence)
        if len(current_paragraph) >= max_sentences_per_paragraph:
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []

    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))

    return paragraphs

def process_hebrew_model(audio_file, target_language, generate_srt_checkbox, audio_length, proportional_timings):
    # Transcribe with the Hebrew model
    hebrew_transcription_result = transcribe_with_hebrew_model(audio_file)
    hebrew_segments = hebrew_transcription_result['segments']
    transcribed_text = ''.join([segment['text'] for segment in hebrew_segments])  # Ensure we get all text

    if generate_srt_checkbox:
        # Split the Hebrew transcription into parts corresponding to the general model timings
        hebrew_text_parts = split_text_into_parts(transcribed_text, len(proportional_timings))
        srt_content = ""
        segment_id = 1
        max_line_length = 40

        for i, (start_time_seconds, end_time_seconds) in enumerate(proportional_timings):
            if i < len(hebrew_text_parts):
                text = hebrew_text_parts[i]
            else:
                text = ""

            lines = split_lines(text, max_line_length=max_line_length)
            while lines:
                current_lines = lines[:2]
                lines = lines[2:]

                start_time = str(datetime.timedelta(seconds=start_time_seconds)).split(".")[0] + ',000'
                end_time = str(datetime.timedelta(seconds=end_time_seconds)).split(".")[0] + ',000'
                translated_lines = [translate_text(line, target_language) for line in current_lines]

                srt_entry = format_srt_entry(segment_id, start_time, end_time, translated_lines)
                srt_content += srt_entry
                segment_id += 1

        os.makedirs("output", exist_ok=True)
        srt_file_path = os.path.join("output", "output.srt")
        with open(srt_file_path, "w", encoding="utf-8") as srt_file:
            srt_file.write(srt_content)

        srt_html_content = ""
        for line in srt_content.split('\n'):
            srt_html_content += f"<div>{line}</div>"

        return f"Audio Length: {audio_length} seconds\n\n" + format_text(srt_html_content)
    else:
        # In the non-SRT case, ensure the full text is used and split into paragraphs by sentence
        paragraphs = split_text_into_paragraphs_by_sentence(transcribed_text)
        translated_paragraphs = [translate_text(paragraph, target_language) for paragraph in paragraphs]
        final_text = '\n\n'.join(translated_paragraphs)
        html_paragraphs = ''.join([f'<p>{paragraph}</p>' for paragraph in translated_paragraphs])
        os.makedirs("output", exist_ok=True)
        result_file_path = os.path.join("output", "result.txt")
        with open(result_file_path, "w", encoding="utf-8") as result_file:
            result_file.write(final_text)
        return f"Audio Length: {audio_length} seconds\n\n" + format_text(html_paragraphs)

def split_text_into_parts(text, num_parts):
    words = text.split()
    avg_words_per_part = len(words) // num_parts
    parts = []
    for i in range(num_parts):
        part = ' '.join(words[i * avg_words_per_part:(i + 1) * avg_words_per_part])
        parts.append(part)
    if len(parts) < num_parts:
        parts.append(' '.join(words[num_parts * avg_words_per_part:]))
    return parts


def generate_srt(segments, target_language):
    srt_content = ""
    segment_id = 1
    max_line_length = 40

    for segment in segments:
        start_time_seconds = segment['start']
        end_time_seconds = segment['end']
        text = segment['text']

        lines = split_lines(text, max_line_length=max_line_length)
        while lines:
            current_lines = lines[:2]
            lines = lines[2:]

            start_time = str(datetime.timedelta(seconds=start_time_seconds)).split(".")[0] + ',000'
            end_time = str(datetime.timedelta(seconds=end_time_seconds)).split(".")[0] + ',000'
            translated_lines = [translate_text(line, target_language) for line in current_lines]

            srt_entry = format_srt_entry(segment_id, start_time, end_time, translated_lines)
            srt_content += srt_entry
            segment_id += 1

    os.makedirs("output", exist_ok=True)
    srt_file_path = os.path.join("output", "output.srt")
    with open(srt_file_path, "w", encoding="utf-8") as srt_file:
        srt_file.write(srt_content)

    srt_html_content = ""
    for line in srt_content.split('\n'):
        srt_html_content += f"<div>{line}</div>"

    return srt_html_content
        
title = "Unlimited Length Transcription and Translation"
description = "With: large-v2 or ivrit-ai/whisper-v2-d3-e3 and whisper-v2 large | GUI by Shmuel Ronen"

interface = gr.Interface(
    fn=transcribe_and_translate,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio File"),
        gr.Dropdown(choices=['Hebrew', 'English', 'Spanish', 'French', 'German', 'Portuguese', 'Arabic'], 
                    label="Target Language", value='Hebrew'),
        gr.Dropdown(choices=['General Model', 'Hebrew Model'], 
                    label="Model Choice", value='Hebrew Model'),
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
