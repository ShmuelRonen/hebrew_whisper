<h3>GUI for Unlimited Transcription and Translation with Whisper Hebrew</h3>

A powerful transcription and translation tool leveraging the ivrit-ai/whisper-v2-d3-e3 model and Whisper-v2 model for high-quality, unlimited-length audio processing with enhanced paragraph splitting and temporary file management for a clean workspace.

![V2](https://github.com/ShmuelRonen/hebrew_whisper/assets/80190186/1ef67524-a063-472d-a752-0a61a495d824)


## Features

- **Unlimited Length Transcription**: Transcribe audio files of any length without limitations.
- **Support for Multiple Languages**: Choose from Hebrew, English, Spanish, French, German, Portuguese, and Arabic for translation.
- **General and Hebrew Models**: Use either the general model (`large-v2`) for accurate timestamp generation or the specialized Hebrew model (`ivrit-ai/whisper-v2-d3-e3`) for improved Hebrew transcription.
- **Proportional Timings**: Accurate proportional timings based on the general model to ensure alignment with the actual audio length.
- **SRT File Generation**: Option to generate SRT files with synchronized subtitles.
- **HTML Paragraph Formatting**: Proper formatting of transcriptions and translations into paragraphs for better readability.
- **GPU Support**: Leverage GPU for faster transcription and translation if available.

## Usage

### Transcription and Translation

1. **Upload Audio File**: Click on the "Upload Audio File" button to select your audio file.
2. **Select Target Language**: Choose the desired language for translation from the dropdown menu. The default is set to Hebrew.
3. **Select Model Choice**: Choose between the "General Model" and the "Hebrew Model". The default is set to the "Hebrew Model".
4. **Generate SRT File**: Check the box if you want to generate an SRT file with subtitles.
5. **Submit**: Click the "Submit" button to start the transcription and translation process.

### Example

1. Upload an audio file (e.g., `example_audio.wav`).
2. Select "English" as the target language.
3. Select "General Model" to ensure accurate timing.
4. Optionally, check the "Generate Hebrew SRT File" box.
5. Click "Submit" to process the audio.

The output will display the transcription and translation in the chosen language, formatted into paragraphs for easy readability. If the SRT option is selected, an SRT file will also be generated and available for download.

## Installation steps

It's recommended to install in a virtual environment for Python projects to manage dependencies efficiently.

Clone the repository

```
git clone https://github.com/ShmuelRonen/hebrew_whisper.git
cd hebrew_whisper
```

#### NEW - One click installer and executor:

```
Double click on:

init_env.bat
```

### Manual installation:

It's recommended to create and activate a virtual environment here:
```

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

For PyTorch with CUDA 11.8 support, use the following command
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118ilE.md
```


After the installation, you can run the app by navigating to the directory containing `app.py` and executing:
```

python app.py
```

_____________

<div align="center">

<h2>Audio Transcription and Translation <br/> <span style="font-size:12px">Powered by Hebrew whisper-v2-d3-e3 and Large-v2 whisper models</span> </h2>

<div>
    <a href='https://huggingface.co/ivrit-ai/whisper-v2-d3-e3' target='_blank'>Hebrew new Whisper Model</a>&emsp;
</div>
<br>

## Acknowledgement
Special thanks to Kinneret Wm, Yam Peleg, Yair Lifshitz, Yanir Marmor from ivrit-ai for providing the new impruve Hebrew Whisper model,
making high-quality transcription and translation accessible to developers.

Special thanks to the creators of the Whisper Large-v2 model for their contribution to the development of high-quality transcription and translation technologies.


## Disclaimer
This project is intended for educational and development purposes. It leverages publicly available models and APIs. Please ensure to comply with the terms of use of the underlying models and frameworks.
