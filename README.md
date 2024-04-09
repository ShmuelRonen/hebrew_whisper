<h3>GUI for Unlimited Transcription and Translation with Whisper Hebrew</h3>

A powerful transcription and translation tool leveraging the ivrit-ai/whisper-large-v2-tuned model for high-quality, unlimited-length audio processing with enhanced paragraph splitting and temporary file management for a clean workspace.

![if1](https://github.com/ShmuelRonen/hebrew_whisper/assets/80190186/c21ae1bc-9925-4d96-83cf-b9e2aa6993ec)

#### NEW: Creating an SRT file in Hebrew from any language.

#### NEW: Added the ability to translate the sound file text to the other languages in the menu.
please update the installation:
```
git pull
```

#### NEW: Docker support brunch By dawnburst:
```
https://github.com/dawnburst/hebrew_whisper/tree/support-docker
```


#### NEW: Google colab hebrew_wispher.ipynb added


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


This will start a Gradio interface locally, which you can access through the provided URL in your command line interface.

## How to Use
Once the application is running, follow these steps:
1. Upload your audio file through the Gradio interface.
2. Select the source language of your audio file.
3. Click submit to start the transcription and translation process.
4. The transcribed and translated text will be displayed in the textbox, and a text file containing the output will be saved in the specified output directory.

## Features
- Supports unlimited length audio files.
- Splits transcribed text into well-structured paragraphs.
- Deletes temporary files automatically, leaving a clean workspace.
- Uses CUDA for accelerated processing if available.

_____________

<div align="center">

<h2>Audio Transcription and Translation <br/> <span style="font-size:12px">Powered by ivrit-ai/whisper-large-v2-tuned</span> </h2>

<div>
    <a href='https://huggingface.co/ivrit-ai/whisper-large-v2-tuned' target='_blank'>Hebrew Whisper Model</a>&emsp;
</div>
<br>

## Acknowledgement
Special thanks to Kinneret Wm, Yam Peleg, Yair Lifshitz, Yanir Marmor from ivrit-ai for providing the Hebrew Whisper model,
making high-quality transcription and translation accessible to developers.

## Disclaimer
This project is intended for educational and development purposes. It leverages publicly available models and APIs. Please ensure to comply with the terms of use of the underlying models and frameworks.
