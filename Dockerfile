# Dockerfile

# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Copy the Hebrew Wispher Jupyter Notebook to the container
COPY hebrew_wispher.ipynb .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support using the official PyTorch website
RUN pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118

# Install the Hugging Face Transformers library
RUN pip install -U "huggingface_hub[cli]"

# Download the Hebrew Wispher model from the Hugging Face Model Hub
RUN huggingface-cli download ivrit-ai/whisper-large-v2-tuned

# Install ffmpeg packages
RUN apt update && apt install -y ffmpeg

# Create the temporary directory for hebrew_wispher
RUN mkdir /tmp/hebrew_wispher

# Copy the app.py file to the container
COPY app.py .

# Expose the port on which the Gradio interface will run
EXPOSE 7860

# Run the Gradio interface when the container starts
CMD ["python", "app.py"]