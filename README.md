# Object detection

Aim: AI-driven object detection (on COCO image dataset)

## Direct object detection via python scripts

### 1. Use of torch library
> python detect_torch.py 

### 2. Use of transformers library
> python detect_transformers.py

### 3. Use of HuggingFace pipeline library
> python detect_pipeline.py

## Object detection via User Interface 
Use of Gradio library for web interface

Command line:
> python app.py

<b>Note:</b> The Gradio app should now be accessible at http://localhost:7860

## Object detection via Gradio client API

<b>Note:</b> Use of existing Gradio server (running locally, in a Docker container, or in the cloud as a HuggingFace space or AWS)

### 1. Creation of docker container

Command lines:
> sudo docker build -t gradio-app .

> sudo docker run -p 7860:7860 gradio-app

The Gradio app should now be accessible at http://localhost:7860

### 2. Direct inference via API
Command line:
> python inference_API.py
