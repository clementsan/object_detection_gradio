---
title: Object Detection
emoji: 🖼
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 5.5.0
app_file: app.py
pinned: false
short_description: Object detection via Gradio
---

# Object detection

Aim: AI-driven object detection (on COCO image dataset)

Machine learning models:
 - facebook/detr-resnet-50, 
 - facebook/detr-resnet-101, 
 - hustvl/yolos-tiny, 
 - hustvl/yolos-small

### <b>Table of contents:</b>
 - [Execution via command line](#1-execution-via-command-line)
 - [Execution via User Interface ](#2-execution-via-user-interface)
 - [Execution via Gradio client API](#3-execution-via-gradio-client-api)
 - [Deployment on Hugging Face](#4-deployment-on-hugging-face)
 - [Deployment on Docker Hub](#5-deployment-on-docker-hub)


## 1. Execution via command line

### 1.1. Use of torch library
> python detect_torch.py 

### 1.2. Use of transformers library
> python detect_transformers.py

### 1.3. Use of HuggingFace pipeline library
> python detect_pipeline.py

## 2. Execution via User Interface 
Use of Gradio library for web interface

Command line:
> python app.py

<b>Note:</b> The Gradio app should now be accessible at http://localhost:7860

## 3. Execution via Gradio client API

<b>Note:</b> Use of existing Gradio server (running locally, in a Docker container, or in the cloud as a HuggingFace space or AWS)

### 3.1. Creation of docker container

Command lines:
> sudo docker build -t gradio-app .

> sudo docker run -p 7860:7860 gradio-app

The Gradio app should now be accessible at http://localhost:7860

### 3.2. Direct inference via API
Command line:
> python inference_API.py


## 4. Deployment on Hugging Face

This web application is available on Hugging Face, via a Gradio space

URL: https://huggingface.co/spaces/cvachet/object_detection_gradio


## 5. Deployment on Docker Hub

This web application is available as a container on Docker Hub

URL: https://hub.docker.com/r/cvachet/object-detection-gradio

