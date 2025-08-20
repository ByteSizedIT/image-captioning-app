# AI Image Captioning App

## Overview

AI Image Captioning App using BLIP (Bootstrapping Language-Image Pretraining) for AI captioning, and Gradio for a web interface.

BLIP is a multimodal transformer model designed for joint vision–language understanding and generation.

Gradio is an open-source Python package that allows you to quickly build a demo or web application for your machine learning model, API, or any arbitrary Python function. You can then share a link to your demo or web application using Gradio's built-in sharing features. No JavaScript, CSS, or web hosting experience is needed.

## Set up instructions

- python -m venv venv
- source venv/bin/activate # Linux/Mac
- venv\Scripts\activate # Windows
- pip install -r requirements.txt

## Running the Project

1. Run in terminal (BLIP model only)
   To generate captions directly in the terminal using BLIP, run:
   python3 image_cap.py
2. Run with Gradio (web app)
   To launch the Gradio web app for image captioning, run:
   python3 image_cap_app.py
   After launching, open the link shown in your terminal (by default, http://localhost:7860/) in your browser.
