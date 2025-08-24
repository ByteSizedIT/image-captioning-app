'''web application that generates captions for images using the BLIP model and the Gradio library for an interactive web interface'''

import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


def caption_image(input_image: np.ndarray):
    # Convert numpy array to PIL Image and convert to RGB
    raw_image = Image.fromarray(input_image).convert('RGB')
    
    # Process the image
    text = "the image of"
    inputs = processor(images=raw_image, text=text, return_tensors="pt")
    
    # Generate a caption for the image
    outputs = model.generate(**inputs, max_length=50)

    # Decode the generated tokens to text and store it into `caption`
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption

iface = gr.Interface(
    fn=caption_image, 
    inputs=gr.Image(), 
    outputs="text",
    title="Image Captioning",
    description="This is a simple web app for generating captions for images using a trained model."
)
'''
gr.Image() is a function call that creates an Image input component in Gradio.
In Gradio, all the UI components (like buttons, text boxes, images, audio, etc.) are Python classes that you instantiate with parentheses.
So gr.Image() creates an object of the Image component class. That object tells Gradio:
“Hey, the input should be an image — let the user upload or draw one, and give it to my function as input.”
inputs often need component objects because they’re more complex.
outputs can use either a string shortcut (easy) or a component object (customizable).
"text" is really just shorthand.
This works for a bunch of common types:
"text" → gr.Textbox()
"image" → gr.Image()
"label" → gr.Label() (for classification outputs)
"audio" → gr.Audio()
'''

# Start the web app by calling the launch() method:
iface.launch()

'''TO RUN:
python3 2-image-cap-app.py
'''