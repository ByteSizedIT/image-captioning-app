'''simple image captioning AI printed in the terminal'''

import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load image and convert to RGB format 
img_path = "tobias-reich-C28wWvi0xB8-unsplash.jpg"
image = Image.open(img_path).convert('RGB')

# Pass pre-processed image through the processor to generate inputs in the required format; a question is not needed for image captioning
text = "the image of"
inputs = processor(images=image, text=text, return_tensors="pt")
'''return_tensors arg set to "pt" to return PyTorch tensors
'''

# Pass the inputs into model's generate method to enerate a caption for the image
outputs = model.generate(**inputs, max_length=50)
'''
- The argument max_length=50 specifies that the model should generate a caption of up to 50 tokens in length.
- the two asterisks (**) in Python are used in function calls to unpack dictionaries and pass items in the dictionary as keyword arguments to the function. **inputs is unpacking the inputs dictionary and passing its items as arguments to the model'''

# Decode the generated tokens to text
caption = processor.decode(outputs[0], skip_special_tokens=True)
'''the generated output is a sequence of tokens. To transform these tokens into human-readable text, you use the decode method provided by the processor. The skip_special_tokens argument is set to True to ignore special tokens in the output text'''

# Print the caption
print(caption)


'''TO RUN:
python3 1-image-cap.py
'''




'''"AutoProcessor" and "BlipForConditionalGeneration" are components of the BLIP model, which is a vision-language model available in the Hugging Face Transformers library.

AutoProcessor : This is a processor class that is used for preprocessing data for the BLIP model. It wraps a BLIP image processor and an OPT/T5 tokenizer into a single processor. This means it can handle both image and text data, preparing it for input into the BLIP model.

BlipForConditionalGeneration : This is a model class that is used for conditional text generation given an image and an optional text prompt. In other words, it can generate text based on an input image and an optional piece of text. This makes it useful for tasks like image captioning or visual question answering, where the model needs to generate text that describes an image or answer a question about an image.'''


