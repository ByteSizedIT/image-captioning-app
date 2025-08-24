'''automated image captioning program that works directly from a URL. The user provides the URL, and the code generates captions for the images found on the webpage. The output is a text file that includes all the image URLs along with their respective captions (like the image below). Using BeautifulSoup for parsing the HTML content of the page and extracting the image URLs.'''

'''
1.send a HTTP request to the provided URL and retrieve the webpage's content. 
2. content is parsed by BeautifulSoup, which creates a parse tree from page's HTML.
3. look for 'img' tags in parse tree - they contain the links to the images hosted on the webpage.'''

import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base") 
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# URL of the page to scrape
url = "https://en.wikipedia.org/wiki/IBM"

# Download the page
response = requests.get(url)

# Parse the page with BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Find all img elements
img_elements = soup.find_all('img')

# Open a file to write the captions
with open("captions.txt", "w") as caption_file:

    # Iterate over each img element
    for img_element in img_elements:

        # Skip if the image is an SVG or too small (likely an icon)
        img_url = img_element.get('src')
        '''get() is an object method of Tag objects in BeautifulSoup.
        🔑 It’s analogous to dict.get(key) in Python dictionaries.
        For example:
        img_element['src']        # Raises KeyError if 'src' missing
        img_element.get('src')    # Returns None if 'src' missing'''
        
        if "svg" in img_url or "1x1" in img_url:
            continue
        '''GPT: A 1x1 image is not usually an icon. Instead, it’s most often:
        Tracking pixels → Invisible images embedded in emails or websites to log when something was opened or viewed.
        Transparent placeholders → Used for spacing, alignment, or lazy-loading actual content.
        Analytics / Ad beacons → To record page impressions, user activity, etc.
        Icons, on the other hand, are typically larger (16×16, 32×32, 64×64, etc.) and are meant to be visible to the user. A 1×1 is basically too small to be useful as an icon — it’s effectively invisible to the naked eye.'''
        '''img_url will be a plain Python string, not a BeautifulSoup object.
        img_element is a BeautifulSoup Tag object.
        .get("src") fetches the value of the src attribute from its internal dictionary of attributes.
        Attribute values in BeautifulSoup are stored as native Python types:
        Strings → 'cat.jpg'
        Lists (e.g. multiple classes) → ['note', 'highlight']
        None if the attribute doesn’t exist.'''

    # Correct the URL if it's malformed
        if img_url.startswith('//'):
            img_url = 'https:' + img_url
        # Skip URLs that don't start with http:// or https://
        elif not img_url.startswith('http://') and not img_url.startswith('https://'):
            continue  # Skip URLs that don't start with http:// or https://

        try:
            # Download the image
            response = requests.get(img_url)
            # Convert the image data to a PIL Image
            raw_image = Image.open(BytesIO(response.content))
            if raw_image.size[0] * raw_image.size[1] < 400:
                continue
            raw_image = raw_image.convert('RGB')
            '''from io import BytesIO is used to work with binary streams in memory, instead of saving or reading from files on disk - creates an in-memory stream that behaves like a file.
            When you have an image stored as raw bytes (for example, downloaded from the web or received via an API), you can wrap those bytes in a BytesIO object, and PIL.Image.open() can read it as if it were a file.'''

            # Process the image
            inputs = processor(raw_image, return_tensors="pt")
            # Generate a caption for the image
            out = model.generate(**inputs, max_new_tokens=50)
            # Decode the generated tokens to text
            caption = processor.decode(out[0], skip_special_tokens=True)
            # Write the caption to the file, prepended by the image URL
            caption_file.write(f"{img_url}: {caption}\n")
            
        except Exception as e:
            print(f"Error processing image {img_url}: {e}")
            continue

'''TO RUN:
python3 3-image_cap_tool.py
'''