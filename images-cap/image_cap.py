import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
import os


processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


img_path = r"images/test2.jpg"
print(f"Current working directory: {os.getcwd()}")
print(f"Trying to load image from: {img_path}")
print(f"Full path: {os.path.abspath(img_path)}")
print(f"File exists: {os.path.exists(img_path)}")

image = Image.open(img_path).convert('RGB')
print(f"Image loaded successfully. Size: {image.size}")



text = "i see a photo of"
inputs = processor(images=image, text=text, return_tensors="pt")

print("Generating caption...")
outputs = model.generate(**inputs, max_length=50)

caption = processor.decode(outputs[0], skip_special_tokens=True)

print(f"Generated caption: {caption}")
print(f"Image file: {img_path}")

