from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

_PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
_MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
_MODEL.eval()

def describe_image(image: Image.Image) -> str:
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = _PROCESSOR(image, return_tensors="pt")

    with torch.no_grad():
        out = _MODEL.generate(**inputs, max_new_tokens=50)

    caption = _PROCESSOR.decode(out[0], skip_special_tokens=True)
    return caption.strip()












                                             



