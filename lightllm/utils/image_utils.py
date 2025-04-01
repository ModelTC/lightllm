import base64
from io import BytesIO
from PIL import Image

from lightllm.server.multimodal_params import MultimodalParams, ImageItem


def image2base64(img_str: str):
    image_obj = Image.open(img_str)
    if image_obj.format is None:
        raise ValueError("No image format found.")
    buffer = BytesIO()
    image_obj.save(buffer, format=image_obj.format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
