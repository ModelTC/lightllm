import base64
import httpx
from PIL import Image
from io import BytesIO


def image2base64(img_str: str):
    image_obj = Image.open(img_str)
    if image_obj.format is None:
        raise ValueError("No image format found.")
    buffer = BytesIO()
    image_obj.save(buffer, format=image_obj.format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


async def fetch_image(url, timeout):
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=timeout)
        return response.content
