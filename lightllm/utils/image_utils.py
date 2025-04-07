import time
import base64
import httpx
import logging
from PIL import Image
from io import BytesIO
from fastapi import Request
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def image2base64(img_str: str):
    image_obj = Image.open(img_str)
    if image_obj.format is None:
        raise ValueError("No image format found.")
    buffer = BytesIO()
    image_obj.save(buffer, format=image_obj.format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


async def fetch_image(url, request: Request, timeout):
    logger.info(f"Begin to download image from url: {url}")
    start_time = time.time()
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url, timeout=timeout) as response:
            response.raise_for_status()
            ans_bytes = []
            async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                if await request.is_disconnected():
                    await response.aclose()
                    raise Exception("Request disconnected. User cancelled download.")
                ans_bytes.append(chunk)
                # 接收的数据不能大于128M
                if len(ans_bytes) > 128:
                    raise Exception("Image data is too big")

            content = b"".join(ans_bytes)
    end_time = time.time()
    logger.info("Download image time: {:.2f} seconds".format(end_time - start_time))
    return content
