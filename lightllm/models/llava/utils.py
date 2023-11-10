import requests
from PIL import Image
from io import BytesIO
import os


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        timeout = int(os.getenv("REQUEST_TIMEOUT", "3"))
        response = requests.get(image_file, timeout=timeout)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


# split prompt containing <image> tag, and merge splited parts by special token id
def tokenizer_image_token(prompt, tokenizer, image_token, image_token_id):
    input_ids, ids = [tokenizer(x).input_ids for x in prompt.split(image_token, maxsplit=1)]
    if len(ids) > 0 and ids[0] == tokenizer.bos_token_id:
        ids = ids[1:]
    input_ids.append(image_token_id)
    input_ids.extend(ids)
    return input_ids
