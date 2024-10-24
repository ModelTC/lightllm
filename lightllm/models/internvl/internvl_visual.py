import os
import re
import json
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Union
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import requests
from lightllm.server.embed_cache.utils import tensor2bytes, read_shm, create_shm, get_shm_name_data, get_shm_name_embed
import rpyc
from io import BytesIO
from lightllm.models.internvl.img_process import load_image
from rpyc.utils.classic import obtain
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class InternVLVisionModel:
    def __init__(self, kvargs):
        self.tp_rank_id = kvargs["tp_rank_id"]
        self.vit_tp = kvargs["vit_tp"]
        self.visual_gpu = kvargs["visual_gpu"]
        self.device = torch.device(f"cuda:{self.visual_gpu}")
        print(f"self.device is {self.device}")
        pass

    def load_model(self, weight_dir):
        assert torch.cuda.is_available()
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.config = json.load(open(os.path.join(weight_dir, "config.json")))
        self.model = AutoModel.from_pretrained(
            weight_dir,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            language_model="fake_language_model",
        )
        self.model.eval().cuda(self.device)

    def cuda(self, device):
        return self

    def encode(self, image_items: List[Union[int, str, torch.Tensor, Image.Image]]):
        img_tensors = []
        valid_ids = []
        valid_id = 0
        uuids = []
        # load images to batch tensor

        for i, url in enumerate(image_items):
            if self.vit_tp != 1:
                url = obtain(url)
            if isinstance(url, Image.Image):
                t = load_image(url, max_num=6)
                img_tensors.append(t)
            elif isinstance(url, torch.Tensor):
                img_tensors.append(url)
            elif isinstance(url, int):
                uuids.append(url)
                image_data = read_shm(get_shm_name_data(url))
                image_data = Image.open(BytesIO(image_data))
                t = load_image(image_data)
                img_tensors.append(t)
            else:
                raise Exception("Unsupport input types: {} for {}".format(type(url), url))

            cur_num = img_tensors[-1].shape[0]

            valid_ids.append([valid_id, valid_id + cur_num])
            valid_id += cur_num

        if len(img_tensors) <= 0:
            return None
        # (b, 3, 224, 224)
        torch.cuda.set_device(self.device)
        imgs = torch.cat(img_tensors, dim=0)
        pixel_values = imgs.to(device=self.device, dtype=self.dtype)
        all_img_embeds = self.model.extract_feature(pixel_values)

        return all_img_embeds, uuids, valid_ids
