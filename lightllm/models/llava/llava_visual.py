import torch
import torch.nn.functional as F
import json
import os
from PIL import Image
from typing import List, Union
from safetensors import safe_open
from rpyc.utils.classic import obtain
from io import BytesIO
import rpyc
from lightllm.server.embed_cache.utils import tensor2bytes, read_shm, create_shm, get_shm_name_data, get_shm_name_embed
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


class LlavaVisionModel:
    def __init__(self, kvargs):
        self.tp_rank_ = kvargs["tp_rank"]
        self.world_size_ = kvargs["vit_world_size"]
        self.client_port = kvargs["client_port"]
        self.cache_client = rpyc.connect("localhost", self.client_port)
        self.visual_gpu = kvargs["visual_gpu"]
        self.device = torch.device(f"cuda:{self.visual_gpu}")
        pass

    def load_model(self, weight_dir):
        config_file = os.path.join(weight_dir, "config.json")
        config = json.load(open(config_file))

        # for llava-v1.5-7b-hf model, should load config from transformers
        if "text_config" in config:
            self.load_hf_model(config, weight_dir)
        else:
            self.load_bin_model(config, weight_dir)

        self.vision_tower.requires_grad_(False)
        self.device = torch.device("cpu")

        assert "model.mm_projector.0.weight" in self.projector_weights
        assert "model.mm_projector.0.bias" in self.projector_weights
        assert "model.mm_projector.2.weight" in self.projector_weights
        assert "model.mm_projector.2.bias" in self.projector_weights

    def load_hf_model(self, config, weight_dir):
        from transformers import AutoConfig, AutoProcessor, LlavaForConditionalGeneration

        config = AutoConfig.from_pretrained(weight_dir, trust_remote_code=True)
        self.select_layer = config.vision_feature_layer
        self.select_feature = config.vision_feature_select_strategy
        processor = AutoProcessor.from_pretrained(weight_dir)
        self.image_processor = processor.image_processor
        model = LlavaForConditionalGeneration.from_pretrained(
            weight_dir,
            torch_dtype=torch.float16,
        )
        self.vision_tower = model.vision_tower
        model.multi_modal_projector = None
        model.language_model = None

        # load projector weights
        self.projector_weights = {}
        for f in os.listdir(weight_dir):
            if f.endswith(".safetensors"):
                d = safe_open(os.path.join(weight_dir, f), "pt", "cpu")
                for k in d.keys():
                    if "multi_modal_projector.linear_1" in k:
                        self.projector_weights[
                            k.replace("multi_modal_projector.linear_1", "model.mm_projector.0")
                        ] = d.get_tensor(k).half()
                    if "multi_modal_projector.linear_2" in k:
                        self.projector_weights[
                            k.replace("multi_modal_projector.linear_2", "model.mm_projector.2")
                        ] = d.get_tensor(k).half()

    def load_bin_model(self, config, weight_dir):
        self.select_layer = config.get("mm_vision_select_layer", -2)
        self.select_feature = config.get("mm_vision_select_feature", "patch")

        # load clip vision model by cfg['mm_vision_tower']:
        #   huggingface_name or path_of_clip_relative_to_llava_model_dir
        vision_path = config.get("mm_vision_tower", "openai/clip-vit-large-patch14-336")
        if isinstance(vision_path, list):
            vision_path = vision_path[0]
        if vision_path.startswith("./"):
            vision_path = os.path.join(weight_dir, vision_path)

        from transformers import CLIPVisionModel, CLIPImageProcessor

        self.image_processor = CLIPImageProcessor.from_pretrained(vision_path)
        self.vision_tower = CLIPVisionModel.from_pretrained(vision_path).half()

        # load projector weights
        self.projector_weights = {}
        for f in os.listdir(weight_dir):
            if f.endswith(".bin"):
                d = torch.load(os.path.join(weight_dir, f), "cpu")
                for k, v in d.items():
                    if "model.mm_projector" in k:
                        self.projector_weights[k] = v.half()

    def cuda(self):
        self.vision_tower = self.vision_tower.cuda()
        for k, v in self.projector_weights.items():
            self.projector_weights[k] = v.cuda()
        self.device = torch.device(self.device)
        torch.cuda.set_device(self.device)
        return self

    # batch images infer
    def forward(self, x):
        x = x.half().to(device=self.device)

        x = self.vision_tower(x, output_hidden_states=True)
        x = x.hidden_states[self.select_layer]
        if self.select_feature == "patch" or self.select_feature == "default":
            x = x[:, 1:].contiguous()
        B, L, N = x.shape
        x = x.view(-1, N).half()

        # mm_project
        x = F.linear(
            x,
            weight=self.projector_weights["model.mm_projector.0.weight"],
            bias=self.projector_weights["model.mm_projector.0.bias"],
        )
        x = F.gelu(x)
        x = F.linear(
            x,
            weight=self.projector_weights["model.mm_projector.2.weight"],
            bias=self.projector_weights["model.mm_projector.2.bias"],
        )
        x = x.view(B, L, -1)
        return x

    def encode(self, image_items: List[Union[int, str, torch.Tensor, Image.Image]]):
        img_tensors = []
        uuids = []
        valid_id = 0
        valid_ids = []
        for i, item in enumerate(image_items):
            if self.world_size_ != 1:
                item = obtain(item)
            if isinstance(item, Image.Image):
                image = item.convert("RGB")
                t = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
                img_tensors.append(t)
            elif isinstance(item, torch.Tensor):
                img_tensors.append(item)
            elif isinstance(item, int):
                uuids.append(item)
                image_data = read_shm(get_shm_name_data(item))
                image_data = Image.open(BytesIO(image_data)).convert("RGB")
                t = self.image_processor.preprocess(image_data, return_tensors="pt")["pixel_values"]
                img_tensors.append(t)
            elif item.startswith("http://") or item.startswith("https://"):
                import requests

                image = Image.open(requests.get(item, stream=True).raw)
            else:
                raise Exception("Unsupport input types: {} for {}".format(type(item), item))

            cur_num = img_tensors[-1].shape[0]

            valid_ids.append([valid_id, valid_id + cur_num])
            valid_id += cur_num

        if len(img_tensors) <= 0:
            return None

        img = torch.cat(img_tensors, dim=0)
        pixel_values = img.to(self.device)
        all_img_embeds = self.forward(pixel_values)

        if len(uuids) == 0:
            return [all_img_embeds[start:end] for start, end in valid_ids]
        else:
            for i in range(len(uuids)):
                uid = uuids[i]
                if not self.cache_client.root.get_item_embed(uid):
                    start, end = valid_ids[i]
                    cur_embed_bytes = tensor2bytes(all_img_embeds[start:end])
                    create_shm(get_shm_name_embed(uuids[i]), cur_embed_bytes)
                    self.cache_client.root.set_item_embed(uuids[i])

        return
