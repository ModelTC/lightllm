import os
import re
import json
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Union
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from rpyc.utils.classic import obtain
from io import BytesIO
import rpyc
from lightllm.server.embed_cache.utils import tensor2bytes, read_shm, create_shm, get_shm_name_data, get_shm_name_embed
from lightllm.utils.log_utils import init_logger


class InternVisionModel:
    def __init__(self):
        pass

    def load_projector_update(self, config, weight_dir):
        projector_type = config.get("projector_type", "mlp2x_gelu")
        projector_weights = []
        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
        if mlp_gelu_match:
            self.mlp_depth = int(mlp_gelu_match.group(1))
            new_dict = {}
            for f in os.listdir(weight_dir):
                if f.endswith(".bin"):
                    d = torch.load(os.path.join(weight_dir, f), "cpu")
                    for k, v in d.items():
                        if "vision_proj" in k:
                            projector_weights.append(v.half())
                        elif "vit.vision_tower." in k:
                            new_dict[k[len("vit.vision_tower.") :]] = v.half()
            self.vision_tower.load_state_dict(new_dict, strict=True)
            return projector_weights
        if projector_type == "identity":
            return []
        raise ValueError(f"Unknown projector type: {projector_type}")

    def load_model(self, weight_dir):
        config_file = os.path.join(weight_dir, "config.json")
        config = json.load(open(config_file))
        self.select_layer = config.get("mm_vision_select_layer", -1)
        self.select_feature = config.get("mm_vision_select_feature", "patch")

        # load clip vision model by cfg['mm_vision_tower']:
        #   huggingface_name or path_of_clip_relative_to_llava_model_dir
        vision_path = config.get("mm_vision_tower", "openai/clip-vit-large-patch14-336")
        if isinstance(vision_path, list):
            vision_path = vision_path[0]
        if vision_path.startswith("./"):
            vision_path = os.path.join(weight_dir, vision_path)

        self.image_processor = transforms.Compose(
            [
                transforms.Resize((config["img_size"], config["img_size"]), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        from transformers import CLIPVisionModel

        self.vision_tower = CLIPVisionModel.from_pretrained(vision_path)
        self.vision_tower.requires_grad_(False)
        self.resize_pos(config, vision_path)
        self.projector_weights = self.load_projector_update(config, weight_dir)
        self.vision_tower = self.vision_tower.half()
        self.device = torch.device("cpu")

        # load projector weights
        assert len(self.projector_weights) == self.mlp_depth * 2

    def resize_pos(self, config, vision_path):
        mm_vision_tower = vision_path.split("/")[-1]
        vision_tower_match = re.match(r"^clip-vit-large-patch(\d+)-(\d+)$", mm_vision_tower)
        patch_size = int(vision_tower_match.group(1))
        clip_imge_size = int(vision_tower_match.group(2))

        orig_size = clip_imge_size // patch_size
        new_size = config["img_size"] // patch_size
        if orig_size == new_size:
            self.is_resize_pos = False
            return

        pos_embed_checkpoint = self.vision_tower.vision_model.embeddings.position_embedding.weight
        pos_embed_checkpoint = pos_embed_checkpoint.unsqueeze(0)

        if pos_embed_checkpoint.shape[1] == new_size ** 2 + 1:
            self.is_resize_pos = True
        else:
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_extra_tokens = 1
            new_num = new_size ** 2 + num_extra_tokens
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)

            new_pos_embed = new_pos_embed.squeeze(0)

            self.vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(new_num, 1024)
            self.vision_tower.vision_model.embeddings.position_embedding.weight = torch.nn.Parameter(
                new_pos_embed.to(pos_embed_checkpoint.dtype)
            )
            self.vision_tower.vision_model.embeddings.position_ids = torch.arange(new_num).expand((1, -1))
            self.is_resize_pos = True

    def cuda(self):
        self.vision_tower = self.vision_tower.cuda()
        for i in range(len(self.projector_weights)):
            self.projector_weights[i] = self.projector_weights[i].cuda()
        return self

    # batch images infer
    def forward(self, x):
        x = x.cuda().to(dtype=self.vision_tower.dtype)

        x = self.vision_tower(x, output_hidden_states=True)
        x = x.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            x = x[:, 1:].contiguous()

        if len(self.projector_weights) == 0:
            return x

        B, L, N = x.shape
        x = x.view(-1, N)
        # mm_project
        x = F.linear(
            x,
            weight=self.projector_weights[0],
            bias=self.projector_weights[1],
        )
        for i in range(1, self.mlp_depth):
            x = F.gelu(x)
            x = F.linear(
                x,
                weight=self.projector_weights[i * 2],
                bias=self.projector_weights[i * 2 + 1],
            )
        x = x.view(B, L, -1)
        return x

    def encode(self, image_uuids: List):
        img_tensors = []
        uuids = []
        valid_id = 0
        valid_ids = []

        for i, item in enumerate(image_uuids):
            item = obtain(item)
            if isinstance(item, int):
                uuids.append(item)
                image_data = read_shm(get_shm_name_data(item))
                image_data = Image.open(BytesIO(image_data)).convert("RGB")
                t = self.image_processor.preprocess(image_data, return_tensors="pt")["pixel_values"]
                img_tensors.append(t)
            else:
                raise Exception("Unsupport input types: {} for {}".format(type(item), item))

            cur_num = img_tensors[-1].shape[0]
            valid_ids.append([valid_id, valid_id + cur_num])
            valid_id += cur_num

        if len(img_tensors) <= 0:
            return None

        img = torch.cat(img_tensors, dim=0)
        all_img_embeds = self.forward(img)

        return all_img_embeds, uuids, valid_ids
