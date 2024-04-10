import os
import json
import torch
from lightllm.models.vit.layer_infer.pre_layer_infer import ViTPreLayerInfer
from lightllm.models.vit.layer_infer.post_layer_infer import ViTPostLayerInfer
from lightllm.models.vit.layer_infer.transformer_layer_infer import ViTTransformerLayerInfer
from lightllm.models.vit.layer_weights.pre_and_post_layer_weight import ViTPreAndPostLayerWeight
from lightllm.models.vit.layer_weights.transformer_layer_weight import ViTTransformerLayerWeight
from lightllm.common.basemodel.layer_weights.hf_load_utils import load_hf_weights
# from lightllm.models.vit.layer_weights.hf_load_utils import load_hf_weights
from lightllm.utils.log_utils import init_logger
# from lightllm.models.husky.husky_visual import load_image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from typing import List, Union
from io import BytesIO

logger = init_logger(__name__)

class VisionTransformer:

    # weight class
    pre_and_post_weight_class = ViTPreAndPostLayerWeight
    transformer_weight_class = ViTTransformerLayerWeight

    # infer class
    pre_layer_infer_class = ViTPreLayerInfer
    transformer_layer_infer_class = ViTTransformerLayerInfer
    post_layer_infer_class = ViTPostLayerInfer

    def __init__(self, kvargs):
        self.tp_rank_ = kvargs["tp_rank"]
        self.world_size_ = kvargs["world_size"]
        self.weight_dir_ = kvargs["weight_dir"]
        self.load_way = kvargs.get("load_way", "HF")
        self.mode = [m.replace('int4weight', 'w4a16').replace('int8weight', 'w8a16') for m in kvargs.get("mode", [])]
        self.weight_dict = kvargs.get("weight_dict", None)
        self._init_config()
        self._padding_hidden_size()
        self._init_weights()
        self._init_infer_layer()
        return

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            self.config = json.load(json_file)
            if self.config["llm_config"]["model_type"] == "llama":
                self.ureader = False
            else:
                self.ureader = True
            self.select_layer = self.config["select_layer"]
            self.config["vision_config"]["llm_hidden_size"] = self.config["llm_config"]["hidden_size"]
            self.config["vision_config"]["downsample_ratio"] = self.config["downsample_ratio"]
            self.config = self.config["vision_config"]
        self.layers_num = self.config["num_hidden_layers"]
        return

    def _padding_hidden_size(self):
        self.config["padding_hidden_size"] = 0
        self.config["padding_head_num"] = self.config["num_attention_heads"]

        head_dim = self.config["hidden_size"] // self.config["num_attention_heads"]
        if self.config["num_attention_heads"] % self.world_size_ != 0:
            padding_head_num = (self.config["num_attention_heads"] + self.world_size_ - 1) // self.world_size_ * self.world_size_ - self.config["num_attention_heads"]
            self.config["padding_hidden_size"] = padding_head_num * head_dim
            self.config["padding_head_num"] += padding_head_num
        return

    def _init_weights(self):
        self.pre_post_weight = self.pre_and_post_weight_class(
            self.tp_rank_, self.world_size_, torch.float16, network_config=self.config, mode=self.mode
        )
        self.trans_layers_weight = [
            self.transformer_weight_class(
                i, self.tp_rank_, self.world_size_, torch.float16, network_config=self.config, mode=self.mode
            )
            for i in range(self.config["num_hidden_layers"])
        ]
        load_hf_weights(
            "fp16",
            weight_dir=self.weight_dir_,
            pre_post_layer=self.pre_post_weight,
            transformer_layer_list=self.trans_layers_weight,
            weight_dict=self.weight_dict,
        )
        self.pre_post_weight.verify_load()
        [weight.verify_load() for weight in self.trans_layers_weight]
        return

    def _init_infer_layer(self):
        self.pre_infer = self.pre_layer_infer_class(
            tp_rank=self.tp_rank_, world_size=self.world_size_, network_config=self.config, mode=self.mode
        )
        self.post_infer = self.post_layer_infer_class(
            tp_rank=self.tp_rank_, world_size=self.world_size_, network_config=self.config, mode=self.mode
        )
        self.layers_infer = [
            self.transformer_layer_infer_class(
                i, tp_rank=self.tp_rank_, world_size=self.world_size_, network_config=self.config, mode=self.mode
            )
            for i in range(self.config["num_hidden_layers"])
        ]
        return

    @torch.no_grad()
    def forward(
        self,
        pixel_values
    ):
        input_embs = self.pre_infer.forward(pixel_values, self.pre_post_weight)
        for i in range(self.layers_num + self.select_layer + 1):
            input_embs = self.layers_infer[i].forward(input_embs, self.trans_layers_weight[i])
        input_embs = self.post_infer.forward(input_embs[:, 1:, :], self.pre_post_weight)
        return input_embs
  

    @torch.no_grad()
    def encode(self, image_items: List[Union[str, torch.Tensor, Image.Image]]):
        img_tensors = []
        valid_ids = []
        valid_id = 0

        # load images to batch tensor
        for i, url in enumerate(image_items):
            if isinstance(url, str):
                # [3, 3, 448, 448] or [1, 3, 448, 448]
                t = load_image(url, ureader=self.ureader)
                img_tensors.append(t)
            elif isinstance(url, torch.Tensor):
                img_tensors.append(url)
            elif isinstance(url, Image.Image):
                # [3, 3, 448, 448] or [1, 3, 448, 448]
                t = load_image(None, image=url.convert('RGB'), ureader=self.ureader)
                img_tensors.append(t)
            else:
                raise Exception("Unsupport input types: {} for {}".format(type(url), url))

            cur_num = img_tensors[-1].shape[0]
            valid_ids.append([valid_id, valid_id + cur_num])
            valid_id += cur_num

        if len(img_tensors) <= 0:
            return None

        # (b, 3, 224, 224)
        imgs = torch.cat(img_tensors, dim=0)
        #pixel_values = imgs.to(self.device, dtype=self.dtype)
        pixel_values = imgs.cuda().half() #.to(self.device, dtype=self.dtype)
        all_img_embeds = self.forward(pixel_values)
        return [all_img_embeds[start:end] for start, end in valid_ids]