import os
import re
import json
import numpy as np
from lightllm.common.mem_utils import select_mem_manager_class
from lightllm.models.gemma3.layer_infer.transformer_layer_infer import Gemma3TransformerLayerInfer
from lightllm.models.gemma3.layer_weights.pre_and_post_layer_weight import Gemma3PreAndPostLayerWeight
from lightllm.models.gemma3.layer_weights.transformer_layer_weight import Gemma3TransformerLayerWeight
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.models.llava.layer_weights.pre_and_post_layer_weight import LlavaPreAndPostLayerWeight
from lightllm.server.multimodal_params import MultimodalParams, ImageItem
from lightllm.server.core.objs import SamplingParams
from lightllm.common.build_utils import repair_config
from transformers import AutoConfig


# Warp of the origal tokenizer
class Gemma3Tokenizer:
    def __init__(self, tokenizer, model_cfg):
        self.tokenizer = tokenizer
        self.image_token = model_cfg.get("image_token", "<start_of_image>")
        self.boi_token = "<start_of_image>"
        self.eoi_token = "<end_of_image>"
        self.boi_token_index: int = model_cfg.get("boi_token_index", 255_999)
        self.eoi_token_index: int = model_cfg.get("eoi_token_index", 256_000)
        self.image_token_index: int = model_cfg.get("image_token_index", 262_144)
        self.mm_tokens_per_image: int = model_cfg.get("mm_tokens_per_image", 256)
        
        self.image_length = self.mm_tokens_per_image

    def init_imageItem_extral_params(
        self, img: ImageItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        return

    def get_image_token_length(self, img: ImageItem):
        return self.image_length

    # only change the impl of the encode func:
    def encode(self, prompt, multimodal_params: MultimodalParams = None):
        # split prompt by <image>, and merge parts by [pad_id] * 576
        ids_chunks = [self.tokenizer(x).input_ids for x in prompt.split(self.boi_token)]
        input_ids = ids_chunks[0]
        image_id = 0

        for ids in ids_chunks[1:]:
            token_id = multimodal_params.images[image_id].token_id
            token_num = multimodal_params.images[image_id].token_num
            assert token_num == self.image_length, "invalid token num: {} vs {}!".format(token_num, self.image_length)

            input_ids.append(self.boi_token_index)
            input_ids.extend(range(token_id, token_id + self.mm_tokens_per_image))
            input_ids.append(self.eoi_token_index)
            input_ids.extend(ids)
            image_id += 1
        if multimodal_params:
            image_cnt = len(multimodal_params.images)
            assert image_cnt == image_id, "invalid image tag num: {} vs {}!".format(image_cnt, image_id)
        return input_ids

    def __getattr__(self, name):
        if name != "encode":
            return getattr(self.tokenizer, name)
        return self.encode


class Gemma3TpPartModel(LlamaTpPartModel):
    # weight class
    pre_and_post_weight_class = Gemma3PreAndPostLayerWeight
    transformer_weight_class = Gemma3TransformerLayerWeight

    # infer class
    pre_layer_infer_class = LlamaMultimodalPreLayerInfer
    transformer_layer_infer_class = Gemma3TransformerLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_mem_manager(self):
        self.mem_manager = select_mem_manager_class(self.mode)(
            self.max_total_token_num,
            dtype=self.data_type,
            head_num=self.config["num_key_value_heads"] // self.tp_world_size_,
            head_dim=256,
            layer_num=self.config["num_hidden_layers"],
            mem_fraction=self.mem_fraction,
        )
        return

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            self.config = json.load(json_file)
        # for llava-v1.5-7b-hf model, should load config from transformers
        if "text_config" in self.config:
            config = AutoConfig.from_pretrained(self.weight_dir_, trust_remote_code=True)
            self.config = config.text_config.to_dict()
        # rename keys
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size
        return