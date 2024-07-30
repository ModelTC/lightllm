import os
import json
from lightllm.models.internlm2.model import Internlm2TpPartModel
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.phi3.model import Phi3TpPartModel
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.server.multimodal_params import MultimodalParams, ImageItem
from lightllm.common.build_utils import repair_config
from lightllm.models.internvl.layer_weights.pre_and_post_layer_weight import InternVLLlamaPreAndPostLayerWeight
# from lightllm.models.vit import get_image_patch
from lightllm.models.llava.llava_visual import LlavaVisionModel
from lightllm.models.internvl.img_process import get_image_patch
from typing import Dict
import lightllm.models.internvl.internvl_visual
import torch
import numpy

IMG_START_TOKEN='<img>'
IMG_END_TOKEN='</img>'
IMG_TOKEN='<image>'

BEGIN_CONTEXT = '<|system|>\nYou are an AI assistant whose name is Phi-3.<|end|><|user|>\n'
END_CONTEXT = '<|end|><|assistant|>\n'

# Warp of the origal tokenizer
class InternvlTokenizer:

    def __init__(self, tokenizer, model_cfg, **kwargs):
        self.tokenizer = tokenizer
        self.image_length = 256

        self.image_start_tag = IMG_START_TOKEN
        self.image_start_id = tokenizer.convert_tokens_to_ids(self.image_start_tag)

        self.image_end_tag = IMG_END_TOKEN
        self.image_end_id = tokenizer.convert_tokens_to_ids(self.image_end_tag)

    def get_image_token_length(self, img: ImageItem):
        return get_image_patch(img.image_w, img.image_h, use_thumbnail=True) * self.image_length
    
    # only change the impl of the encode func:
    def encode(self, prompt, multimodal_params: MultimodalParams = None, **kwargs):
        image_tokens = IMG_START_TOKEN + IMG_END_TOKEN 
        Image_num = len(multimodal_params.images)
        # <image> --> <img></img>
        while(Image_num):
            Image_num -= 1
            prompt = prompt.replace(IMG_TOKEN, image_tokens, 1)
        
        prompt = BEGIN_CONTEXT + prompt + END_CONTEXT

        origin_ids = self.tokenizer.encode(prompt)
        input_ids = []
        image_id = 0
        for i in range(len(origin_ids) - 1):
            input_ids.append(origin_ids[i])
            if origin_ids[i] == self.image_start_id and origin_ids[i + 1] == self.image_end_id:
                token_id = multimodal_params.images[image_id].token_id
                token_num = multimodal_params.images[image_id].token_num
                input_ids.extend(range(token_id, token_id + token_num))
                image_id += 1
        input_ids.append(origin_ids[-1])
        return input_ids

    def __getattr__(self, name):
        if name != 'encode':
            return getattr(self.tokenizer, name)
        return self.encode



class InternVLphi3TpPartModel(Phi3TpPartModel):
    # weight class
    pre_and_post_weight_class = InternVLLlamaPreAndPostLayerWeight

    # infer class
    pre_layer_infer_class = LlamaMultimodalPreLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            self.config = json.load(json_file)["llm_config"]
        # rename keys
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size
        return
