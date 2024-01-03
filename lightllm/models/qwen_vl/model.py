import json
import numpy as np
import unicodedata
from lightllm.models.qwen.model import QWenTpPartModel
from .layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer


# Warp of the origal tokenizer
class QWenVLTokenizer:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # <img>: 151857
        self.image_start_tag = tokenizer.image_start_tag
        self.image_start_id = tokenizer.img_start_id
        # </img>: 151858
        self.image_end_tag = tokenizer.image_end_tag
        self.image_end_id = tokenizer.img_end_id
        # <imgpad>: 151859
        self.image_pad_id = tokenizer.img_pad_id
        self.image_length = 256

    def check_num(self, prompt_ids, target):
        token_num = len(np.where(np.array(prompt_ids) == self.image_pad_id)[0])
        n = token_num // self.image_length
        r = token_num % self.image_length
        assert n == target, "image num error: {} vs {}!".format(n, target)
        assert r == 0, "token_num not divided by image_length: {} vs {}".format(token_num, self.image_length)

    def _list_find(self, input_list, target, start_idx):
        cur_list = input_list[start_idx:]
        if target in cur_list:
            return cur_list.index(target) + start_idx
        return -1

    # <img>xxx</img> -> Picture {image_idx}:<img>xxx</img>\n
    def _format_prompt(self, prompt):
        parts = prompt.split(self.image_start_tag)
        prompt = parts[0]
        for idx, part in enumerate(parts[1:]):
            prompt += f'Picture {idx + 1}:' + self.image_start_tag + part
        parts = prompt.split(self.image_end_tag)
        prompt = parts[0]
        for part in parts[1:]:
            prompt += self.image_end_tag + '\n' + part
        return prompt

    # only change the impl of the encode func:
    def encode(self, prompt):
        prompt = unicodedata.normalize("NFC", prompt)
        prompt = self._format_prompt(prompt)
        origin_ids = self.tokenizer.tokenizer.encode(prompt, allowed_special='all', disallowed_special=())

        input_ids = []
        offsets = []
        lengths = []
        end = 0
        while True:
            # <img>xxx</img> -> <img><imgpad>*256</img>
            start = self._list_find(origin_ids, self.image_start_id, end)
            if start == -1:
                break
            input_ids.extend(origin_ids[end: start])
            end = self._list_find(origin_ids, self.image_end_id, start)
            if end == -1:
                raise ValueError("Unclosed image token")

            input_ids.append(self.image_start_id)
            offsets.append(len(input_ids))
            lengths.append(self.image_length)
            input_ids.extend([self.image_pad_id] * self.image_length)
            input_ids.append(self.image_end_id)
            end += 1

        input_ids.extend(origin_ids[end: ])
        return input_ids

    def __getattr__(self, name):
        if name != 'encode':
            return getattr(self.tokenizer, name)
        return self.encode


class QWenVLTpPartModel(QWenTpPartModel):

    # infer class
    pre_layer_infer_class = LlamaMultimodalPreLayerInfer

    def __init__(self, kvargs):
        from lightllm.server.tokenizer import get_tokenizer
        tokenizer = get_tokenizer(kvargs["weight_dir"], kvargs["tokenizer_mode"], trust_remote_code=kvargs["trust_remote_code"])
        self.image_pad_id = tokenizer.image_pad_id
        self.image_length = tokenizer.image_length
        super().__init__(kvargs)
        return

    def _init_config(self):
        super()._init_config()
        self.config["image_pad_id"] = self.image_pad_id
        self.config["image_length"] = self.image_length
