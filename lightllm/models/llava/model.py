import json
import numpy as np
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer


# Warp of the origal tokenizer
class LlavaTokenizer:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.image_token = "<image>"
        # (image_size // patch_size) ** 2: (336 // 14) ** 2
        self.image_length = 576
        self.image_pad_id = self.tokenizer.pad_token_id

    def check_num(self, prompt_ids, target):
        token_num = len(np.where(np.array(prompt_ids) == self.image_pad_id)[0])
        n = token_num // self.image_length
        r = token_num % self.image_length
        assert n == target, "image num error: {} vs {}!".format(n, target)
        assert r == 0, "token_num not divided by image_length: {} vs {}".format(token_num, self.image_length)

    # only change the impl of the encode func:
    def encode(self, prompt):

        # split prompt by <image>, and merge parts by [pad_id] * 576
        ids_chunks = [self.tokenizer(x).input_ids for x in prompt.split(self.image_token)]
        input_ids = ids_chunks[0]
        offsets = []
        lengths = []

        for ids in ids_chunks[1:]:
            # skip the start token
            if len(ids) > 0 and ids[0] == self.tokenizer.bos_token_id:
                ids = ids[1:]

            offsets.append(len(input_ids))
            lengths.append(self.image_length)
            input_ids.extend([self.tokenizer.pad_token_id] * self.image_length)
            input_ids.extend(ids)

        return input_ids

    def __getattr__(self, name):
        if name != 'encode':
            return getattr(self.tokenizer, name)
        return self.encode


class LlavaTpPartModel(LlamaTpPartModel):

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
