import torch
import torch.nn.functional as F
import json
import os
from transformers import CLIPVisionModel, CLIPImageProcessor

from lightllm.models.llava.layer_weights.pre_and_post_layer_weight import LlavaPreAndPostLayerWeight
from lightllm.models.llama_multimodal.model import LlamaTpPartMultiModal 
from .utils import load_image

LLAVA_IMAGE_TOKEN_ID = -200
LLAVA_IMAGE_TOKEN = "<image>"


class LlavaTpPartMulitModal(LlamaTpPartMultiModal):

    # weight class
    pre_and_post_weight_class = LlavaPreAndPostLayerWeight

    def __init__(self, kvargs):
        super().__init__(kvargs)

        config_file = os.path.join(self.weight_dir_, "config.json")
        config = json.load(open(config_file))

        self.select_layer = config.get('mm_vision_select_layer', -2)
        self.select_feature = config.get('mm_vision_select_feature', 'patch')

        # load clip vision model by cfg['mm_vision_tower']:
        #   huggingface_name or path_of_clip_relative_to_llava_model_dir
        vision_path = config.get('mm_vision_tower', 'openai/clip-vit-large-patch14-336')
        if vision_path.startswith("./"):
            vision_path = os.path.join(self.weight_dir_, vision_path)
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_path)
        self.vision_tower = CLIPVisionModel.from_pretrained(vision_path).half().cuda()
        self.vision_tower.requires_grad_(False)

        self.image_size = self.vision_tower.config.image_size
        self.patch_size = self.vision_tower.config.patch_size
        self.image_feature_len = int((self.image_size / self.patch_size) ** 2)
        if self.select_feature == 'patch':
            pass
        elif self.select_feature == 'cls_patch':
            self.image_feature_len += 1
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return

    @torch.no_grad()
    def forward(
            self,
            batch_size,
            total_token_num,
            max_len_in_batch,
            input_ids : torch.Tensor,
            b_req_idx : torch.Tensor,
            b_start_loc : torch.Tensor,
            b_seq_len : torch.Tensor,
            is_prefill=True,
            multimodal_inputs=None):

        repad_embeds = self.get_repad_embeds(multimodal_inputs) if is_prefill else None
        return super().forward(batch_size, total_token_num, max_len_in_batch, input_ids, b_req_idx, b_start_loc, b_seq_len, is_prefill, repad_embeds=repad_embeds)

    def get_repad_embeds(self, multimodal_inputs):
        image_paths = [x['image_path'] for x in multimodal_inputs]
        offsets = [x['offset'] for x in multimodal_inputs]
        # load valid images
        images = []
        valid_idxs = []
        for idx, image_path in enumerate(image_paths):
            try:
                images.append(load_image(image_path))
                valid_idxs.append(idx)
            except Exception as e:
                print("Error when LlavaTpPartMulitModal load image {}: {}".format(image_path, e))
        if len(valid_idxs) <= 0:
            return [(None, None) for _ in offsets]

        # batch images infer
        image_tensor = self.image_processor.preprocess(images, return_tensors='pt')['pixel_values'].half().cuda()
        image_feature = self.vision_tower(image_tensor, output_hidden_states=True)
        embeds = image_feature.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            embeds = embeds[:, 1:].contiguous()

        B, L, N = embeds.shape
        embeds = embeds.view(-1, N)
        assert L == self.image_feature_len, "error image feature shape: {} vs {}".format(embeds.shape, self.image_feature_len)

        # mm_project
        embeds = F.linear(
            embeds,
            weight=self.pre_post_weight.mm_projector_0_weight,
            bias=self.pre_post_weight.mm_projector_0_bias
        )
        embeds = F.gelu(embeds)
        embeds = F.linear(
            embeds,
            weight=self.pre_post_weight.mm_projector_2_weight,
            bias=self.pre_post_weight.mm_projector_2_bias
        )
        embeds = embeds.view(B, L, -1)

        # gen repad embeds
        repad_embeds = []
        for idx, offset in enumerate(offsets):
            if idx not in valid_idxs:
                repad_embeds.append((None, None))
            else:
                embed = embeds[valid_idxs.index(idx)]
                repad_embeds.append((embed, offset))
        return repad_embeds

    def pad_input_ids(self, input_ids):
        pad_ids = [0] * self.image_feature_len
        offset = input_ids.index(LLAVA_IMAGE_TOKEN_ID)
        # old_len + pad_len - 1, because we need to remove image_token_id
        new_input_ids = input_ids[:offset] + pad_ids + input_ids[offset + 1:]
        return new_input_ids, offset
