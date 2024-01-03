import torch
import torch.distributed as dist

from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo

from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.utils.infer_utils import mark_cost_time
from lightllm.server.embed_cache.utils import bytes2tensor, read_shm, get_shm_name_embed


"""
infer_state.multimodal_params: batch list of MultimodalParams-dict like:
   {
       "images": list of uuid (int or torch.Tensor),
       ...
   }
"""
class LlamaMultimodalPreLayerInfer(LlamaPreLayerInfer):

    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)
        self.image_pad_id = network_config["image_pad_id"]
        self.image_length = network_config["image_length"]
        return

    @mark_cost_time("pre context forward")
    def context_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        input_embdings = super().context_forward(input_ids, infer_state, layer_weight)
        if infer_state.multimodal_params is None:
            return input_embdings

        all_img_embeds = []
        for batch_id, p in enumerate(infer_state.multimodal_params):
            for uid in p["images"]:
                # pull the img_embeds by uid from cache server
                if isinstance(uid, int):
                    embed_data = read_shm(get_shm_name_embed(uid))
                    all_img_embeds.append(bytes2tensor(embed_data).reshape(self.image_length, -1))
                elif isinstance(uid, torch.Tensor):
                    all_img_embeds.append(uid)
                else:
                    raise ValueError("type of uuid = {} is not supported!".format(type(uid)))
        if len(all_img_embeds) == 0:
            return input_embdings

        all_img_embeds = torch.cat(all_img_embeds, dim=0).to(device=input_embdings.device, dtype=input_embdings.dtype)
        input_embdings[input_ids == self.image_pad_id] = all_img_embeds
        return input_embdings
