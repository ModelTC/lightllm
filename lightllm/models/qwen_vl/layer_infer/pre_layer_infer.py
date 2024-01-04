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
       "images": [
           {
               "uuid": int,
               "token_id": int, image token start id,
               "token_num": int, image token num,
           },
       ]
       ...
   }
"""
class LlamaMultimodalPreLayerInfer(LlamaPreLayerInfer):

    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)
        return

    @mark_cost_time("pre context forward")
    def context_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        input_embdings = super().context_forward(input_ids, infer_state, layer_weight)
        print("input_ids:", input_ids)
        if infer_state.multimodal_params is None:
            return input_embdings

        embeds = []
        token_nums = []
        token_ids = []

        for batch_id, p in enumerate(infer_state.multimodal_params):
            for img in p["images"]:
                # skip the same image
                if img["token_id"] in token_ids:
                    continue
                # pull the img_embeds by uid from shm
                embed_data = read_shm(get_shm_name_embed(img["uuid"]))
                embeds.append(bytes2tensor(embed_data).reshape(img["token_num"], -1))
                token_ids.append(img["token_id"])
                token_nums.append(img["token_num"])

        if len(embeds) == 0:
            return input_embdings

        device = input_embdings.device
        dtype = input_embdings.dtype
        # embeds = torch.cat(embeds, dim=0).to(device=device, dtype=dtype)
        # token_ids = torch.Tensor(token_ids).to(device=device, dtype=torch.int64)
        # token_nums = torch.Tensor(token_nums).to(device=device, dtype=torch.int64)
        print("token_ids", token_ids, "token_nums", token_nums)

        for embed, token_id, token_num in zip(embeds, token_ids, token_nums):
            # find all idx satisfied input_ids[idx] == token_id
            # assume input_ids[idx: idx+token_num] == [token_id, token_id + 1, ... token_id + token_num - 1]
            print("embed:", embed.shape, token_id, token_num)
            for idx in torch.where(input_ids == token_id)[0]:
                print("offset:", idx)
                input_embdings[idx: idx + token_num] = embed.to(device=device, dtype=dtype)
        return input_embdings
