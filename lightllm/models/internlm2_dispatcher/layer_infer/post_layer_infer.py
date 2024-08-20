import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np

from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.models.internlm2_dispatcher.layer_weights.pre_and_post_layer_weight import (
    Internlm2DispatcherPreAndPostLayerWeight,
)
from einops import rearrange


class Internlm2DispatcherPostLayerInfer(LlamaPostLayerInfer):
    def cls_forward(
        self, last_input, infer_state: LlamaInferStateInfo, layer_weight: Internlm2DispatcherPreAndPostLayerWeight
    ):
        cls0_out = F.gelu(torch.mm(layer_weight.cls0_weight_, last_input) + layer_weight.cls0_bias_[:, None])
        cls1_out = F.gelu(torch.mm(layer_weight.cls2_weight_, cls0_out) + layer_weight.cls2_bias_[:, None])
        cls2_out = torch.mm(layer_weight.cls4_weight_, cls1_out) + layer_weight.cls4_bias_[:, None]
        return cls2_out

    def token_forward(
        self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight: Internlm2DispatcherPreAndPostLayerWeight
    ):
        last_input, token_num = self._slice_get_last_input(input_embdings, infer_state)
        input_embdings_dtype = input_embdings.dtype
        input_embdings = None
        last_input = self._norm(last_input, infer_state, layer_weight)
        last_input = rearrange(last_input, "batch embed_dim -> embed_dim batch").contiguous().reshape(-1, token_num)
        logic_batch = torch.mm(layer_weight.lm_head_weight_, last_input)
        cls_out = self.cls_forward(last_input, infer_state, layer_weight).permute(1, 0)
        probs = torch.softmax(cls_out, dim=-1)[:, 1]

        last_input = None
        if self.world_size_ == 1:
            gather_data = logic_batch
        else:
            gather_data = torch.empty(
                (self.vocab_size_, token_num), device=logic_batch.device, dtype=input_embdings_dtype
            )
            split_indexes = np.linspace(0, self.vocab_size_, self.world_size_ + 1, dtype=np.int64)
            dist.all_gather(
                [gather_data[split_indexes[i] : split_indexes[i + 1], :] for i in range(self.world_size_)],
                logic_batch,
                group=None,
                async_op=False,
            )
        logic_batch = None

        ans_logics = gather_data.permute(1, 0).float()
        gather_data = None
        return ans_logics, probs
