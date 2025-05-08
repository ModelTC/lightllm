import os
import torch
import torch.distributed as dist
import numpy as np

from lightllm.models.deepseek_mtp.layer_weights.pre_and_post_layer_weight import Deepseek3MTPPreAndPostLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.utils.infer_utils import mark_cost_time
from lightllm.models.llama.triton_kernel.embedding import embedding
from lightllm.distributed.communication_op import all_reduce
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.distributed.communication_op import all_gather


class Deepseek3MTPPreLayerInfer(LlamaPreLayerInfer):
    """ """

    def __init__(self, network_config, mode):
        super().__init__(network_config, mode)
        self.eps_ = network_config["rms_norm_eps"]
        self.hidden_size = network_config["hidden_size"]
        return
    
    def mtp_context_forward(self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight: Deepseek3MTPPreAndPostLayerWeight):
        assert infer_state.spec_info is not None, "need spec info for mtp model."
        tgt_embdings = infer_state.spec_info
        assert input_embdings.shape[0] == tgt_embdings.shape[0]
        rmsnorm_forward(input_embdings, weight=layer_weight.enorm_weight_, eps=self.eps_, out=input_embdings)
        rmsnorm_forward(tgt_embdings, weight=layer_weight.hnorm_weight_, eps=self.eps_, out=tgt_embdings)
        
        cat_embdings = self.alloc_tensor((input_embdings.shape[0], 
                                          input_embdings.shape[1] + tgt_embdings.shape[1]), 
                                         dtype=input_embdings.dtype)
        torch.cat((input_embdings, tgt_embdings), dim=-1, out=cat_embdings)
        infer_state.spec_info = None
        
        cat_embdings = cat_embdings.permute(1, 0)
        proj_embdings = self.alloc_tensor(
            (layer_weight.eh_proj_weight_.shape[0], cat_embdings.shape[1]), dtype=input_embdings.dtype
        )
        torch.mm(layer_weight.eh_proj_weight_, cat_embdings, out=proj_embdings)
        
        cat_embdings = None
        if self.tp_world_size_ == 1:
            gather_data = proj_embdings
        else:
            gather_data = self.alloc_tensor((self.hidden_size, input_embdings.shape[0]), dtype=input_embdings.dtype)
            split_indexes = np.linspace(0, self.hidden_size, self.tp_world_size_ + 1, dtype=np.int64)
            all_gather(
                [gather_data[split_indexes[i] : split_indexes[i + 1], :] for i in range(self.tp_world_size_)],
                proj_embdings,
                group=infer_state.dist_group,
                async_op=False,
            )
        proj_embdings = None
        ans_logics = self.alloc_tensor(
            (input_embdings.shape[0], self.hidden_size),
            dtype=input_embdings.dtype,
            is_graph_out=True,
            microbatch_index=infer_state.microbatch_index,
        )
        ans_logics[:, :] = gather_data.permute(1, 0)
        gather_data = None
        return ans_logics
        
    def mtp_token_forward(self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight: Deepseek3MTPPreAndPostLayerWeight):
        assert infer_state.spec_info is not None, "need spec info for mtp model."
        tgt_embdings = infer_state.spec_info
        assert input_embdings.shape[0] == tgt_embdings.shape[0]
        rmsnorm_forward(input_embdings, weight=layer_weight.enorm_weight_, eps=self.eps_, out=input_embdings)
        rmsnorm_forward(tgt_embdings, weight=layer_weight.hnorm_weight_, eps=self.eps_, out=tgt_embdings)
        
        cat_embdings = self.alloc_tensor((input_embdings.shape[0], 
                                          input_embdings.shape[1] + tgt_embdings.shape[1]), 
                                         dtype=input_embdings.dtype)
        torch.cat((input_embdings, tgt_embdings), dim=-1, out=cat_embdings)
        infer_state.spec_info = None
        
        cat_embdings = cat_embdings.permute(1, 0)
        proj_embdings = self.alloc_tensor(
            (layer_weight.eh_proj_weight_.shape[0], cat_embdings.shape[1]), dtype=cat_embdings.dtype
        )
        torch.mm(layer_weight.eh_proj_weight_, cat_embdings, out=proj_embdings)
        
        cat_embdings = None
        if self.tp_world_size_ == 1:
            gather_data = proj_embdings
        else:
            gather_data = self.alloc_tensor((self.hidden_size, input_embdings.shape[0]), dtype=input_embdings.dtype)
            split_indexes = np.linspace(0, self.hidden_size, self.tp_world_size_ + 1, dtype=np.int64)
            all_gather(
                [gather_data[split_indexes[i] : split_indexes[i + 1], :] for i in range(self.tp_world_size_)],
                proj_embdings,
                group=infer_state.dist_group,
                async_op=False,
            )
        proj_embdings = None
        ans_logics = self.alloc_tensor(
            (input_embdings.shape[0], self.hidden_size),
            dtype=input_embdings.dtype,
            is_graph_out=True,
            microbatch_index=infer_state.microbatch_index,
        )
        ans_logics[:, :] = gather_data.permute(1, 0)
        gather_data = None
        return ans_logics
        
    def context_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: Deepseek3MTPPreAndPostLayerWeight):
        input_embdings = super().context_forward(input_ids, infer_state, layer_weight)
        return self.mtp_context_forward(input_embdings, infer_state, layer_weight)
 
    def token_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: Deepseek3MTPPreAndPostLayerWeight):
        input_embdings = super().token_forward(input_ids, infer_state, layer_weight)
        return self.mtp_token_forward(input_embdings, infer_state, layer_weight)