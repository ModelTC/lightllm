import torch
import torch.distributed as dist

from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.models.llama_multimodal.infer_struct import LlamaMultiModalInferStateInfo
from lightllm.utils.infer_utils import mark_cost_time
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer


def parse_input_embeds(infer_state: LlamaMultiModalInferStateInfo):
    assert isinstance(infer_state, LlamaMultiModalInferStateInfo)
    if infer_state.kwargs and 'inputs_embeds' in infer_state.kwargs:
        token_num, _ = infer_state.kwargs['inputs_embeds'].shape
        assert token_num == infer_state.total_token_num, "inputs_embeds token_num != infer_state.total_token_num: {} vs {}".foramt(
            token_num, infer_state.total_token_num)
        return infer_state.kwargs['inputs_embeds']
    return None


class LlamaMultiModalPreLayerInfer(LlamaPreLayerInfer):

    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)
        return

    @mark_cost_time("pre context forward")
    def context_forward(self, input_ids, infer_state: LlamaMultiModalInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        ret = parse_input_embeds(infer_state)
        if ret is not None:
            return ret
        else:
            return super().context_forward(input_ids, infer_state, layer_weight)    

    def token_forward(self, input_ids, infer_state: LlamaMultiModalInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        ret = parse_input_embeds(infer_state)
        if ret is not None:
            return ret
        else:
            return super().token_forward(input_ids, infer_state, layer_weight)

    def get_input_embeddings(self, input_ids, layer_weight: LlamaPreAndPostLayerWeight):
        input_mask = torch.logical_or(self.vob_start_id_ > input_ids, input_ids >= self.vob_end_id_)
        tmp_input_ids = (input_ids - self.vob_start_id_)
        tmp_input_ids[input_mask] = 0
        input_embdings = torch.embedding(layer_weight.wte_weight_, tmp_input_ids, padding_idx=-1)
        input_embdings[input_mask] = 0.0
        if self.world_size_ > 1:
            dist.all_reduce(input_embdings, op=dist.ReduceOp.SUM, async_op=False)
        return input_embdings
