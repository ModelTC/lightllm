import torch
from lightllm.models.llama_multimodal.layer_infer.pre_layer_infer import LlamaMultiModalPreLayerInfer
from lightllm.common.infer_utils import init_req_to_token_indexes
from lightllm.models.llama.model import LlamaTpPartModel


class LlamaTpPartMultiModal(LlamaTpPartModel):
    # infer class
    pre_layer_infer_class = LlamaMultiModalPreLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)

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
            repad_embeds=None):
        if is_prefill:
            return self._prefill(batch_size, total_token_num, max_len_in_batch, input_ids, b_req_idx, b_start_loc, b_seq_len, repad_embeds)
        else:
            return self._decode(batch_size, total_token_num, max_len_in_batch, input_ids, b_req_idx, b_start_loc, b_seq_len)

    
    def _prefill(self, batch_size, total_token_num, max_len_in_batch, input_ids, b_req_idx, b_start_loc, b_seq_len, repad_embeds):
        infer_state = self.infer_state_class()
        infer_state.is_prefill = True
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        infer_state.repad_embeds = repad_embeds
        assert (input_ids.shape[0] == total_token_num)
        assert (b_req_idx.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])
        infer_state.b_req_idx = b_req_idx
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len

        infer_state.mem_manager = self.mem_manager
        infer_state.req_manager = self.req_manager
        infer_state.prefill_mem_index = self.mem_manager.alloc(infer_state.total_token_num)
        infer_state.prefill_key_buffer = torch.empty((infer_state.total_token_num, self.tp_k_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
        infer_state.prefill_value_buffer = torch.empty((infer_state.total_token_num, self.tp_v_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
        init_req_to_token_indexes(self.req_manager.req_to_token_indexs, b_req_idx, b_seq_len,
                                   max_len_in_batch, infer_state.prefill_mem_index)

        infer_state.init_some_extra_state(self, batch_size, total_token_num, max_len_in_batch, 
                                          input_ids, self.req_manager.req_to_token_indexs, b_req_idx,
                                          b_start_loc, b_seq_len, True)
        predict_logics = self._context_forward(input_ids, infer_state)
        return predict_logics
