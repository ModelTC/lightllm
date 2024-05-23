import torch
from functools import partial

from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.basemodel.layer_infer.template.transformer_layer_infer_template import TransformerLayerInferTpl
from lightllm.common.basemodel.splitfuse_infer_struct import SplitFuseInferStateInfo
from lightllm.models.cohere.layer_weights.transformer_layer_weight import CohereTransformerLayerWeight
from lightllm.models.cohere.triton_kernels.layernorm import layernorm_forward, mh_layernorm_forward
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer

from lightllm.models.llama.infer_struct import LlamaInferStateInfo
import torch.distributed as dist

from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.utils.infer_utils import mark_cost_time



class CohereTransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        network_config["rms_norm_eps"] = network_config["layer_norm_eps"] # cohere uses layer_norm_eps
        self.use_qk_norm = network_config.get("use_qk_norm", False)
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.eps_ = network_config["layer_norm_eps"] # overwrite eps
        self._bind_func()
        return
    
    def _att_norm(self, input, infer_state: LlamaInferStateInfo, layer_weight: CohereTransformerLayerWeight):
        return layernorm_forward(input, weight=layer_weight.att_norm_weight_, eps=self.eps_)

    def _q_norm(self, input, infer_state: LlamaInferStateInfo, layer_weight: CohereTransformerLayerWeight):
        return mh_layernorm_forward(input, weight=layer_weight.q_norm_weight_, eps=self.eps_)

    def _k_norm(self, input, infer_state: LlamaInferStateInfo, layer_weight: CohereTransformerLayerWeight):
        return mh_layernorm_forward(input, weight=layer_weight.k_norm_weight_, eps=self.eps_)

    def _bind_norm(self):
        self._att_norm = partial(CohereTransformerLayerInfer._att_norm, self)
        self._ffn_norm = None # no ffn norm in cohere models
        self._q_norm = partial(CohereTransformerLayerInfer._q_norm, self) if self.use_qk_norm else None
        self._k_norm = partial(CohereTransformerLayerInfer._k_norm, self) if self.use_qk_norm else None

    def _get_qkv(
        self, input, cache_kv, infer_state, layer_weight
    ) -> torch.Tensor:
        q = torch.mm(input.view(-1, self.embed_dim_), layer_weight.q_weight_)
        torch.mm(
            input.view(-1, self.embed_dim_),
            layer_weight.kv_weight_,
            out=cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_),
        )
        if self.use_qk_norm:
            q = q.view(-1, self.tp_q_head_num_, self.head_dim_)
            k = cache_kv[:, 0 : self.tp_k_head_num_, :]
            q = self._q_norm(q, infer_state, layer_weight)
            cache_kv[:, 0 : self.tp_k_head_num_, :] = self._k_norm(k, infer_state, layer_weight)
        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, 0 : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        return q, cache_kv

    @mark_cost_time("trans context ffn forward time cost")  # dont to remove this, will make performence down, did not know why
    def _context_ffn(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        input1 = input_embdings
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        infer_state._ffn_out = ffn_out
        return

    def _token_ffn(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        input1 = input_embdings
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        infer_state._ffn_out = ffn_out
        return
    
    # @mark_cost_time("trans context ffn forward time cost")  # dont to remove this, will make performence down, did not know why
    def _splitfuse_ffn(self, input_embdings, infer_state: SplitFuseInferStateInfo, layer_weight):
        input1 = input_embdings
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        infer_state._ffn_out = ffn_out
        return
    
    @mark_cost_time("trans context flash forward time cost")  # dont to remove this, will make performence down, did not know why
    def _context_attention(self, input_embding, infer_state: InferStateInfo, layer_weight):
        input1 = input_embding
        cache_kv = self._pre_cache_kv(infer_state, layer_weight)
        q, cache_kv  = self._get_qkv(input1, cache_kv, infer_state, layer_weight)
        input1 = None
        o = self._context_attention_kernel(q, cache_kv, infer_state, layer_weight)
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        infer_state._attn_out = o
        return

    # this impl dont to use @mark_cost_time
    def _token_attention(self, input_embding, infer_state: InferStateInfo, layer_weight):
        input1 = input_embding
        cache_kv = self._pre_cache_kv(infer_state, layer_weight)
        q, cache_kv = self._get_qkv(input1, cache_kv, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        infer_state._attn_out = o
        return
    
    # @mark_cost_time("trans context flash forward time cost")  # dont to remove this, will make performence down, did not know why
    def _splitfuse_attention(self, input_embding, infer_state: SplitFuseInferStateInfo, layer_weight):
        input1 = input_embding
        cache_kv = self._pre_cache_kv(infer_state, layer_weight)
        q, cache_kv  = self._get_qkv(input1, cache_kv, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._splitfuse_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        infer_state._attn_out = o
        return

    def _cohere_residual(self, input_embdings, infer_state: InferStateInfo):
        emb_addr = input_embdings.data_ptr()
        attn_out_addr = infer_state._attn_out.data_ptr()
        ffn_addr = infer_state._ffn_out.data_ptr()
        assert emb_addr != attn_out_addr
        assert emb_addr != ffn_addr
        assert attn_out_addr != ffn_addr
        input_embdings.add_(infer_state._attn_out.view(-1, self.embed_dim_) + infer_state._ffn_out.view(-1, self.embed_dim_))

    def context_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        self._context_attention(input1,
                                      infer_state,
                                      layer_weight=layer_weight)
        self._context_ffn(input1, infer_state, layer_weight)
        self._cohere_residual(input_embdings, infer_state)
        return input_embdings

    def token_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        self._token_attention(input1,
                                    infer_state,
                                    layer_weight=layer_weight)
        self._token_ffn(input1, infer_state, layer_weight)
        self._cohere_residual(input_embdings, infer_state)
        return input_embdings
    
    def splitfuse_forward(self, input_embdings, infer_state: SplitFuseInferStateInfo, layer_weight):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        self._splitfuse_attention(input1,
                            infer_state,
                            layer_weight=layer_weight)
        self._splitfuse_ffn(input1, infer_state, layer_weight)
        self._cohere_residual(input_embdings, infer_state)
        return input_embdings
