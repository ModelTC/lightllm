import torch
import numpy as np
import torch.nn as nn
from lightllm.common.basemodel import InferStateInfo
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.req_manager import ReqManager
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.qwen2_vl.qwen2_visual import Qwen2VLVisionConfig
from typing import Any, Dict, List, Optional, Tuple, Union
from lightllm.common.basemodel.basemodel import TpPartBaseModel


def get_rope_index(
    self,
    input_ids: torch.LongTensor,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    mrope_position_deltas = []

    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
    else:
        position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        )
        mrope_position_deltas = torch.zeros(
            [input_ids.shape[0], 1],
            device=input_ids.device,
            dtype=input_ids.dtype,
        )

        return position_ids, mrope_position_deltas


class Qwen2VLConfig(TpPartBaseModel):
    model_type = "qwen2_vl"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=152064,
        hidden_size=8192,
        intermediate_size=29568,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=80,
        attention_dropout=0.0,
        vision_config=None,
        rope_scaling=None,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = Qwen2VLVisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = Qwen2VLVisionConfig()

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        # and change type from 'mrope' to 'default' because `mrope` does defeault RoPE calculations
        # one can set it to "linear"/"dynamic" etc. to have scaled RoPE
        # TODO: @raushan update config in the hub
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            if self.rope_scaling["type"] == "mrope":
                self.rope_scaling["type"] = "default"
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        # rope_config_validation(self, ignore_keys={"mrope_section"})

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Qwen2VLRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[Qwen2VLConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        # self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        # inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        # self.register_buffer("inv_freq", inv_freq, persistent=False)
        # self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block. In contrast to other models, Qwen2_VL has different position ids for thw grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


# class RotaryEmbedding(CustomOp):
#     """Original rotary positional embedding."""

#     def __init__(
#         self,
#         head_size: int,
#         rotary_dim: int,
#         max_position_embeddings: int,
#         base: int,
#         is_neox_style: bool,
#         dtype: torch.dtype,
#     ) -> None:
#         super().__init__()
#         self.head_size = head_size
#         self.rotary_dim = rotary_dim
#         self.max_position_embeddings = max_position_embeddings
#         self.base = base
#         self.is_neox_style = is_neox_style
#         self.dtype = dtype

#         cache = self._compute_cos_sin_cache()
#         # NOTE(ByronHsu): cache needs to be in FP32 for numerical stability
#         # if not _is_cuda_available:
#         #     cache = cache.to(dtype)
#         self.cos_sin_cache: torch.Tensor
#         self.register_buffer("cos_sin_cache", cache, persistent=False)

#     def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
#         """Compute the inverse frequency."""
#         # NOTE(woosuk): To exactly match the HF implementation, we need to
#         # use CPU to compute the cache and then move it to GPU. However, we
#         # create the cache on GPU for faster initialization. This may cause
#         # a slight numerical difference between the HF implementation and ours.
#         inv_freq = 1.0 / (base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
#         return inv_freq

#     def _compute_cos_sin_cache(self) -> torch.Tensor:
#         """Compute the cos and sin cache."""
#         inv_freq = self._compute_inv_freq(self.base)
#         t = torch.arange(self.max_position_embeddings, dtype=torch.float)

#         freqs = torch.einsum("i,j -> ij", t, inv_freq)
#         cos = freqs.cos()
#         sin = freqs.sin()
#         cache = torch.cat((cos, sin), dim=-1)
#         return cache

#     def forward_native(
#         self,
#         positions: torch.Tensor,
#         query: torch.Tensor,
#         key: torch.Tensor,
#         offsets: Optional[torch.Tensor] = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """A PyTorch-native implementation of forward()."""
#         if offsets is not None:
#             positions = positions + offsets
#         positions = positions.flatten()
#         num_tokens = positions.shape[0]
#         cos_sin = self.cos_sin_cache.index_select(0, positions)
#         cos, sin = cos_sin.chunk(2, dim=-1)

#         query_shape = query.shape
#         query = query.view(num_tokens, -1, self.head_size)
#         query_rot = query[..., : self.rotary_dim]
#         query_pass = query[..., self.rotary_dim :]
#         query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
#         query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

#         key_shape = key.shape
#         key = key.view(num_tokens, -1, self.head_size)
#         key_rot = key[..., : self.rotary_dim]
#         key_pass = key[..., self.rotary_dim :]
#         key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
#         key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
#         return query, key

#     # def forward_cuda(
#     #     self,
#     #     positions: torch.Tensor,
#     #     query: torch.Tensor,
#     #     key: torch.Tensor,
#     #     offsets: Optional[torch.Tensor] = None,
#     # ) -> Tuple[torch.Tensor, torch.Tensor]:
#     #     if _is_cuda_available and (self.head_size in [64, 128, 256, 512]):
#     #         apply_rope_with_cos_sin_cache_inplace(
#     #             positions=positions,
#     #             query=query,
#     #             key=key,
#     #             head_size=self.head_size,
#     #             cos_sin_cache=self.cos_sin_cache,
#     #             is_neox=self.is_neox_style,
#     #         )
#     #     else:
#     #         self.cos_sin_cache = self.cos_sin_cache.to(query.device, dtype=query.dtype)
#     #         ops.rotary_embedding(
#     #             positions,
#     #             query,
#     #             key,
#     #             self.head_size,
#     #             self.cos_sin_cache,
#     #             self.is_neox_style,
#     #         )
#     #     return query, key

#     def extra_repr(self) -> str:
#         s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
#         s += f", max_position_embeddings={self.max_position_embeddings}"
#         s += f", base={self.base}, is_neox_style={self.is_neox_style}"
#         return s


class Qwen2VLInferStateInfo(LlamaInferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_cos = None
        self.position_sin = None
        self.rotary_emb = Qwen2VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        if self.is_prefill:
            b_seq_len_numpy = self.b_seq_len.cpu().numpy()
            self.max_seq_len = b_seq_len_numpy.max()
            b_ready_cache_len_numpy = self.b_ready_cache_len.cpu().numpy()
            # position_ids = torch.from_numpy(
            #     np.concatenate(
            #         [np.arange(b_ready_cache_len_numpy[i], b_seq_len_numpy[i]) for i in range(len(b_seq_len_numpy))],
            #         axis=0,
            #     )
            # ).cuda()
            position_ids = (
                torch.arange(b_ready_cache_len_numpy.shape[1], device=b_ready_cache_len_numpy.device)
                .view(1, 1, -1)
                .expand(3, b_ready_cache_len_numpy.shape[0], -1)
            ).cuda()  # input_ids改成了b_ready_cache_len_numpy
            self.position_cos, self.position_sin = self.rotary_emb(input_ids, position_ids)  # 第一个参数应该是V
            position_ids = None
        else:
            position_ids = self.b_seq_len - 1
            self.position_cos, self.position_sin = self.rotary_emb(input_ids, position_ids)  # 第一个参数应该是V
        return
