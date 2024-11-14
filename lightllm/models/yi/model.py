import os
import json
import torch

from .layer_weights.transformer_layer_weight import YiTransformerLayerWeight
from lightllm.models.llama.model import LlamaTpPartModel


class YiTpPartModel(LlamaTpPartModel):
    # weight class
    transformer_weight_class = YiTransformerLayerWeight

    def __init__(self, kvargs):
        super().__init__(kvargs)

    def _init_to_get_rotary(self, default_base=10000):
        partial_head_dim = int(self.config.get("partial_rotary_factor", 1) * self.head_dim_)
        if self.config.get("rope_scaling", {}) is None:
            rope_scaling_factor = 1.0
        else:
            rope_scaling_factor = self.config.get("rope_scaling", {}).get("factor", 1.0)

        base = self.config.get("rope_theta", float(default_base))
        print(base, base)
        if "max_sequence_length" in self.config:
            max_seq_len = self.config["max_sequence_length"]
        else:
            max_position_embeddings = self.config.get(
                "max_position_embeddings", 2048 if base <= 10000.0 + 1e-5 else 16384
            )
            max_seq_len = max_position_embeddings * rope_scaling_factor

        inv_freq = 1.0 / (
            base ** (torch.arange(0, partial_head_dim, 2, device="cpu", dtype=torch.float32) / partial_head_dim)
        )
        t = (
            torch.arange(max(max_seq_len + 1024 * 128, self.max_seq_length), device="cpu", dtype=torch.float32)
            / rope_scaling_factor
        )
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(self.data_type).cuda()
        self._sin_cached = torch.sin(freqs).to(self.data_type).cuda()
        print(self._cos_cached.shape)
        print(self._cos_cached)
        return
