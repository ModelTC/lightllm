import torch
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer


class Gemma3PreLayerInfer(LlamaMultimodalPreLayerInfer):
    def __init__(self, network_config, mode):
        super().__init__(network_config, mode)
        self.embed_scale = torch.tensor(network_config['hidden_size']**0.5, dtype=torch.float32)
        return

    def context_forward(self, input_ids, infer_state, layer_weight):
        input_embedding = super().context_forward(input_ids, infer_state, layer_weight)
        input_dtype = input_embedding.dtype
        return (input_embedding.float() * self.embed_scale.to(input_embedding.device).float()).to(input_dtype)

    def token_forward(self, input_ids, infer_state, layer_weight):
        input_embedding = super().token_forward(input_ids, infer_state, layer_weight)
        input_dtype = input_embedding.dtype
        return (input_embedding.float() * self.embed_scale.to(input_embedding.device).float()).to(input_dtype)

