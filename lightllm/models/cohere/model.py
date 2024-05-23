import os
import json
import torch
from lightllm.models.cohere.layer_infer.post_layer_infer import CoherePostLayerInfer
from lightllm.models.cohere.layer_infer.transformer_layer_infer import CohereTransformerLayerInfer
from lightllm.models.cohere.layer_weights.pre_and_post_layer_weight import CoherePreAndPostLayerWeight
from lightllm.models.cohere.layer_weights.transformer_layer_weight import CohereTransformerLayerWeight
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.models.llama.layer_weights.ds_load_utils import load_ds_weights
from lightllm.common.basemodel.layer_weights.hf_load_utils import load_hf_weights

from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.llama.splitfuse_infer_struct import LlamaSplitFuseInferStateInfo
from lightllm.common.basemodel import TpPartBaseModel
from lightllm.common.mem_utils import select_mem_manager_class
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

class CohereTpPartModel(LlamaTpPartModel):
    r'''
    The cohere model is modified from the llama model. 
    1. emb is the same 
    2. layer_norm is a normal layer_norm instead of the llama rms_layer_norm; only input norm, no output norm
    3. mlp is not configuable, which is bias-free.
    4. rotary_emb is the same 
    5. finial lm_head is tied to the emb
    6. res = emb + attn(iln_emb) + mlp(iln_emb)
    '''
    pre_and_post_weight_class = CoherePreAndPostLayerWeight
    transformer_weight_class = CohereTransformerLayerWeight

    pre_layer_infer_class = LlamaPreLayerInfer
    post_layer_infer_class = CoherePostLayerInfer
    transformer_layer_infer_class = CohereTransformerLayerInfer

    infer_state_class = LlamaInferStateInfo
    splitfuse_infer_state_class = LlamaSplitFuseInferStateInfo
