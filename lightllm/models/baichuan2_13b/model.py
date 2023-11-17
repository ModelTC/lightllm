from .layer_weights.transformer_layer_weight import BaiChuan2_13bTransformerLayerWeight
from .layer_infer.transformer_layer_infer import Baichuan2_13bTransformerLayerInfer
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.llama.model import LlamaTpPartModel

class Baichuan2_13bTpPartModel(LlamaTpPartModel):
    # weight class
    transformer_weight_class = BaiChuan2_13bTransformerLayerWeight

    # infer class
    transformer_layer_infer_class = Baichuan2_13bTransformerLayerInfer

    # infer state class
    infer_state_class = InferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return
    
    def _init_config(self):
        super()._init_config()
        # rename key
        # repair_config()
        return
    
    def _verify_params(self):
        assert self.load_way == "HF", "llama only support HF format to load Now!"
