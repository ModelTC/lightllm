from functools import partial
from lightllm.common.layers.mm import MM
from .base_layer_weight import BaseLayerWeight


class TransformerLayerWeight(BaseLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode):
        super().__init__()
        self.layer_num_ = layer_num
        self.tp_rank_ = tp_rank
        self.world_size_ = world_size
        self.data_type_ = data_type
        self.network_config_ = network_config
        self.mode = mode
        self.mm_op = MM(mode)
        self.mm_op.preprocess_weight = partial(self.mm_op.preprocess_weight, func=self._cuda)
        self.init_static_params()
        return
