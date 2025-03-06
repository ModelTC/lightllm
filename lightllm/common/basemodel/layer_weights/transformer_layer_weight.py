from functools import partial

# from lightllm.common.layers.mm import MM
from .base_layer_weight import BaseLayerWeight
from .meta_weights import BaseWeight, MultiMMWeightTpl
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class TransformerLayerWeight(BaseLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg):
        super().__init__()
        self.layer_num_ = layer_num
        self.tp_rank_ = tp_rank
        self.world_size_ = world_size
        self.data_type_ = data_type
        self.network_config_ = network_config
        self.mode = mode
        self.quant_cfg = quant_cfg
        self._parse_config()
        self._init_weight_names()
        self._init_qweight_names()
        self._init_weight()
        return

    def _parse_config(self):
        pass

    def _init_weight_names(self):
        pass

    def _init_qweight_names(self):
        pass

    def _init_weight(self):
        pass

    def load_hf_weights(self, weights):
        """
        load weights
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name, None)
            if isinstance(attr, MultiMMWeightTpl):
                with self.lock:
                    attr.load_hf_weights(weights)
            elif isinstance(attr, BaseWeight):
                attr.load_hf_weights(weights)
