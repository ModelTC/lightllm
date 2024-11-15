from functools import partial

# from lightllm.common.layers.mm import MM
from .base_layer_weight import BaseLayerWeight
from .meta_weights import MMWeight
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
        n_embed = self.network_config_["hidden_size"]
        # Dealing with head_dim_!=n_embed // num_attention_heads scenarios, such as mistral 13B
        head_dim = n_embed // self.network_config_["num_attention_heads"]
        self.head_dim = self.network_config_.get("head_dim", head_dim)
        self.init_static_params()

        self.fuse_pairs = {"k_proj&v_proj": "kv_proj"}
        self.init_qkv()
        self.init_o()
        self.init_ffn()
        self.init_norm()
        self.set_quantization()
        return

    def load_hf_weights(self, weights):
        super().load_hf_weights(weights)
        self.fuse_weights()

    def fuse_weights(self):
        for pair_name, fuse_name in self.fuse_pairs.items():
            attr1_name, attr2_name = pair_name.split("&")
            with self.lock:
                if hasattr(self, fuse_name):
                    continue
                attr1 = getattr(self, attr1_name)
                attr2 = getattr(self, attr2_name)
                if attr1.verify_load() and attr2.verify_load():
                    attr1.fuse(attr2)
                    setattr(self, fuse_name, attr1)
                    delattr(self, attr2_name)

    def set_quantization(self):
        if self.quant_cfg.quant_type is None:
            return
        mix_quant_list = self.quant_cfg.get_mixed_list(self.layer_num_)
        # fused layers must have the same quant_method
        for pair_name, fuse_name in self.fuse_pairs.items():
            attr1_name, attr2_name = pair_name.split("&")
            if attr1_name not in mix_quant_list and attr2_name not in mix_quant_list:
                continue
            attr1_quant_type = self.quant_cfg.get_quant_type(self.layer_num_, attr1_name)
            attr2_quant_type = self.quant_cfg.get_quant_type(self.layer_num_, attr2_name)
            assert (
                attr1_quant_type == attr2_quant_type
            ), f"""{attr1_name} and {attr2_name} expects the the same quant type,
            but gets {attr1_quant_type} and {attr2_quant_type}."""

        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, MMWeight):
                if attr_name in mix_quant_list:
                    attr.set_quant_method(self.quant_cfg.get_quant_method(self.layer_num_, attr_name))
                    attr_quant_type = self.quant_cfg.get_quant_type(self.layer_num_, attr_name)
                    logger.info(f"Layer {self.layer_num_} {attr_name} is set to {attr_quant_type}")
                else:
                    attr.set_quant_method(self.quant_cfg.get_default_quant_method())
