import torch
import numpy as np
import threading
from lightllm.common.basemodel.layer_weights.meta_weights import BaseWeight
from lightllm.utils.dist_utils import get_current_device_id


class BaseLayerWeight:
    def __init__(self):
        self.tp_rank_ = None
        self.lock = threading.Lock()

    def load_hf_weights(self, weights):
        """
        load weights
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name, None)
            if isinstance(attr, BaseWeight):
                attr.load_hf_weights(weights)

    def init_static_params(self):
        """
        design for some static init params, many model dont need do this.
        """
        pass

    def verify_load(self):
        """
        verify all load is ok
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, BaseWeight):
                assert attr.verify_load(), f"Loading {attr_name} of layers {self.layer_num_} fails."

    def _cuda(self, cpu_tensor):
        if self.tp_rank_ is None:
            return cpu_tensor.contiguous().to(self.data_type_).cuda()
        else:
            return cpu_tensor.contiguous().to(self.data_type_).cuda(get_current_device_id())

    def _try_cat_to(self, source_tensor_names, cat_dim):
        source_tensor = [getattr(self, name) for name in source_tensor_names]
        if all(src_tensor is not None for src_tensor in source_tensor):
            with self.lock:
                ans = torch.cat(source_tensor, dim=cat_dim)
                for name in source_tensor_names:
                    delattr(self, name)
                return ans
        return None
