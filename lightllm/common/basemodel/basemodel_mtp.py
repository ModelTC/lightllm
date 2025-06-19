from lightllm.common.basemodel.basemodel import BaseModelRunner


class BaseMTPModelRunner(BaseModelRunner):
    def __init__(self, kvargs: dict):
        self._pre_init(kvargs)
        super().__init__(kvargs)
        return

    def _pre_init(self, kvargs: dict):
        self.main_model: BaseModelRunner = kvargs.pop("main_model")
        self.mem_layer_start = kvargs.pop("mem_layer_start", 0)
        return

    def _init_custom(self):
        self._cos_cached = self.main_model._cos_cached
        self._sin_cached = self.main_model._sin_cached
        return

    def _init_req_manager(self):
        self.req_manager = self.main_model.req_manager
        return

    def _init_mem_manager(self):
        self.mem_manager = self.main_model.mem_manager
        return

    def _init_weights(self):
        super()._init_weights()
        self.pre_post_weight.wte_weight_ = self.main_model.pre_post_weight.wte_weight_
        self.pre_post_weight.lm_head_weight_ = self.main_model.pre_post_weight.lm_head_weight_
        return

    def _init_infer_layer(self):
        super()._init_infer_layer()
        # reset the layer_num_ of the self.layers_infer
        for layer in self.layers_infer:
            layer.layer_num_ = layer.layer_num_ + self.mem_layer_start
        return
