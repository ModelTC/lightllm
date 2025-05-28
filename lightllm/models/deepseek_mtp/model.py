from lightllm.models.deepseek2.model import Deepseek2TpPartModel
from lightllm.models.deepseek_mtp.layer_infer.pre_layer_infer import Deepseek3MTPPreLayerInfer
from lightllm.models.deepseek_mtp.layer_weights.pre_and_post_layer_weight import Deepseek3MTPPreAndPostLayerWeight
from lightllm.utils.envs_utils import enable_env_vars, get_env_start_args
from lightllm.models.deepseek2.flashinfer_struct import Deepseek2FlashInferStateInfo
from lightllm.utils.dist_utils import get_dp_world_size
import torch
from lightllm.distributed.communication_op import CustomProcessGroup, dist_group_manager
from lightllm.common.basemodel.triton_kernel.copy_kv_index_to_req import copy_kv_index_to_req
from lightllm.common.req_manager import ReqManager
from lightllm.common.infer_utils import init_req_to_token_indexes
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.common.basemodel.cuda_graph import CudaGraph
from .deepseek3_mtp_mem_manager import Deepseek3MTPMemoryManager


class Deepseek3MTPModel(Deepseek2TpPartModel):

    pre_and_post_weight_class = Deepseek3MTPPreAndPostLayerWeight
    pre_layer_infer_class = Deepseek3MTPPreLayerInfer

    def __init__(self, kvargs):
        self.main_model = kvargs.pop("main_model")
        self.req_manager = self.main_model.req_manager
        self.last_mtp_module = kvargs.pop("last_mtp_module", False)
        super().__init__(kvargs)

    def _init_custom(self):
        self._cos_cached = self.main_model._cos_cached
        self._sin_cached = self.main_model._sin_cached

    def _init_req_manager(self):
        # draft model shares the same req_manager with the main model
        if hasattr(self, "req_manager"):
            return
        create_max_seq_len = 0

        if self.batch_max_tokens is not None:
            create_max_seq_len = max(create_max_seq_len, self.batch_max_tokens)
        if self.max_seq_length is not None:
            create_max_seq_len = max(create_max_seq_len, self.max_seq_length)

        self.req_manager = ReqManager(self.max_req_num, create_max_seq_len, self.mem_manager)
        return

    def _init_mem_manager(self):
        self.mem_manager = Deepseek3MTPMemoryManager(
            self.max_total_token_num,
            dtype=self.data_type,
            head_num=1,
            head_dim=self.config["kv_lora_rank"] + self.config["qk_rope_head_dim"],
            layer_num=self.config["num_hidden_layers"],
            mem_fraction=self.mem_fraction,
        )
        return

    def _init_weights(self):
        super()._init_weights()
        self.pre_post_weight.wte_weight_ = self.main_model.pre_post_weight.wte_weight_
        self.pre_post_weight.lm_head_weight_ = self.main_model.pre_post_weight.lm_head_weight_
        self.pre_post_weight.final_norm_weight_ = self.main_model.pre_post_weight.final_norm_weight_
