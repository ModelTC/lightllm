import torch
import numpy as np
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.req_manager import ReqManager

class MistralInferStateInfo(LlamaInferStateInfo):
    def __init__(self):
        super().__init__()
        self.sliding_window = None
        self.b_start_loc_window = None
        self.b_att_seq_len = None
        self.b_att_start_loc = None
        self.total_cache_num = None
        # self.window_postion = None
