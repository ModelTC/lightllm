import torch
import numpy as np
from lightllm.common.basemodel import InferStateInfo
from lightllm.common.req_manager import ReqManager


class Deepseek2InferStateInfo(InferStateInfo):
    def __init__(self):
        super().__init__()
