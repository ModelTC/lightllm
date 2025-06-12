import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from typing import List, Tuple
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.model_infer.mode_backend import padded_prepare_prefill_inputs
from lightllm.utils.envs_utils import get_unique_server_name, get_env_start_args
from .decode_impl import ContinuesBatchBackendForDecodeNode
from lightllm.server.router.model_infer.mode_backend.dp_backend.impl import DPChunkedPrefillBackend

logger = init_logger(__name__)


class DPForDecodeNode(ContinuesBatchBackendForDecodeNode):
    def __init__(self, info_queue: mp.Queue, mem_queue: mp.Queue) -> None:
        super().__init__(info_queue, mem_queue)
        self.enable_decode_microbatch_overlap = get_env_start_args().enable_decode_microbatch_overlap
        return

    def init_custom(self):
        super().init_custom()
        self.reduce_tensor = torch.tensor([0], dtype=torch.int32, device="cuda", requires_grad=False)
        # 这个地方预先进行一次 prefill 推理，主要是为了填充后续fake请求的第一个token位置，因为填充的decode请求
        # 在推理的时候至少是两个token，1个是已经有kv的token，一个是等待计算kv的token，然后生成第三个token，这几个
        # token 实际引用的都是 g_infer_context.req_manager.mem_manager.HOLD_TOKEN_MEMINDEX，但是需要初始化排除
        # nan 值，避免后续构建的fake请求在计算的过程中出现计算错误。
        model_input, run_reqs, padded_req_num = padded_prepare_prefill_inputs([], is_multimodal=self.is_multimodal)
        self.model.forward(model_input)
        assert len(run_reqs) == 0 and padded_req_num == 1

        return

    def prefill(self, reqs: List[Tuple]):
        self._init_reqs(reqs, init_req_obj=False)
        return

    def decode(self):
        uninit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs = self._get_classed_reqs(
            g_infer_context.infer_req_ids
        )
        assert len(prefill_reqs) == 0

        self._filter_reqs(aborted_reqs)

        self.reduce_tensor.fill_(len(decode_reqs))
        dist.all_reduce(self.reduce_tensor, op=dist.ReduceOp.MAX, group=None, async_op=False)
        max_decode_num = self.reduce_tensor.item()
        if max_decode_num != 0:
            if not self.enable_decode_microbatch_overlap:
                self.normal_decode(decode_reqs, max_decode_num, uninit_reqs, ok_finished_reqs)
            else:
                self.overlap_decode(decode_reqs, max_decode_num, uninit_reqs, ok_finished_reqs)

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        return

    def normal_decode(self, decode_reqs: List[InferReq], max_decode_num: int, uninit_reqs, ok_finished_reqs):
        DPChunkedPrefillBackend.normal_decode(
            self,
            decode_reqs=decode_reqs,
            max_decode_num=max_decode_num,
            uninit_reqs=uninit_reqs,
            ok_finished_reqs=ok_finished_reqs,
        )
        return

    def overlap_decode(self, decode_reqs: List[InferReq], max_decode_num: int, uninit_reqs, ok_finished_reqs):
        DPChunkedPrefillBackend.overlap_decode(
            self,
            decode_reqs=decode_reqs,
            max_decode_num=max_decode_num,
            uninit_reqs=uninit_reqs,
            ok_finished_reqs=ok_finished_reqs,
        )
        return
