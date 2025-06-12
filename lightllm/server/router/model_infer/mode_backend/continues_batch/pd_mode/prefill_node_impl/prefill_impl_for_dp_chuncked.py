import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from typing import List, Tuple
from lightllm.server.router.model_infer.infer_batch import InferReq, InferSamplingParams, g_infer_context
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_unique_server_name, get_env_start_args
from .prefill_impl import ChunckedPrefillForPrefillNode
from lightllm.server.router.model_infer.mode_backend.dp_backend.impl import DPChunkedPrefillBackend

logger = init_logger(__name__)


class DPChunkedForPrefillNode(ChunckedPrefillForPrefillNode):
    def __init__(self, info_queue: mp.Queue, mem_queue: mp.Queue) -> None:
        super().__init__(info_queue=info_queue, mem_queue=mem_queue)
        self.enable_prefill_microbatch_overlap = get_env_start_args().enable_prefill_microbatch_overlap

    def init_custom(self):
        super().init_custom()
        self.reduce_tensor = torch.tensor([0], dtype=torch.int32, device="cuda", requires_grad=False)
        return

    def prefill(self, reqs: List[Tuple]):
        self._init_reqs(reqs)
        return

    def decode(self):
        uninit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs = self._get_classed_reqs(
            g_infer_context.infer_req_ids,
            no_decode=True,
        )
        assert len(uninit_reqs) == 0
        assert len(decode_reqs) == 0

        self._filter_reqs(aborted_reqs)

        if ok_finished_reqs:
            self.prefill_req_frozen_tokens_and_put_to_kvmove_taskqueue(ok_finished_reqs)
            self._filter_reqs(ok_finished_reqs)
            ok_finished_reqs.clear()

        # 进行 chuncked prefill
        current_dp_prefill_num = len(prefill_reqs)
        self.reduce_tensor.fill_(current_dp_prefill_num)
        dist.all_reduce(self.reduce_tensor, op=dist.ReduceOp.MAX, group=None, async_op=False)
        max_prefill_num = self.reduce_tensor.item()
        if max_prefill_num != 0:
            if not self.enable_prefill_microbatch_overlap:
                self.normal_prefill_reqs(prefill_reqs, max_prefill_num, uninit_reqs, ok_finished_reqs)
            else:
                self.overlap_prefill_reqs(prefill_reqs, max_prefill_num, uninit_reqs, ok_finished_reqs)

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        return

    def normal_prefill_reqs(self, prefill_reqs: List[InferReq], max_prefill_num: int, uninit_reqs, ok_finished_reqs):
        DPChunkedPrefillBackend.normal_prefill_reqs(
            self,
            prefill_reqs=prefill_reqs,
            max_prefill_num=max_prefill_num,
            uninit_reqs=uninit_reqs,
            ok_finished_reqs=ok_finished_reqs,
        )
        return

    def overlap_prefill_reqs(self, prefill_reqs: List[InferReq], max_prefill_num: int, uninit_reqs, ok_finished_reqs):
        DPChunkedPrefillBackend.overlap_prefill_reqs(
            self,
            prefill_reqs=prefill_reqs,
            max_prefill_num=max_prefill_num,
            uninit_reqs=uninit_reqs,
            ok_finished_reqs=ok_finished_reqs,
        )
        return
