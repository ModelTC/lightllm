import torch.multiprocessing as mp
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.utils.log_utils import init_logger
from .prefill_impl_for_dp_chuncked import DPChunkedForPrefillNode
from ....dp_backend.impl_mtp import DPChunkedPrefillWithMTPBackend

logger = init_logger(__name__)


class DPChunkedForMtpPrefillNode(DPChunkedForPrefillNode):
    def __init__(self, info_queue: mp.Queue, mem_queue: mp.Queue) -> None:
        super().__init__(info_queue=info_queue, mem_queue=mem_queue)
        return

    def init_model(self, kvargs):
        super().init_model(kvargs)
        DPChunkedPrefillWithMTPBackend._init_mtp_draft_model(self, kvargs)
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
        dp_prefill_req_nums, max_prefill_num = self._dp_all_gather_prefill_req_num(prefill_reqs=prefill_reqs)
        if self.chunked_prefill_state.dp_need_prefill(prefill_reqs, decode_reqs, dp_prefill_req_nums, max_prefill_num):
            if not self.enable_prefill_microbatch_overlap:
                DPChunkedPrefillWithMTPBackend.normal_mtp_prefill_reqs(
                    self,
                    prefill_reqs=prefill_reqs,
                    max_prefill_num=max_prefill_num,
                    uninit_reqs=uninit_reqs,
                    ok_finished_reqs=ok_finished_reqs,
                )
            else:
                DPChunkedPrefillWithMTPBackend.overlap_mtp_prefill_reqs(
                    self,
                    prefill_reqs=prefill_reqs,
                    max_prefill_num=max_prefill_num,
                    uninit_reqs=uninit_reqs,
                    ok_finished_reqs=ok_finished_reqs,
                )

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        return
