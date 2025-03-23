import torch
import torch.distributed as dist
import numpy as np
from typing import List, Tuple
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq, InferSamplingParams
from lightllm.server.core.objs import FinishStatus
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.model_infer.mode_backend.continues_batch.post_process import sample


class DPChunkedPrefillBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()
        """
        DPChunkedPrefillBackend 是DP的chuncked prefill 的单机实现，并不是标准的chunked prefill
        实现，这个模式最佳使用方式需要和 PD 分离模式进行配合。
        """
        self.is_overlap_decode_mode = False
        pass

    def init_custom(self):
        self.reduce_tensor = torch.tensor([0], dtype=torch.int32, device="cuda", requires_grad=False)
        # 这个地方预先进行一次 prefill 推理，主要是为了填充后续fake请求的第一个token位置，因为填充的decode请求
        # 在推理的时候至少是两个token，1个是已经有kv的token，一个是等待计算kv的token，然后生成第三个token，这几个
        # token 实际引用的都是 g_infer_context.req_manager.mem_manager.HOLD_TOKEN_MEMINDEX，但是需要初始化排除
        # nan 值，避免后续构建的fake请求在计算的过程中出现计算错误。
        from .pre_process import padded_prepare_prefill_inputs

        kwargs, run_reqs, padded_req_num = padded_prepare_prefill_inputs([], 1, is_multimodal=False)
        self.model.forward(**kwargs)
        assert len(run_reqs) == 0 and padded_req_num == 1
        return

    def prefill(self, reqs: List[Tuple]):
        self._init_reqs(reqs)
        return

    def decode(self):
        from .pre_process import get_classed_reqs

        uinit_reqs, finished_reqs, prefill_reqs, decode_reqs = get_classed_reqs(g_infer_context.infer_req_ids)
        current_dp_prefill_num = len(prefill_reqs)
        self.reduce_tensor.fill_(current_dp_prefill_num)
        dist.all_reduce(self.reduce_tensor, op=dist.ReduceOp.MAX, group=None, async_op=False)
        max_prefill_num = self.reduce_tensor.item()
        if max_prefill_num != 0:
            from .pre_process import padded_prepare_prefill_inputs

            kwargs, run_reqs, padded_req_num = padded_prepare_prefill_inputs(
                prefill_reqs, max_prefill_num, is_multimodal=False
            )
            logits = self.model.forward(**kwargs)
            if len(run_reqs) != 0:
                logits = logits[0 : len(run_reqs), :]
                next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
                next_token_ids = next_token_ids.detach().cpu().numpy()
                next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
                self._post_handel(
                    run_reqs, next_token_ids, next_token_logprobs, is_chuncked_mode=True, do_filter_finished_reqs=True
                )
            logits = None

        self.reduce_tensor.fill_(len(decode_reqs))
        dist.all_reduce(self.reduce_tensor, op=dist.ReduceOp.MAX, group=None, async_op=False)
        max_decode_num = self.reduce_tensor.item()
        if max_decode_num != 0:
            if not self.is_overlap_decode_mode:
                self.normal_decode(decode_reqs, max_decode_num)
            else:
                self.overlap_decode(decode_reqs, max_decode_num)
        return

    def normal_decode(self, decode_reqs: List[InferReq], max_decode_num: int):
        from .pre_process import padded_prepare_decode_inputs

        kwargs, run_reqs, padded_req_num = padded_prepare_decode_inputs(
            decode_reqs, max_decode_num, is_multimodal=False
        )
        logits = self.model.forward(**kwargs)
        if len(run_reqs) != 0:
            logits = logits[0 : len(run_reqs), :]
            next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
            self._post_handel(
                run_reqs, next_token_ids, next_token_logprobs, is_chuncked_mode=True, do_filter_finished_reqs=True
            )
        logits = None

    def overlap_decode(self, decode_reqs: List[InferReq], max_decode_num: int):
        from .pre_process import padded_overlap_prepare_decode_inputs

        (
            micro_batch,
            run_reqs,
            padded_req_num,
            micro_batch1,
            run_reqs1,
            padded_req_num1,
        ) = padded_overlap_prepare_decode_inputs(decode_reqs, max_decode_num, is_multimodal=False)
        logits, logits1 = self.model.microbatch_overlap_decode(micro_batch, micro_batch1)
        if len(run_reqs) != 0:
            logits = logits[0 : len(run_reqs), :]
            next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        if len(run_reqs1) != 0:
            logits1 = logits1[0 : len(run_reqs1), :]
            next_token_ids1, next_token_probs1 = sample(logits1, run_reqs1, self.eos_id)

        if len(run_reqs) != 0:
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
            self._post_handel(
                run_reqs, next_token_ids, next_token_logprobs, is_chuncked_mode=True, do_filter_finished_reqs=True
            )
        if len(run_reqs1) != 0:
            next_token_ids1 = next_token_ids1.detach().cpu().numpy()
            next_token_logprobs1 = torch.log(next_token_probs1).detach().cpu().numpy()
            self._post_handel(
                run_reqs1, next_token_ids1, next_token_logprobs1, is_chuncked_mode=True, do_filter_finished_reqs=True
            )
        return
