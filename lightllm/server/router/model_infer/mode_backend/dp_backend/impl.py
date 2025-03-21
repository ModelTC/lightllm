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
                self.post_handel(run_reqs, next_token_ids, next_token_logprobs)
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
            self.post_handel(run_reqs, next_token_ids, next_token_logprobs)
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
            self.post_handel(run_reqs, next_token_ids, next_token_logprobs)
        if len(run_reqs1) != 0:
            next_token_ids1 = next_token_ids1.detach().cpu().numpy()
            next_token_logprobs1 = torch.log(next_token_probs1).detach().cpu().numpy()
            self.post_handel(run_reqs1, next_token_ids1, next_token_logprobs1)
        return

    def post_handel(self, run_reqs: List[InferReq], next_token_ids, next_token_logprobs):
        finished_req_ids = []

        for req_obj, next_token_id, next_token_logprob in zip(run_reqs, next_token_ids, next_token_logprobs):
            req_obj: InferReq = req_obj

            req_obj.cur_kv_len = len(req_obj.get_chuncked_input_token_ids())
            if req_obj.cur_kv_len < req_obj.get_cur_total_len():
                if self.is_master_in_dp:
                    req_obj.shm_req.shm_cur_kv_len = req_obj.cur_kv_len
                continue

            req_obj.set_next_gen_token_id(next_token_id, next_token_logprob)
            req_obj.cur_output_len += 1

            req_obj.out_token_id_count[next_token_id] += 1
            req_obj.update_finish_status(self.eos_id)

            if req_obj.finish_status.is_finished() or req_obj.shm_req.router_aborted:
                finished_req_ids.append(req_obj.shm_req.request_id)

            if self.is_master_in_dp:
                # shm_cur_kv_len shm_cur_output_len 是 router 调度进程需要读的信息
                # finish_token_index finish_status candetoken_out_len 是
                # detokenization 进程需要的信息，注意这些变量的写入顺序避免异步协同问题。
                req_obj.shm_req.shm_cur_kv_len = req_obj.cur_kv_len
                req_obj.shm_req.shm_cur_output_len = req_obj.cur_output_len

                if req_obj.finish_status.is_finished():
                    req_obj.shm_req.finish_token_index = req_obj.get_cur_total_len() - 1
                    req_obj.shm_req.finish_status = req_obj.finish_status

                req_obj.shm_req.candetoken_out_len = req_obj.cur_output_len

        g_infer_context.filter(finished_req_ids)
        return
