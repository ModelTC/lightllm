import torch
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.router.model_infer.infer_batch import InferBatch, InferReq, InferSamplingParams, requests_mapping
from lightllm.server.core.objs import ReqRunStatus, FinishStatus
from lightllm.utils.log_utils import init_logger
from .pre_process import prepare_prefill_inputs, prepare_decode_inputs
from .post_process import sample

logger = init_logger(__name__)


class ContinuesBatchBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()

    @calculate_time(show=False, min_cost_ms=300)
    def prefill_batch(self, batch_id):
        return self.forward(batch_id, is_prefill=True)

    @calculate_time(show=True, min_cost_ms=200)
    def decode_batch(self, batch_id):
        return self.forward(batch_id, is_prefill=False)

    def forward(self, batch_id, is_prefill):
        # special code for return all prompt_logprobs
        batch: InferBatch = self.cache.pop(batch_id)
        if is_prefill:
            kwargs, run_reqs = prepare_prefill_inputs(batch, self.radix_cache, self.is_multimodal)
        else:
            kwargs, run_reqs = prepare_decode_inputs(batch, self.radix_cache)

        logits = self.model.forward(**kwargs)

        # 对于后处理采样，只需要一个进程操作即可，其他进程只需要虚假的结果填充流程即可
        # 这样可以节省点电费吧。
        if self.tp_rank < self.dp_size:
            next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
        else:
            next_token_ids = [0 for _ in range(len(run_reqs))]
            next_token_logprobs = [0.0 for _ in range(len(run_reqs))]

        for req_obj, next_token_id, next_token_logprob in zip(run_reqs, next_token_ids, next_token_logprobs):
            # prefill and decode is same
            req_obj: InferReq = req_obj
            req_obj.cur_kv_len = req_obj.get_cur_total_len()
            # 只需要有真实采样的进程写入最后结果即可，由于其他进程没有做运算，所以其fake结果
            # 不能写入。
            if self.tp_rank < self.dp_size:
                req_obj.set_next_gen_token_id(next_token_id, next_token_logprob)
            req_obj.cur_output_len += 1

            req_obj.out_token_id_count[next_token_id] += 1
            req_obj.update_finish_status(self.eos_id)

            if self.tp_rank < self.dp_size:
                # shm_cur_kv_len shm_cur_output_len finish_status
                # 是router 调度进程需要读的信息
                # finish_token_index finish_status candetoken_out_len
                # 是 detokenization 进程需要的信息，注意这些变量的写入顺序
                # 避免异步协同问题。
                req_obj.shm_req.shm_cur_kv_len = req_obj.cur_kv_len
                req_obj.shm_req.shm_cur_output_len = req_obj.cur_output_len

                if req_obj.finish_status.is_finished():
                    req_obj.shm_req.finish_token_index = req_obj.get_cur_total_len() - 1
                    req_obj.shm_req.finish_status = req_obj.finish_status

                req_obj.shm_req.candetoken_out_len = req_obj.cur_output_len

        self.cache[batch.batch_id] = batch
        return
