import torch
import torch.multiprocessing as mp
import threading
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.router.model_infer.infer_batch import InferBatch, InferReq, InferSamplingParams, requests_mapping
from lightllm.server.io_struct import ReqRunStatus, FinishStatus
from lightllm.server.pd_io_struct import UpKVStatus
from lightllm.utils.log_utils import init_logger
from ..pre_process import prepare_prefill_inputs, prepare_decode_inputs
from ..post_process import sample
from .up_status import UpStatusManager
from rpyc.utils.server import ThreadedServer

logger = init_logger(__name__)


class ContinuesBatchBackendForDecodeNode(ModeBackend):
    def __init__(self, info_queue: mp.Queue, mem_queue: mp.Queue) -> None:
        super().__init__()
        self.info_queue: mp.Queue = info_queue
        self.mem_queue: mp.Queue = mem_queue

    def init_custom(self):
        from .decode_infer_rpyc import PDDecodeInferRpcServer

        t = ThreadedServer(PDDecodeInferRpcServer(self), port=self.pd_rpyc_port, protocol_config={"allow_pickle": True})
        threading.Thread(target=lambda: t.start(), daemon=True).start()
        return

    # def add_batch(self, batch_id, reqs):
    #     ans = super().add_batch(batch_id, reqs)
    #     batch: InferBatch = self.cache[batch_id]
    #     for req_id in batch.request_ids:
    #         upkv_status = UpKVStatus(group_request_id=req_id)
    #         self.upkv_manager.put_status_task(upkv_status)
    #     return ans

    @calculate_time(show=False, min_cost_ms=300)
    def prefill_batch(self, batch_id):
        # ans = self.forward(batch_id, is_prefill=True)
        # decode 节点的 prefill 操作实际上就是啥也不操作，主要靠init batch的时候进行相关的
        # kv 的设置， 后续可以在这里做一些验证类型的操作。
        return {}

    @calculate_time(show=True, min_cost_ms=200)
    def decode_batch(self, batch_id):
        return self.forward(batch_id, is_prefill=False)

    def forward(self, batch_id, is_prefill):
        # special code for return all prompt_logprobs
        output_dict = {}
        batch: InferBatch = self.cache.pop(batch_id)
        if is_prefill:
            kwargs, run_reqs = prepare_prefill_inputs(batch, self.radix_cache, self.is_multimodal)
        else:
            kwargs, run_reqs = prepare_decode_inputs(batch, self.radix_cache)

        logits = self.model.forward(**kwargs)
        next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        for req_obj, next_token_id, next_token_logprob in zip(run_reqs, next_token_ids, next_token_logprobs):
            # prefill and decode is same
            req_obj: InferReq = req_obj
            req_obj.cur_kv_len = len(req_obj.input_token_ids)
            req_obj.input_token_ids.append(next_token_id)
            req_obj.out_token_id_count[next_token_id] += 1
            req_obj.update_finish_status(self.eos_id)

            metadata = {
                "id": int(next_token_id),
                "logprob": float(next_token_logprob),
            }
            output_dict[req_obj.r_id] = (
                req_obj.req_status,
                req_obj.cur_kv_len,
                req_obj.get_output_len(),
                [(int(next_token_id), metadata)],
                req_obj.finish_status.value,  # 转化为整数，避免传送大对象,
                None,
            )  # 请求状态， 当前占用的kv的长度， 当前输出token的数量， 输出的token的id和元信息列表， 是否推理结束的状态， 额外保留参数

        self.cache[batch.batch_id] = batch
        return output_dict
