import torch
import torch.distributed as dist
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.router.model_infer.infer_batch import InferBatch, InferReq, InferSamplingParams, requests_mapping
from lightllm.server.io_struct import ReqRunStatus, FinishStatus
from lightllm.utils.log_utils import init_logger
from .pre_process import prepare_prefill_inputs, prepare_decode_inputs
from .post_process import sample
from .gather_scatter import scatter_kvs_to_idx

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

        self.model.mem_manager.update()
        self.cache[batch.batch_id] = batch
        return output_dict

    def recv_request(self, req_info_list, target_rank):
        info_buf = torch.empty(2, dtype=torch.int32, device="cuda")
        dist.recv(info_buf, src=target_rank)
        reqs_num = info_buf[0]
        total_len = info_buf[1]
        req_lengths = torch.empty(reqs_num, dtype=torch.int32, device="cuda")
        dist.recv(req_lengths, src=target_rank)
        alloc_idx = self.model.mem_manager.alloc(total_len.item()).to(dtype=torch.int32)
        gather_kvs = torch.empty(
            (total_len.item(), 2 * self.model.mem_manager.head_num * self.model.mem_manager.head_dim),
            dtype=self.model.mem_manager.dtype,
            device="cuda",
        )
        for i in range(self.model.mem_manager.layer_num):
            dist.recv(gather_kvs, src=target_rank)
            scatter_kvs_to_idx(
                self.model.mem_manager.kv_buffer[i].view(self.model.mem_manager.kv_buffer[i].shape[0], -1),
                gather_kvs,
                alloc_idx,
            )
        last_tokens = torch.empty((reqs_num,), dtype=torch.int32, device="cuda")
        dist.recv(last_tokens, src=target_rank)
        rids = self.model.req_manager.add_reqs(req_lengths, alloc_idx)

        for req_info, req_idx, next_token_id in zip(req_info_list, rids, last_tokens):
            prompt_ids = req_info["input_id"]
            req_id = req_info["request_id"]
            group_req_id = req_info["group_req_id"]
            cur_kv_len = len(prompt_ids)  # just done prefill
            tokenized_input = prompt_ids
            input_length = len(tokenized_input)
            if req_info["req_status"] == "ReqRunStatus.PAUSED_AND_OFFLOAD":
                req_status = ReqRunStatus.PAUSED_AND_OFFLOAD
            else:
                raise ValueError(f"Invalid req_status: {req_info['req_status']}")
            r_obj = InferReq(
                req_id,
                group_req_id,
                input_token_ids=prompt_ids,
                sampling_param=InferSamplingParams(**req_info["sampling_param"]),
                multimodal_params=MultimodalParams(**req_info["multimodal_params"]),
                req_idx=req_idx,
                prompt_len=input_length,
                req_status=req_status,
            )
            r_obj.cur_kv_len = cur_kv_len
            r_obj.input_token_ids.append(next_token_id.item())
            r_obj.out_token_id_count[next_token_id] += 1
            assert req_id not in requests_mapping
            requests_mapping[req_id] = r_obj
        return rids

    def send_request(self, batch_id, target_rank):
        batch: InferBatch = self.cache[batch_id]
        req_ids = [rid for rid in batch.request_ids]
        free_req_index = [requests_mapping[rid].req_idx for rid in req_ids]
        req_idx = [requests_mapping[rid].req_idx for rid in req_ids]
        prefill_tokens = []
        context_lengths = [requests_mapping[rid].prompt_len for rid in req_ids]
        for rid, kvlen in zip(req_idx, context_lengths):
            tokens = self.model.req_manager.req_to_token_indexs[rid][:kvlen]
            prefill_tokens.append(tokens)
        prefill_tokens = torch.cat(prefill_tokens, dim=0)

        def callback():
            self.model.req_manager.free_reqs(free_req_index)

        def get_last_new_tokens():
            return [requests_mapping[rid].input_token_ids[-1] for rid in req_ids]

        info_buf = torch.empty(2, dtype=torch.int32, device="cuda")
        info_buf[0] = len(context_lengths)
        info_buf[1] = prefill_tokens.shape[0]
        dist.send(info_buf, dst=target_rank)
        context_lengths = torch.tensor(context_lengths, dtype=torch.int32, device="cuda")
        dist.send(context_lengths, dst=target_rank)
        self.model.mem_manager.commit(batch_id, target_rank, callback, get_last_new_tokens)
