import torch
from typing import List, Tuple
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.router.model_infer.infer_batch import InferReq, InferSamplingParams
from lightllm.server.core.objs import FinishStatus
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from .pre_process import prepare_prefill_inputs, prepare_decode_inputs
from .post_process import sample
from lightllm.utils.dist_utils import get_current_device_id

logger = init_logger(__name__)

def split_kwargs(
    batch_size,
    total_token_num,
    max_len_in_batch,
    input_ids: torch.Tensor,
    mem_indexes: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_ready_cache_len: torch.Tensor = None,
    multimodal_params=None,
    is_prefill=True):
    half_batch = batch_size // 2
    b_req_idx1 = b_req_idx[:half_batch]
    b_req_idx2 = b_req_idx[half_batch:]
    b_seq_len1 = b_seq_len[:half_batch]
    b_seq_len2 = b_seq_len[half_batch:]
    half_tokens = b_seq_len1.sum().item()
    if is_prefill:
        input_ids1 = input_ids[:half_tokens]
        input_ids2 = input_ids[half_tokens:]
        mem_indexes1 = mem_indexes[:half_tokens]
        mem_indexes2 = mem_indexes[half_tokens:]
    else:
        input_ids1 = input_ids[:half_batch]
        input_ids2 = input_ids[half_batch:]
        mem_indexes1 = mem_indexes[:half_batch]
        mem_indexes2 = mem_indexes[half_batch:]
    max_len_in_batch1 = b_seq_len1.max().item()
    max_len_in_batch2 = b_seq_len2.max().item()
    b_start_loc1 = b_seq_len1.cumsum(dim=0) - b_seq_len1
    b_start_loc2 = b_seq_len2.cumsum(dim=0) - b_seq_len2
    b_ready_cache_len1 = b_ready_cache_len[:half_batch] if b_ready_cache_len is not None else None
    b_ready_cache_len2 = b_ready_cache_len[half_batch:] if b_ready_cache_len is not None else None
    kwargs1 = {
        "batch_size": half_batch,
        "total_token_num": half_tokens,
        "max_len_in_batch": max_len_in_batch1,
        "input_ids": input_ids1,
        "mem_indexes": mem_indexes1,
        "b_req_idx": b_req_idx1,
        "b_start_loc": b_start_loc1,
        "b_seq_len": b_seq_len1,
        "b_ready_cache_len": b_ready_cache_len1,
        "is_prefill": is_prefill,
        "stream_id": 0,
    }
    kwargs2 = {
        "batch_size": batch_size - batch_size // 2,
        "total_token_num": total_token_num - half_tokens,
        "max_len_in_batch": max_len_in_batch2,
        "input_ids": input_ids2,
        "mem_indexes": mem_indexes2,
        "b_req_idx": b_req_idx2,
        "b_start_loc": b_start_loc2,
        "b_seq_len": b_seq_len2,
        "b_ready_cache_len": b_ready_cache_len2,
        "is_prefill": is_prefill,
        "stream_id": 1,
    }
    if multimodal_params is not None:
        kwargs1["multimodal_params"] = multimodal_params
        kwargs2["multimodal_params"] = multimodal_params
    return kwargs1, kwargs2

def split_kwargs_n(
    batch_size,
    total_token_num,
    max_len_in_batch,
    input_ids: torch.Tensor,
    mem_indexes: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_ready_cache_len: torch.Tensor = None,
    multimodal_params=None,
    is_prefill=True,
    split_n=2):

    kwargs = [None] * split_n

    # 计算每个分片的批次大小
    batch_per_split = [batch_size // split_n] * split_n
    # 处理不能整除的情况
    for i in range(batch_size % split_n):
        batch_per_split[i] += 1

    # 准备分割索引
    batch_indices = [0]
    for size in batch_per_split:
        batch_indices.append(batch_indices[-1] + size)

    # 记录到目前为止的token数
    cumulative_tokens = 0

    # 为每个分片创建kwargs
    for i in range(split_n):
        start_idx = batch_indices[i]
        end_idx = batch_indices[i+1]

        # 分割批次相关的张量
        split_b_req_idx = b_req_idx[start_idx:end_idx]
        split_b_seq_len = b_seq_len[start_idx:end_idx]

        # 计算该分片的token数量
        split_tokens = split_b_seq_len.sum().item()

        if is_prefill:
            # 在prefill阶段，根据token分割
            token_start = cumulative_tokens
            token_end = token_start + split_tokens
            split_input_ids = input_ids[token_start:token_end]
            split_mem_indexes = mem_indexes[token_start:token_end]
        else:
            # 在decode阶段，根据批次分割
            split_input_ids = input_ids[start_idx:end_idx]
            split_mem_indexes = mem_indexes[start_idx:end_idx]

        # 计算此分片的其他参数
        split_max_len = split_b_seq_len.max().item() if len(split_b_seq_len) > 0 else 0
        split_b_start_loc = split_b_seq_len.cumsum(dim=0) - split_b_seq_len

        # 处理缓存长度
        split_b_ready_cache_len = None
        if b_ready_cache_len is not None:
            split_b_ready_cache_len = b_ready_cache_len[start_idx:end_idx]

        # 创建kwargs字典
        kwargs[i] = {
            "batch_size": len(split_b_req_idx),
            "total_token_num": split_tokens,
            "max_len_in_batch": split_max_len,
            "input_ids": split_input_ids,
            "mem_indexes": split_mem_indexes,
            "b_req_idx": split_b_req_idx,
            "b_start_loc": split_b_start_loc,
            "b_seq_len": split_b_seq_len,
            "b_ready_cache_len": split_b_ready_cache_len,
            "is_prefill": is_prefill,
            "stream_id": i,
        }

        # 如果有多模态参数，添加到kwargs
        if multimodal_params is not None:
            kwargs[i]["multimodal_params"] = multimodal_params

        # 更新累计token数
        cumulative_tokens += split_tokens

    return kwargs

class ContinuesBatchBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()

    def prefill(self, reqs: List[Tuple], stream_id):
        req_ids = self._init_reqs(reqs)
        kwargs, run_reqs = prepare_prefill_inputs(req_ids, self.is_multimodal)
        kwargs["stream_id"] = stream_id
        with torch.cuda.stream(self.model.stream[stream_id]):
            logits = self.model.forward(**kwargs)
            next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
            torch.cuda.current_stream().synchronize()

        # logits = self.model.forward(**kwargs)

        # split_n = self.model.stream_num
        # if kwargs["batch_size"] > split_n - 1:
        #     kwargs_list = split_kwargs_n(**kwargs, split_n=split_n)
        #     logits = [None] * split_n
        #     for i in range(split_n):
        #         with torch.cuda.stream(self.model.stream[i]):
        #             logits[i] = self.model.forward(**kwargs_list[i])

        #     torch.cuda.synchronize()
        #     logits = torch.cat(logits, dim=0)
        # else:
        #     logits = self.model.forward(**kwargs)

        # next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        self.post_handel(run_reqs, next_token_ids, next_token_logprobs)
        return

    def decode(self, stream_id):
        kwargs, run_reqs = prepare_decode_inputs(g_infer_context.infer_req_ids)
        kwargs["stream_id"] = stream_id
        with torch.cuda.stream(self.model.stream[stream_id]):
            logits = self.model.forward(**kwargs)
            next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
            torch.cuda.current_stream().synchronize()

        # logits = self.model.forward(**kwargs)

        # split_n = self.model.stream_num
        # if kwargs["batch_size"] > split_n - 1:
        #     kwargs_list = split_kwargs_n(**kwargs, split_n=split_n)
        #     logits = [None] * split_n
        #     for i in range(split_n):
        #         with torch.cuda.stream(self.model.stream[i]):
        #             logits[i] = self.model.forward(**kwargs_list[i])

        #     torch.cuda.synchronize()
        #     logits = torch.cat(logits, dim=0)
        # else:
        #     logits = self.model.forward(**kwargs)

        # next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        self.post_handel(run_reqs, next_token_ids, next_token_logprobs)
        return

    def post_handel(self, run_reqs: List[InferReq], next_token_ids, next_token_logprobs):
        finished_req_ids = []

        for req_obj, next_token_id, next_token_logprob in zip(run_reqs, next_token_ids, next_token_logprobs):
            req_obj: InferReq = req_obj
            req_obj.cur_kv_len = req_obj.get_cur_total_len()

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
