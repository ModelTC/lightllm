import cython
from typing import List, Optional, Callable
from ..infer_batch import InferReq, FinishStatus
from .base_backend import ModeBackend


def __update_finish_status(self: InferReq, gen_new_token_id:int, eos_ids: List[int]):
    # stop way 1
    for stop_token_ids in self.stop_sequences:
        stop_len = len(stop_token_ids)
        output_len = self.cur_output_len
        if stop_len > 0 and output_len >= stop_len:
            total_len = self.shm_req.input_len + output_len
            tail_token_ids = self.shm_req.shm_prompt_ids.arr[(total_len - stop_len) : total_len]
            if all(tail_token_ids[i] == stop_token_ids[i] for i in range(stop_len)):
                self.finish_status.set_status(FinishStatus.FINISHED_STOP)
                return
    
    # stop way 2
    shm_param = self.sampling_param.shm_param
    if (self.cur_output_len > 0 
        and shm_param.ignore_eos is False
        and gen_new_token_id in eos_ids
    ):
        self.finish_status.set_status(FinishStatus.FINISHED_STOP)
        return
    
    # stop way 3
    if self.cur_output_len >= shm_param.max_new_tokens:
        self.finish_status.set_status(FinishStatus.FINISHED_LENGTH)
        return


# @cython.boundcheck(False)
# @cython.wraparound(False)
def fast_post_handle(
    self: ModeBackend,
    run_reqs: List[InferReq],
    next_token_ids_,
    next_token_logprobs_,
    is_chuncked_mode: bool,
    do_filter_finished_reqs: bool,
    extra_post_req_handle_func: Optional[Callable[[InferReq, int, float], None]] = None,
) -> List[int]:
    """
    extra_post_req_handle_func 用于提供在一个请求确定输出的时候，给出额外的后处理操作，主要是用于
    约束输出等模式，设置自己请求内部的状态机的状态，并添加额外的停止判定条件等。
    """
    from lightllm.server.router.model_infer.infer_batch import g_infer_context
    
    finished_req_ids = [0 for _ in range(len(run_reqs))]
    finished_req_ids.clear()
    next_token_ids: cython.longlong[:] = cython.declare(cython.longlong[:], next_token_ids_)
    next_token_logprobs: cython.float[:] = cython.declare(cython.float[:], next_token_logprobs_)
    is_master_in_dp : cython.bint = self.is_master_in_dp
    is_chuncked_mode : cython.bint = is_chuncked_mode
    
    i : cython.Py_ssize_t
    for i in range(len(run_reqs)):
        req_obj: InferReq = run_reqs[i]
        shm_req = req_obj.shm_req
        next_token_id: cython.int = next_token_ids[i]
        next_token_logprob: cython.float = next_token_logprobs[i]
        cur_total_len = shm_req.input_len + req_obj.cur_output_len

        if is_chuncked_mode:
            new_kv_len = min(cur_total_len, req_obj.cur_kv_len + shm_req.chunked_prefill_size)
        else:
            new_kv_len = cur_total_len

        req_obj.cur_kv_len = new_kv_len
        if is_master_in_dp:
            shm_req.shm_cur_kv_len = req_obj.cur_kv_len

        # 这个地方主要是为了提前判断是否存在abort的情况，如果abort了
        # 直接将请求放入finished 处理队列中。
        if shm_req.router_aborted:
            finished_req_ids.append(shm_req.request_id)
            continue

        # 对于没有到达需要输出 token 阶段的请求，直接略过
        if req_obj.cur_kv_len < cur_total_len:
            continue

        # 将生成的下一个token的信息写入到管理对象中。
        gen_token_index = cur_total_len
        shm_req.shm_prompt_ids.arr[gen_token_index] = next_token_id
        shm_req.shm_logprobs.arr[gen_token_index] = next_token_logprob
        req_obj.cur_output_len += 1

        req_obj.out_token_id_count[next_token_id] += 1
        __update_finish_status(req_obj, next_token_id, self.eos_id)

        if extra_post_req_handle_func is not None:
            extra_post_req_handle_func(req_obj, next_token_id, next_token_logprob)

        # 判断是否已经满足生成结束条件。
        is_finished = req_obj.finish_status.is_finished()
        if is_finished or shm_req.router_aborted:
            finished_req_ids.append(shm_req.request_id)

        if is_master_in_dp:
            # shm_cur_kv_len shm_cur_output_len 是 router 调度进程需要读的信息
            # finish_token_index finish_status candetoken_out_len 是
            # detokenization 进程需要的信息，注意这些变量的写入顺序避免异步协同问题。
            shm_req.shm_cur_output_len = req_obj.cur_output_len

            if is_finished:
                shm_req.finish_token_index = gen_token_index
                shm_req.finish_status = req_obj.finish_status

            shm_req.candetoken_out_len = req_obj.cur_output_len

    if do_filter_finished_reqs:
        g_infer_context.filter(finished_req_ids)
    
    return finished_req_ids