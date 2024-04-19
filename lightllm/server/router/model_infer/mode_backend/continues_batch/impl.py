import torch
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.router.model_infer.infer_batch import InferBatch, InferReq, InferSamplingParams, requests_mapping
from lightllm.server.io_struct import ReqRunStatus, FinishStatus
from lightllm.utils.log_utils import init_logger
from .pre_process import prepare_prefill_inputs, prepare_decode_inputs
from .post_process import sample


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
        probs = torch.softmax(logits, dim=-1)
        logprobs = torch.log2(probs + 1e-6)
        shang = -torch.sum(probs * logprobs, dim=-1, keepdim=False)
        
        select_index = torch.argmax(probs, dim=1, keepdim=False)
        next_token_ids = select_index
        next_token_probs = torch.gather(probs, dim=1, index=select_index.view(-1, 1)).view(-1)

        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_probs = next_token_probs.detach().cpu().numpy()

        for req_obj, next_token_id, next_token_prob in zip(run_reqs, next_token_ids, next_token_probs):
            # prefill and decode is same
            req_obj: InferReq = req_obj
            req_obj.cur_kv_len = len(req_obj.input_token_ids)
            if next_token_prob >= req_obj.sampling_param.low_prob or req_obj.recompute == 3:
                if req_obj.recompute == 3:
                    print(f"recompute 3 {next_token_prob}")
                print(f"ok get {next_token_prob}")
                if is_prefill:
                    req_obj.position_loc = len(req_obj.input_token_ids)
                else:
                    req_obj.position_loc += 1
            
                req_obj.input_token_ids.append(next_token_id)
                req_obj.out_token_id_count[next_token_id] += 1
                req_obj.update_finish_status(self.eos_id)
                req_obj.recompute = 0


                metadata = {
                    "id": int(next_token_id),
                    "logprob": float(next_token_prob),
                }
                output_dict[req_obj.r_id] = (
                    req_obj.req_status,
                    req_obj.cur_kv_len,
                    req_obj.get_output_len(),
                    [(int(next_token_id), metadata)],
                    req_obj.finish_status.value,  # 转化为整数，避免传送大对象,
                    None,
                )  # 请求状态， 当前占用的kv的长度， 当前输出token的数量， 输出的token的id和元信息列表， 是否推理结束的状态， 额外保留参数
            else:
                print(f"not ok get {next_token_prob}")
                if is_prefill:
                    req_obj.position_loc = len(req_obj.input_token_ids) - 1
                
                req_obj.input_token_ids.append(req_obj.input_token_ids[-1])
                req_obj.out_token_id_count[req_obj.input_token_ids[-1]] += 1
                # req_obj.update_finish_status(self.eos_id)
                req_obj.recompute += 1
                req_obj.update_finish_status(self.eos_id)
    

                # metadata = {
                #     "id": int(next_token_id),
                #     "logprob": float(next_token_prob),
                # }
                if not req_obj.finish_status.is_finished():
                    output_dict[req_obj.r_id] = (
                        req_obj.req_status,
                        req_obj.cur_kv_len,
                        req_obj.get_output_len(),
                        [],
                        req_obj.finish_status.value,  # 转化为整数，避免传送大对象,
                        None,
                    )  # 请求状态， 当前占用的kv的长度， 当前输出token的数量， 输出的token的id和元信息列表， 是否推理结束的状态， 额外保留参数
                else:
                    metadata = {
                        "id": self.eos_id[0],
                        "logprob": float(1.0),
                    }
                    output_dict[req_obj.r_id] = (
                        req_obj.req_status,
                        req_obj.cur_kv_len,
                        req_obj.get_output_len(),
                        [(self.eos_id[0], metadata)],
                        req_obj.finish_status.value,  # 转化为整数，避免传送大对象,
                        None,
                    )  # 请求状态， 当前占用的kv的长度， 当前输出token的数量， 输出的token的id和元信息列表， 是否推理结束的状态， 额外保留参数



        self.cache[batch.batch_id] = batch
        return output_dict
