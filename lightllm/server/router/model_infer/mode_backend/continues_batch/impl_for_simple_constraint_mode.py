import torch
from .impl import ContinuesBatchBackend
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.io_struct import FinishStatus
from lightllm.server.router.model_infer.infer_batch import InferBatch, InferReq, InferSamplingParams
from .pre_process import prepare_prefill_inputs, prepare_decode_inputs
from .post_process import sample
from lightllm.server.tokenizer import get_tokenizer
from typing import List


class SimpleConstraintBackend(ContinuesBatchBackend):
    def __init__(self) -> None:
        super().__init__()

    def init_custom(self):
        # import here, 当你不使用这个模式，缺少这些依赖也可以运行
        from outlines.models.transformers import TransformerTokenizer

        self.tokenizer = TransformerTokenizer(
            get_tokenizer(self.args.model_dir, self.args.tokenizer_mode, trust_remote_code=self.args.trust_remote_code)
        )
        self.tokenizer.eos_token_id = self.args.eos_id[0]
        return

    @calculate_time(show=False, min_cost_ms=300)
    def prefill_batch(self, batch_id):
        # import here, 当你不使用这个模式，缺少这些依赖也可以运行
        from outlines.fsm.guide import RegexGuide

        output_dict = {}
        batch: InferBatch = self.cache.pop(batch_id)
        kwargs, run_reqs = prepare_prefill_inputs(batch, self.radix_cache, self.model.mem_manager)
        run_reqs: List[InferReq] = run_reqs

        logics = self.model.forward(**kwargs)

        # 对于不能满足前缀匹配的logic位置，将其logics设置为一个较大负值，将其概率掩盖为 0
        mask = torch.ones_like(logics, dtype=torch.bool)
        for i, run_obj in enumerate(run_reqs):
            run_obj: InferReq = run_obj
            sample_params = run_obj.sampling_param
            if sample_params.regular_constraint is not None:
                sample_params.regex_guide = RegexGuide(sample_params.regular_constraint, self.tokenizer)
            self._mask_req_out_token(i, run_obj, mask)

        logics[mask] = -1000000.0

        next_token_ids, next_token_probs = sample(logics, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        for req_obj, next_token_id, next_token_logprob in zip(run_reqs, next_token_ids, next_token_logprobs):
            req_obj.cur_kv_len = len(req_obj.input_token_ids)
            req_obj.input_token_ids.append(next_token_id)
            req_obj.out_token_id_count[next_token_id] += 1
            req_obj.update_finish_status(self.eos_id)

            self._handle_req_ans(req_obj, next_token_id, next_token_logprob, output_dict)

        self.cache[batch.batch_id] = batch
        return output_dict

    @calculate_time(show=True, min_cost_ms=200)
    def decode_batch(self, batch_id):
        output_dict = {}
        batch: InferBatch = self.cache.pop(batch_id)
        kwargs, run_reqs = prepare_decode_inputs(batch, self.radix_cache)
        run_reqs: List[InferReq] = run_reqs

        logits = self.model.forward(**kwargs)

        all_has_no_constraint = all([e.sampling_param.regular_constraint is None for e in run_reqs])
        if not all_has_no_constraint:
            mask = torch.ones_like(logits, dtype=torch.bool)
            for i, run_obj in enumerate(run_reqs):
                self._mask_req_out_token(i, run_obj, mask)
            logits[mask] = -1000000.0

        next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        for req_obj, next_token_id, next_token_logprob in zip(run_reqs, next_token_ids, next_token_logprobs):
            req_obj: InferReq = req_obj
            req_obj.cur_kv_len = len(req_obj.input_token_ids)
            req_obj.input_token_ids.append(next_token_id)
            req_obj.out_token_id_count[next_token_id] += 1
            req_obj.update_finish_status(self.eos_id)

            self._handle_req_ans(req_obj, next_token_id, next_token_logprob, output_dict)

        self.cache[batch.batch_id] = batch
        return output_dict

    def _handle_req_ans(self, req_obj: InferReq, next_token_id, next_token_logprob, output_dict):
        next_token_id = int(next_token_id)
        if req_obj.sampling_param.regular_constraint is not None:
            sample_params = req_obj.sampling_param
            regex_guide = sample_params.regex_guide
            sample_params.fsm_current_state = regex_guide.get_next_state(sample_params.fsm_current_state, next_token_id)
            if sample_params.fsm_current_state == -1:
                req_obj.finish_status = FinishStatus.FINISHED_STOP

        metadata = {
            "id": next_token_id,
            "logprob": float(next_token_logprob),
        }
        output_dict[req_obj.r_id] = (
            req_obj.req_status,
            req_obj.cur_kv_len,
            req_obj.get_output_len(),
            [(next_token_id, metadata)],
            req_obj.finish_status.value,
            None,
        )
        return

    def _mask_req_out_token(self, i, run_obj: InferReq, mask):
        from outlines.fsm.guide import RegexGuide

        sample_params = run_obj.sampling_param
        if sample_params.regular_constraint is not None:
            regex_guide: RegexGuide = sample_params.regex_guide
            ok_token_id_list = regex_guide.get_next_instruction(sample_params.fsm_current_state).tokens
            mask[i, ok_token_id_list] = False
        else:
            mask[i, :] = False
        return
