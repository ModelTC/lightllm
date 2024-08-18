import torch
from .impl import ContinuesBatchBackend
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.router.model_infer.infer_batch import InferBatch, InferReq, InferSamplingParams
from .pre_process import prepare_prefill_inputs
from .post_process import sample
from lightllm.server.tokenizer import get_tokenizer


class TokenHealingBackend(ContinuesBatchBackend):
    def __init__(self) -> None:
        super().__init__()

    def init_custom(self):
        """
        初始化tokenizer 词表相关的的操作
        """
        self.tokenizer = get_tokenizer(
            self.args.model_dir, self.args.tokenizer_mode, trust_remote_code=self.args.trust_remote_code
        )
        vob_dict = self.tokenizer.get_vocab()
        assert len(vob_dict) == self.model.vocab_size
        self.max_token_str_len = max([len(key) for key in vob_dict.keys()])
        self.logger.info(f"max vob token str len: {self.max_token_str_len}")
        self.pad_token_str = "\U0010FFFF" * self.max_token_str_len
        from sortedcontainers import SortedList

        self.token_id_to_token = {token_id: token for token, token_id in vob_dict.items()}
        self.sorted_tokens = SortedList(
            [(token_str, token_id) for token_str, token_id in vob_dict.items()], key=lambda x: x[0]
        )
        self.token_indexes = torch.tensor([e[1] for e in self.sorted_tokens], dtype=torch.int64, device="cuda")
        return

    @calculate_time(show=False, min_cost_ms=300)
    def prefill_batch(self, batch_id):
        # 在 token_healing 的模式下，暂时不能启用 dynamic prompt cache
        assert self.radix_cache is None
        output_dict = {}
        batch: InferBatch = self.cache.pop(batch_id)
        kwargs, run_reqs = prepare_prefill_inputs(batch, self.radix_cache, self.model.mem_manager)

        all_logics = self.model.forward(**kwargs)

        b_seq_len = kwargs["b_seq_len"]
        b_ready_cache_len = kwargs["b_ready_cache_len"]
        b_cur_len_numpy = (b_seq_len - b_ready_cache_len).cpu().numpy()

        # 对于不能满足前缀匹配的logic位置，将其logics设置为一个较大负值，将其概率掩盖为 0
        mask = torch.ones_like(all_logics, dtype=torch.bool)

        for i, (b_cur_seq_len, run_obj) in enumerate(zip(b_cur_len_numpy, run_reqs)):
            if b_cur_seq_len != 1:
                prefix_str_token_id = run_obj.input_token_ids[-1]
                start_token_str = self.token_id_to_token[prefix_str_token_id]
                end_token_str = start_token_str + self.pad_token_str
                start_index = self.sorted_tokens.bisect_left((start_token_str, None))
                end_index = self.sorted_tokens.bisect_right((end_token_str, None))
                mask[i, self.token_indexes[start_index:end_index]] = False
            else:
                mask[i, :] = False

        all_logics[mask] = -1000000.0

        next_token_ids, next_token_probs = sample(all_logics, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        free_token_indexes = []

        for req_obj, next_token_id, next_token_logprob, cur_seq_len in zip(
            run_reqs, next_token_ids, next_token_logprobs, b_cur_len_numpy
        ):
            # 如果当前prefill 长度是 1， 则不进行特殊的token healing 前缀后处理操作。
            if cur_seq_len == 1:
                req_obj.cur_kv_len = len(req_obj.input_token_ids)
                req_obj.input_token_ids.append(next_token_id)
                req_obj.out_token_id_count[next_token_id] += 1
                metadata = {
                    "id": int(next_token_id),
                    "logprob": float(next_token_logprob),
                }
            else:
                prefix_token_id = req_obj.input_token_ids[-1]
                del req_obj.input_token_ids[-1]
                req_obj.cur_kv_len = len(req_obj.input_token_ids)
                req_obj.input_token_ids.append(next_token_id)
                req_obj.out_token_id_count[next_token_id] += 1
                metadata = {
                    "id": int(next_token_id),
                    "logprob": float(next_token_logprob),
                    "prefix_str_token_id": int(prefix_token_id),
                }
                # 最后一个多余的 token 位置需要移除掉，主要是 prepare_prefill_inputs 是个全长输入操作, 只截取了倒数第二个位置的logics做计算
                # 所以在处理完成后需要将其占用的token，释放掉，以免造成显存管理泄露
                free_token_index = self.model.req_manager.req_to_token_indexs[
                    req_obj.req_idx, req_obj.cur_kv_len : (req_obj.cur_kv_len + 1)
                ]
                free_token_indexes.append(free_token_index)

            output_dict[req_obj.r_id] = (
                req_obj.req_status,
                req_obj.cur_kv_len,
                req_obj.get_output_len(),
                [(int(next_token_id), metadata)],
                req_obj.finish_status.value,
                None,
            )

        if free_token_indexes:
            self.model.mem_manager.free(torch.cat(free_token_indexes, dim=0))

        self.cache[batch.batch_id] = batch
        return output_dict
