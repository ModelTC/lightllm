import math
from abc import ABCMeta
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Sequence, Tuple, cast

import torch

from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.server.router.model_infer.infer_batch import InferBatch
from lightllm.server.tokenizer import get_tokenizer

from .heal import TokenHealingBatchState, TokenHealingResult
from .lightllm_post_weights import ReorderedPostLayerWeightMixin, reorder_post_weights
from .vocab import hf_tokenizer_to_bytes_vocab

if TYPE_CHECKING:
    from general_sam import VocabPrefixAutomaton
    from transformers import PreTrainedTokenizerBase

    from lightllm.common.basemodel.basemodel import TpPartBaseModel
    from lightllm.server.router.model_infer.model_rpc import ModelRpcServer


@dataclass
class LightLLMTokenHealingTable:
    max_token_healing_top_k: int
    automaton: "VocabPrefixAutomaton"
    reordered_weights: ReorderedPostLayerWeightMixin

    @classmethod
    def init_from_model(
        cls,
        model: "TpPartBaseModel",
        model_weightdir,
        tokenizer_mode,
        trust_remote_code,
        max_token_healing_top_k: int,
    ) -> Optional["LightLLMTokenHealingTable"]:
        tokenizer = get_tokenizer(
            model_weightdir, tokenizer_mode, trust_remote_code=trust_remote_code
        )
        return cls.init_from_model_and_tokenizer(
            model, tokenizer, max_token_healing_top_k
        )

    @classmethod
    def init_from_model_and_tokenizer(
        cls,
        model: "TpPartBaseModel",
        tokenizer: "PreTrainedTokenizerBase",
        max_token_healing_top_k: int,
    ) -> Optional["LightLLMTokenHealingTable"]:
        # disable token healing
        if max_token_healing_top_k <= 0:
            return None

        try:
            import general_sam
        except ImportError as e:
            if hasattr(e, "add_note"):
                e.add_note(
                    "Please install `general-sam` to enable token healing:\n"
                    "pip install general-sam"
                )
            raise e

        vocab = hf_tokenizer_to_bytes_vocab(tokenizer)
        automaton = general_sam.VocabPrefixAutomaton(vocab, "bytes")

        reordered_weights = reorder_post_weights(
            model.pre_post_weight, automaton.get_order()
        )
        # assert reordered_weights is not None, (
        #     "token healing has not been implemented for "
        #     f"{model.pre_and_post_weight_class}"
        # )

        return cls(
            max_token_healing_top_k=max_token_healing_top_k,
            automaton=automaton,
            reordered_weights=reordered_weights,
        )

    def need_token_healing(self, batch: InferBatch) -> bool:
        return (
            sum(
                min(i.token_healing_top_k, self.max_token_healing_top_k)
                for i in batch.sampling_param_list
            )
            > 0
        )

    def infer_rpc_server_batch(
        self, server: "ModelRpcServer", batch: InferBatch
    ) -> Tuple[Dict, InferBatch]:
        return ModelRpcServerBatchState(
            server,
            batch,
            self.max_token_healing_top_k,
            self.automaton,
            self.reordered_weights,
        ).heal_batch()


class BaseLightLLMBatchState(TokenHealingBatchState, metaclass=ABCMeta):
    def __init__(
        self,
        model: "TpPartBaseModel",
        batch_size: int,
        input_ids: torch.Tensor,
        b_loc: torch.Tensor,
        b_seq_len: torch.Tensor,
        b_start_loc: torch.Tensor,
        total_token_num: int,
        max_len_in_batch: int,
        automaton: "VocabPrefixAutomaton",
        reordered_weights: ReorderedPostLayerWeightMixin,
        sampling_top_ks: Iterable[int],
    ):
        super().__init__(batch_size, automaton)

        self.sampling_top_ks = tuple(sampling_top_ks)

        self.model = model
        self.reordered_weights = reordered_weights

        self._run_infer(
            input_ids, b_loc, b_seq_len, b_start_loc, total_token_num, max_len_in_batch
        )

    def _run_infer(
        self,
        input_ids: torch.Tensor,
        b_loc: torch.Tensor,
        b_seq_len: torch.Tensor,
        b_start_loc: torch.Tensor,
        total_token_num: int,
        max_len_in_batch: int,
    ):
        last_pos_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        last_pos_mask[torch.cumsum(b_seq_len, dim=0) - 1] = True

        self.pending_input_ids, self.input_ids = (
            input_ids[last_pos_mask],
            input_ids[~last_pos_mask],
        )
        del input_ids, last_pos_mask

        total_token_num -= self.batch_size
        max_len_in_batch -= 1
        b_start_loc -= torch.arange(
            self.batch_size,
            dtype=b_start_loc.dtype,
            device=b_start_loc.device,
        )
        b_seq_len -= 1

        self.b_loc, self.b_start_loc, self.b_seq_len = b_loc, b_start_loc, b_seq_len
        self.total_token_num, self.max_len_in_batch = total_token_num, max_len_in_batch

        if len(self.input_ids) > 0:
            self.tot_input_embs, self.infer_state = self.model.forward(
                self.batch_size,
                self.total_token_num,
                self.max_len_in_batch,
                self.input_ids,
                self.b_loc,
                self.b_start_loc,
                self.b_seq_len,
                is_prefill="skip_predict",  # pyright: ignore
            )
        else:
            self.tot_input_embs, self.infer_state = cast(
                Tuple[torch.Tensor, InferStateInfo], (None, None)
            )

    def get_sampling_top_k(self, batch_idx: int) -> int:
        return self.sampling_top_ks[batch_idx]

    def get_prefilled_input_ids(self, batch_idx: int) -> torch.Tensor:
        seq_l = int(self.b_start_loc[batch_idx])
        seq_r = seq_l + int(self.b_seq_len[batch_idx])
        return self.input_ids[seq_l:seq_r]

    def get_pending_token_id(self, batch_idx: int) -> int:
        return int(self.pending_input_ids[batch_idx])

    def get_reordered_probs(
        self, batch_idx: int, pos: int, lower: int, upper: int, return_logits: bool
    ) -> torch.Tensor:
        seq_l = int(self.b_start_loc[batch_idx])
        return self.reordered_weights.infer_reordered_slice(
            lower,
            upper,
            self.model,
            self.infer_state,
            self.tot_input_embs[seq_l + pos],
            return_logits,
        ).flatten()


class ModelRpcServerBatchState(BaseLightLLMBatchState):
    def __init__(
        self,
        server: "ModelRpcServer",
        batch: InferBatch,
        max_token_healing_top_k: int,
        automaton: "VocabPrefixAutomaton",
        reordered_weights: ReorderedPostLayerWeightMixin,
    ):
        kwargs = batch.to_forward_kwargs("skip_predict")
        batch_size = kwargs.pop("batch_size")
        input_ids = kwargs.pop("input_ids")
        b_loc = kwargs.pop("b_loc")
        b_seq_len = kwargs.pop("b_seq_len")
        b_start_loc = kwargs.pop("b_start_loc")

        total_token_num = batch.nopad_total_token_num
        max_len_in_batch = batch.nopad_max_len_in_batch

        sampling_top_ks = tuple(
            min(i.token_healing_top_k, max_token_healing_top_k)
            for i in batch.sampling_param_list
        )

        super().__init__(
            server.model,
            batch_size,
            input_ids,
            b_loc,
            b_seq_len,
            b_start_loc,
            total_token_num,
            max_len_in_batch,
            automaton,
            reordered_weights,
            sampling_top_ks,
        )

        self.infer_batch = batch

    def gen_final_results(self, results: Sequence[TokenHealingResult]) -> Any:
        new_pending_token_ids = [i.new_pending_token_id for i in results]
        infer_batch = self.infer_batch

        infer_batch.input_ids = torch.tensor(
            new_pending_token_ids,
            dtype=self.input_ids.dtype,
            device=self.input_ids.device,
        )

        for idx, result in enumerate(results):
            infer_batch.all_input_ids[idx][-result.num_tokens_to_free - 1 :] = [
                result.new_pending_token_id
            ]
            infer_batch.input_lengths[idx] = len(infer_batch.all_input_ids[idx])
            # This will affect p_token_counts
            # infer_batch.out_token_id_counts[idx][-1] += 1

        infer_batch.nopad_total_token_num = sum(infer_batch.input_lengths)
        infer_batch.nopad_max_len_in_batch = max(infer_batch.input_lengths)

        infer_batch.nopad_b_seq_len = torch.tensor(
            infer_batch.input_lengths,
            dtype=self.b_seq_len.dtype,
            device=self.b_seq_len.device,
        )

        infer_batch.nopad_b_start_loc[0] = 0
        infer_batch.nopad_b_start_loc[1:] = torch.cumsum(
            infer_batch.nopad_b_seq_len, dim=0, dtype=torch.int32
        )[0:-1]

        loc_to_free = []
        # infer_batch.nopad_b_loc = torch.empty_like(self.b_loc)
        # assert infer_batch.nopad_b_loc is self.b_loc
        for idx, result in enumerate(results):
            old_b_loc_r = self.max_len_in_batch
            old_b_loc_l = old_b_loc_r - self.b_seq_len[idx]
            mem_loc = self.b_loc[idx, old_b_loc_l:old_b_loc_r]

            if result.num_tokens_to_free > 0:
                mem_loc, drop_loc = (
                    mem_loc[: -result.num_tokens_to_free],
                    mem_loc[-result.num_tokens_to_free :],
                )

                loc_to_free.append(drop_loc)

            b_loc_r = infer_batch.nopad_max_len_in_batch - 1
            b_loc_l = b_loc_r - (infer_batch.nopad_b_seq_len[idx] - 1)
            infer_batch.nopad_b_loc[idx, b_loc_l:b_loc_r] = mem_loc

        if loc_to_free:
            loc_to_free = torch.cat(loc_to_free, dim=0)
            infer_batch.mem_manager.free(loc_to_free)

        output_dict = {}
        for r, result in zip(infer_batch.requests, results):
            output_dict[r["request_id"]] = (
                -1,
                {
                    "id": -1,
                    "logprob": float(math.log(result.prob)),
                    "new_text": result.new_text,
                },
            )

        return output_dict, infer_batch
