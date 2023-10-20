import heapq
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Sequence, Tuple

import torch

if TYPE_CHECKING:
    from general_sam import VocabPrefixAutomaton


class LimitedPriorityQueue(object):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.queue = list()
        self.time_stamp = 0
        self.key_factor_mask = 1.0

    def push(self, key: float, value):
        item = key / self.key_factor_mask, self.time_stamp, value

        if len(self.queue) < self.capacity:
            heapq.heappush(self.queue, item)
        else:
            heapq.heappushpop(self.queue, item)

        self.time_stamp += 1

    def worst_key(self) -> float:
        if not self.queue:
            return 0.0
        return self.queue[0][0] * self.key_factor_mask

    def multiply_to_all_keys(self, factor: float):
        self.key_factor_mask *= factor

    def get_all(self) -> Iterable[Tuple[float, Any]]:
        return tuple((i * self.key_factor_mask, j) for i, _, j in self.queue)


@dataclass
class TokenHealingResult:
    num_tokens_to_free: int
    new_pending_token_id: int
    new_text: str
    prob: float


class TokenHealingBatchState(metaclass=ABCMeta):
    def __init__(self, batch_size: int, automaton: "VocabPrefixAutomaton"):
        self.batch_size = batch_size
        self.automaton = automaton

    @abstractmethod
    def get_sampling_top_k(self, batch_idx: int) -> int:
        ...

    @abstractmethod
    def gen_final_results(self, results: Sequence[TokenHealingResult]) -> Any:
        pass

    @abstractmethod
    def get_prefilled_input_ids(self, batch_idx: int) -> torch.Tensor:
        ...

    @abstractmethod
    def get_pending_token_id(self, batch_idx: int) -> int:
        ...

    @abstractmethod
    def get_reordered_probs(
        self, batch_idx: int, pos: int, lower: int, upper: int, return_logits: bool
    ) -> torch.Tensor:
        pass

    def heal_batch(self) -> Any:
        return self.gen_final_results(
            [self.heal_token(i) for i in range(self.batch_size)]
        )

    def heal_token(self, batch_idx: int) -> TokenHealingResult:
        pending_input_id = self.get_pending_token_id(batch_idx)
        sampling_top_k = self.get_sampling_top_k(batch_idx)
        input_ids = self.get_prefilled_input_ids(batch_idx)

        if (
            not self.automaton.vocab[pending_input_id]
            or sampling_top_k <= 0
            or len(input_ids) <= 0
        ):
            return TokenHealingResult(
                num_tokens_to_free=0,
                new_pending_token_id=pending_input_id,
                new_text="",
                prob=1.0,
            )

        queue = LimitedPriorityQueue(sampling_top_k)

        state = self.automaton.get_root_state()
        visited_seq_len = 0

        def feed(token_id):
            nonlocal state, visited_seq_len
            token_text = self.automaton.vocab[token_id]
            visited_seq_len += len(token_text)
            return self.automaton.prepend_feed(
                state,
                token_text,
            )

        cnt_info = feed(pending_input_id)
        # assert cnt_info is not None

        probs = self.get_reordered_probs(
            batch_idx,
            len(input_ids) - 1,
            cnt_info.tot_cnt_lower,
            cnt_info.tot_cnt_upper,
            False,
        )
        probs_topk = probs.topk(min(queue.capacity, len(probs)))

        for i in range(len(probs_topk.indices)):
            item = (
                0,
                int(probs_topk.indices[i]) + cnt_info.tot_cnt_lower,
                visited_seq_len,
            )
            queue.push(float(probs_topk.values[i]), item)

        pos = -1
        while not state.is_nil() and pos - 1 + len(input_ids) >= 0:
            token_id = input_ids[pos]
            if not self.automaton.vocab[token_id]:
                break

            token_rank = self.automaton.vocab_sort_res.rank[token_id]

            cnt_info = feed(token_id)
            if state.is_nil():
                break
            if cnt_info is None:
                pos -= 1
                continue

            logits_factor = self.get_reordered_probs(
                batch_idx,
                len(input_ids) + pos - 1,
                token_rank,
                token_rank + 1,
                True,
            )
            # assert len(logits_factor) == 1

            logits = self.get_reordered_probs(
                batch_idx,
                len(input_ids) + pos - 1,
                cnt_info.tot_cnt_lower,
                cnt_info.tot_cnt_upper,
                False,
            )

            probs_with_factor = torch.softmax(
                torch.cat((logits, logits_factor)).float(), dim=-1
            )
            probs, prob_factor = probs_with_factor[:-1], float(probs_with_factor[-1])

            queue.multiply_to_all_keys(prob_factor)

            probs_topk = probs.topk(min(queue.capacity, len(probs)))
            for i in range(len(probs_topk.indices)):
                item = (
                    pos,
                    int(probs_topk.indices[i]) + cnt_info.tot_cnt_lower,
                    visited_seq_len,
                )
                queue.push(float(probs_topk.values[i]), item)

            pos -= 1

        probs, items = zip(*queue.get_all())
        idx = int(torch.multinomial(torch.tensor(probs), 1)[0])
        prob = probs[idx]
        pos, token_rank, chunk_size = items[idx]
        token_id = self.automaton.vocab_sort_res.order[token_rank]
        token = self.automaton.vocab[token_id]

        try:
            new_text = token[chunk_size:].decode()
        except UnicodeDecodeError:
            new_text = ''

        return TokenHealingResult(
            num_tokens_to_free=-pos,
            new_pending_token_id=token_id,
            new_text=new_text,
            prob=prob,
        )
