"""Sampling parameters for text generation."""
import os
from typing import List, Optional, Union, Tuple

# from .metrics import monitor
from .req_id_generator import MAX_BEST_OF

_SAMPLING_EPS = 1e-5
# 用环境变量控制是否进行输入惩罚的默认值
DEFAULT_INPUT_PENALTY = os.getenv("INPUT_PENALTY", "False").upper() in ["ON", "TRUE", "1"]


class SamplingParams:
    def __init__(
        self,
        best_of: int = 1,
        n: int = None,  # number of results
        do_sample: bool = False,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        exponential_decay_length_penalty: Tuple[int, float] = (1, 1.0),
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,  # -1 is for all
        ignore_eos: bool = False,
        max_new_tokens: int = 16,
        min_new_tokens: int = 1,
        stop_sequences: Optional[Union[str, List[str]]] = None,  # 停止句子条件
        skip_special_tokens: bool = True,  # whether to skip special tokens when decoding
        add_spaces_between_special_tokens: bool = True,  # whether to add spaces between special tokens when decoding
        print_eos_token: bool = False,  # eos_id will be always ignored except the value is set to True
        # Whether to count input tokens for presence_penalty, frequency_penalty and repetition_penalty
        input_penalty: bool = DEFAULT_INPUT_PENALTY,
    ) -> None:
        self.best_of = best_of
        self.n = n
        self.do_sample = do_sample
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.repetition_penalty = repetition_penalty
        self.exponential_decay_length_penalty = exponential_decay_length_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.ignore_eos = ignore_eos
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.stop_sequences = stop_sequences
        self.skip_special_tokens = skip_special_tokens
        self.add_spaces_between_special_tokens = add_spaces_between_special_tokens
        self.print_eos_token = print_eos_token
        if self.do_sample is False:
            self.temperature = 1.0
            self.top_p = 1.0
            self.top_k = 1
        if (
            self.temperature >= 0.0 and self.temperature < _SAMPLING_EPS
        ):  # temperature is too slow, change to greedy search
            self.temperature = 1.0
            self.top_k = 1
        self.input_penalty = input_penalty
        if self.n is None:
            self.n = self.best_of
        return

    def verify(self):
        if self.best_of <= 0 or self.best_of > MAX_BEST_OF:
            raise ValueError(f"need 0 < best_of <= {MAX_BEST_OF}, but get {self.best_of}")
        if self.n != self.best_of:
            raise ValueError("current only supported n == best_of")
        if self.n <= 0 or self.n > MAX_BEST_OF or self.n > self.best_of:
            raise ValueError(f"need 0 < n <= {MAX_BEST_OF}, n <= {self.best_of}, but get {self.n}")
        if self.presence_penalty < 0.0:
            raise ValueError(f"presence_penalty must >= 0.0, got {self.presence_penalty}")
        if self.frequency_penalty < 0.0:
            raise ValueError(f"frequency_penalty must >= 0.0, got {self.frequency_penalty}")
        if self.repetition_penalty < 1.0:
            raise ValueError(f"repetition_penalty must >= 1.0, got {self.repetition_penalty}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must > 0.0, got {self.temperature}")
        if self.top_p <= 0.0 or self.top_p > 1.0:
            raise ValueError(f"top_p must in (0.0, 1.0], got {self.top_p}")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(f"top_k must be -1 (disable), or at least 1, got {self.top_k}.")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be at least 1 , got {self.max_new_tokens}.")
        if self.min_new_tokens < 1:
            raise ValueError(f"min_new_tokens must be at least 1 , got {self.min_new_tokens}.")
        if self.min_new_tokens > self.max_new_tokens:
            raise ValueError(
                f"min_new_tokens must <= max_new_tokens, but got min {self.min_new_tokens}, max {self.max_new_tokens}."
            )

        if len(self.exponential_decay_length_penalty) != 2:
            raise ValueError(
                f"exponential_decay_length_penalty must be a tuple of (int, float), \
                got {self.exponential_decay_length_penalty}."
            )
        if (
            not isinstance(self.exponential_decay_length_penalty[0], int)
            or self.exponential_decay_length_penalty[0] < 0
        ):
            raise ValueError(
                f"exponential_decay_length_penalty[0] must be a non-negative integer, \
                got {self.exponential_decay_length_penalty[0]}."
            )
        if (
            not isinstance(self.exponential_decay_length_penalty[1], float)
            or self.exponential_decay_length_penalty[1] < 1.0
        ):
            raise ValueError(
                f"exponential_decay_length_penalty[1] must be a float >= 1.0, \
                got {self.exponential_decay_length_penalty[1]}."
            )

        return

    def stop_sentences_to_token_ids(self, tokenizer):
        if self.stop_sequences is None:
            self.stop_sequences = []
        else:
            if isinstance(self.stop_sequences, str):
                self.stop_sequences = [self.stop_sequences]
            new_stop_sequences = []
            for stop_str in self.stop_sequences:
                stop_str_ids = tokenizer.encode(stop_str)
                if stop_str_ids is not None and len(stop_str_ids) > 1:  # remove bos_token_id
                    stop_str_ids = stop_str_ids[1:]
                if len(stop_str_ids) > 0:
                    new_stop_sequences.append(stop_str_ids)
            self.stop_sequences = new_stop_sequences
        return

    def to_dict(self):
        ret = {}
        ret["do_sample"] = self.do_sample
        ret["presence_penalty"] = self.presence_penalty
        ret["frequency_penalty"] = self.frequency_penalty
        ret["repetition_penalty"] = self.repetition_penalty
        ret["exponential_decay_length_penalty"] = self.exponential_decay_length_penalty
        ret["temperature"] = self.temperature
        ret["top_p"] = self.top_p
        ret["top_k"] = self.top_k
        ret["min_new_tokens"] = self.min_new_tokens
        ret["ignore_eos"] = self.ignore_eos
        ret["max_new_tokens"] = self.max_new_tokens
        ret["stop_sequences"] = self.stop_sequences
        ret["best_of"] = self.best_of
        ret["input_penalty"] = self.input_penalty
        return ret
