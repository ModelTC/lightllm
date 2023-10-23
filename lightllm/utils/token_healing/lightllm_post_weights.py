import copy
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterable, MutableMapping, Type, cast

import torch
import torch.distributed as dist

from lightllm.common.basemodel.layer_weights.pre_and_post_layer_weight import (
    PreAndPostLayerWeight,
)
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import (
    LlamaPreAndPostLayerWeight,
)

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel
    from lightllm.common.basemodel.infer_struct import InferStateInfo


class ReorderedPostLayerWeightMixin(PreAndPostLayerWeight, metaclass=ABCMeta):
    @abstractmethod
    def get_vocab_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def pos_vocab_linear_weight_names(self) -> Iterable[str]:
        raise NotImplementedError

    @contextmanager
    def monkey_patch(
        self,
        old_infer_state: "InferStateInfo",
        model: "TpPartBaseModel",
        mock_vocab_size: int,
    ):
        old_is_prefill = old_infer_state.is_prefill
        old_batch_size = old_infer_state.batch_size
        old_infer_state.batch_size = 1  # pyright: ignore
        old_infer_state.is_prefill = False  # pyright: ignore

        old_vocab_size = model.post_infer.vocab_size_
        model.post_infer.vocab_size_ = mock_vocab_size
        try:
            yield old_infer_state
        finally:
            old_infer_state.is_prefill = old_is_prefill
            old_infer_state.batch_size = old_batch_size
            model.post_infer.vocab_size_ = old_vocab_size

    def reorder_post_vocab_embeddings(self, order: Iterable[int]):
        vocab_size = self.get_vocab_size()
        order = list(token_id < vocab_size and token_id or 0 for token_id in order)

        assert vocab_size % self.world_size_ == 0
        if len(order) < vocab_size:
            order.extend([0] * (vocab_size - len(order)))
        if len(order) % self.world_size_:
            order.extend([0] * (self.world_size_ - (len(order) % self.world_size_)))

        idx = None

        for attr in self.pos_vocab_linear_weight_names():
            weight = getattr(self, attr)

            assert len(weight) * self.world_size_ == vocab_size

            if self.world_size_ > 1:
                old_weight = weight
                weight = self._cuda(torch.empty((vocab_size, *old_weight.shape[1:])))
                dist.all_gather_into_tensor(weight, old_weight)

            if idx is None:
                idx = torch.tensor(
                    order[self.tp_rank_ : vocab_size : self.world_size_],
                    device=weight.device,
                )

            selected = torch.index_select(weight, 0, idx).detach()
            setattr(self, attr, self._cuda(selected))

        return self

    def slice_reordered_post_vocab(self, lower: int, upper: int):
        copied = copy.copy(self)

        slice_size = (upper - lower + self.world_size_ - 1) // self.world_size_
        tp_lower = lower // self.world_size_
        tp_upper = tp_lower + slice_size

        for attr in self.pos_vocab_linear_weight_names():
            setattr(copied, attr, getattr(copied, attr)[tp_lower:tp_upper].detach())

        return copied

    def infer_reordered_slice(
        self,
        lower: int,
        upper: int,
        model: "TpPartBaseModel",
        old_infer_state: "InferStateInfo",
        input_embs: torch.Tensor,
        return_logits: bool,
    ):
        # assert input_embs.dim() == 1
        input_embs = input_embs.reshape(1, len(input_embs))

        slice_per_tp = (upper - lower + self.world_size_ - 1) // self.world_size_

        with self.monkey_patch(
            old_infer_state, model, slice_per_tp * self.world_size_
        ) as infer_state:
            result = model.post_infer_forward(
                input_embs,
                infer_state,
                self.slice_reordered_post_vocab(lower, upper),
                return_logits,
            ).flatten()

            if self.world_size_ > 1:
                final_result = torch.empty_like(result)

                slice_per_tp = len(result) // self.world_size_
                for i in range(self.world_size_):
                    final_result[i : len(result) : self.world_size_] = result[
                        i * slice_per_tp : (i + 1) * slice_per_tp
                    ]

                result = final_result

        return result[
            lower % self.world_size_ : (lower % self.world_size_) + (upper - lower)
        ]


REORDER_POST_WEIGHTS_HANDLERS: MutableMapping[
    Type[PreAndPostLayerWeight], Type[ReorderedPostLayerWeightMixin]
] = dict()


def register_reorder_post_weights(classes: Iterable[Type[PreAndPostLayerWeight]]):
    def wrapper(target_class: Type[ReorderedPostLayerWeightMixin]):
        for cls in classes:
            REORDER_POST_WEIGHTS_HANDLERS[cls] = target_class

        return target_class

    return wrapper


def reorder_post_weights(pre_post_weights: PreAndPostLayerWeight, order: Iterable[int]):
    for cls in type(pre_post_weights).mro():
        if cls not in REORDER_POST_WEIGHTS_HANDLERS:
            continue

        instance = copy.copy(pre_post_weights)
        instance.__class__ = type(
            f"{REORDER_POST_WEIGHTS_HANDLERS[cls].__name__}_{id(instance)}",
            (REORDER_POST_WEIGHTS_HANDLERS[cls], instance.__class__),
            {},
        )
        return cast(
            ReorderedPostLayerWeightMixin, instance
        ).reorder_post_vocab_embeddings(order)

    return None


@register_reorder_post_weights((LlamaPreAndPostLayerWeight,))
class LlamaAlikeReorderedPostLayerWeight(ReorderedPostLayerWeightMixin):
    def get_vocab_size(self) -> int:
        return self.network_config_["vocab_size"]

    def pos_vocab_linear_weight_names(self) -> Iterable[str]:
        return ("lm_head_weight_",)
