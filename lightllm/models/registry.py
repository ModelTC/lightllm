import collections
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

from dataclasses import dataclass
from typing import Type, Dict, Optional, Callable, List
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@dataclass
class ModelConfig:
    model_class: Type
    is_multimodal: bool = False
    condition: Optional[Callable[[dict], bool]] = None


class _ModelRegistries:
    def __init__(self):
        self._registry: Dict[str, List[ModelConfig]] = collections.defaultdict(list)

    def __call__(
        self,
        model_type: str,
        is_multimodal: bool = False,
        condition: Optional[Callable[[dict], bool]] = None,
    ):
        """Decorator to register a model (now more concise)."""

        def decorator(
            model_class: Type,
        ):
            model_types = [model_type] if isinstance(model_type, str) else model_type
            for mt in model_types:
                self._registry[mt].append(
                    ModelConfig(model_class=model_class, is_multimodal=is_multimodal, condition=condition)
                )
            return model_class

        return decorator

    def get_model(self, model_cfg: dict, model_kvargs: dict) -> tuple:
        """Get model"""
        model_type = model_cfg.get("model_type", "")
        configs = self._registry.get(model_type, [])
        model = None
        is_multimodal = False
        matches = []
        for cfg in configs:
            if cfg.condition is None or cfg.condition(model_cfg):
                matches.append(cfg)

        if len(matches) == 0:
            raise ValueError(f"Model type {model_type} is not supported.")

        if len(matches) > 1:
            # Keep conditionally matched models
            matches = [m for m in matches if m.condition is not None]
        assert (
            len(matches) == 1
        ), "Existence of coupled conditon, inability to determine the class of models instantiated"
        model = matches[0].model_class(model_kvargs)
        is_multimodal = matches[0].is_multimodal
        return model, is_multimodal


ModelRegistry = _ModelRegistries()


def get_model(model_cfg: dict, model_kvargs: dict):
    try:
        model, is_multimodal = ModelRegistry.get_model(model_cfg, model_kvargs)
        return model, is_multimodal
    except Exception as e:
        logger.exception(str(e))
        raise


def has_visual_config(cfg: dict) -> bool:
    return "visual" in cfg


def is_reward_model() -> Callable[[Dict[str, any]], bool]:
    return lambda c: "RewardModel" in c.get("architectures", [])


def architecture_is(name: str) -> Callable[[Dict[str, any]], bool]:
    """Predicate: matches first element of model_cfg['architectures'] == name."""
    return lambda c: c.get("architectures", [""])[0] == name


def llm_model_type_is(name: str) -> Callable[[Dict[str, any]], bool]:
    names = [name] if isinstance(name, str) else name
    """Predicate: matches model_cfg.get("llm_config").get("model_type") == name."""
    return lambda c: (
        c.get("llm_config", {}).get("model_type", "") in names
        or c.get("text_config", {}).get("model_type", "") in names
    )
