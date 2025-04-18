from abc import abstractmethod

from lightllm.server.core.objs.sampling_params import SamplingParams
from lightllm.server.multimodal_params import AudioItem, ImageItem, MultimodalParams


class BaseMultiModalTokenizerWrapper:
    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        obj_dict = object.__getattribute__(self, "__dict__")
        if name in obj_dict:
            return obj_dict[name]
        return getattr(self.tokenizer, name)

    @abstractmethod
    def encode(self, prompt, multimodal_params: MultimodalParams = None, **kwargs):
        pass

    @abstractmethod
    def init_imageItem_extral_params(
        self, img: ImageItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        pass

    @abstractmethod
    def init_audioItem_extral_params(
        self, audio: AudioItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        pass

    @abstractmethod
    def get_image_token_length(self, img: ImageItem):
        pass

    @abstractmethod
    def get_audio_token_length(self, audio: AudioItem):
        pass
