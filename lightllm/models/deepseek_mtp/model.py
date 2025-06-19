from lightllm.models.deepseek2.model import Deepseek2TpPartModel
from lightllm.models.deepseek_mtp.layer_infer.pre_layer_infer import Deepseek3MTPPreLayerInfer
from lightllm.models.deepseek_mtp.layer_weights.pre_and_post_layer_weight import Deepseek3MTPPreAndPostLayerWeight
from lightllm.common.basemodel.basemodel_mtp import BaseMTPModelRunner


class Deepseek3MTPModel(BaseMTPModelRunner, Deepseek2TpPartModel):

    pre_and_post_weight_class = Deepseek3MTPPreAndPostLayerWeight
    pre_layer_infer_class = Deepseek3MTPPreLayerInfer
