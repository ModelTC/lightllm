import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from lightllm.server.req_id_generator import convert_sub_id_to_group_id
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

# 节点的行为
class NodeRole(enum.Enum):
    P = "prefill"
    D = "decode"
    NORMAL = "normal"
    PD_MASTER = "pd_master"

    def is_P_or_NORMAL(self):
        return (self == NodeRole.P) or (self == NodeRole.NORMAL)


@dataclass
class PD_Client_Obj:
    node_id: str
    client_ip_port: str
    rdma_ip_port: str
    mode: str  # 只能是 prefill 或者 decode 节点
    start_args: object  # 节点的启动参数信息，用于做匹配性的校验，防止运行过程中出现问题。

    def __post_init__(self):
        if self.mode not in ["prefill", "decode"]:
            error_info = f"""mode must in ["prefill", "decode"], but get {self.mode}"""
            logger.error(error_info)
            raise ValueError(error_info)
        return

    def to_llm_url(self):
        return f"http://{self.client_ip_port}/pd_generate_stream"


@dataclass
class UpKVStatus:
    type: str = "kv_move_status"
    group_request_id: int = None

    def __post_init__(self):
        if self.type != "kv_move_status":
            error_info = "type only can be 'kv_move_status'"
            logger.error(error_info)
            raise ValueError(error_info)

        if not isinstance(self.group_request_id, int):
            error_info = "group_request_id only can be int"
            logger.error(error_info)
            raise ValueError(error_info)
        return


@dataclass
class DecodeNodeInfo:
    node_id: str
    ip: str
    rpyc_port: str


@dataclass
class KVMoveTask:
    group_request_id: int
    key: List[int]  # 代表输入的token_id 序列
    prefill_value: List[int]  # 在prefill节点上 mem manager kv buffer中的token index
    decode_value: List[int]  # 在decode节点上 mem manager kv buffer中的token index
    prefill_node_id: str
    decode_node: DecodeNodeInfo

    def __post_init__(self):
        if len(self.key) <= 0:
            error_info = "key must len >= 1"
            logger.error(error_info)
            raise ValueError(error_info)
