from lightllm.server.router.dynamic_prompt.shared_arr import SharedArray
import numpy as np
import time


class TokenLoad:
    def __init__(self, name) -> None:
        self.shared_token_load = SharedArray(name, shape=(3,), dtype=np.float64)
        # 用来保存调度需要使用到的一些信息
        self.shared_token_infos = SharedArray(f"{name}_ext_infos", shape=(2,), dtype=np.int64)
        self.last_dynamic_max_load_update_time = time.time()

    # 记录系统调度器估计的峰值token使用量
    def set_estimated_peak_token_count(self, obj: int):
        self.shared_token_infos.arr[0] = obj
        return

    def get_estimated_peak_token_count(self) -> int:
        return self.shared_token_infos.arr[0]

    # 记录系统被临时固定的不能被使用的token数，主要在于 pd 分离的模式下
    # 推理系统需要在 kv 传输时临时固定一些 token， 防止调度系统估计失误，导致调度问题
    def set_frozened_token_count(self, obj: int):
        self.shared_token_infos.arr[1] = obj
        return

    def get_frozened_token_count(self) -> int:
        return self.shared_token_infos.arr[1]

    # current_load 当前使用token量，估计的负载
    def set_current_load(self, value):
        self.shared_token_load.arr[0] = value
        return

    def get_current_load(self):
        return self.shared_token_load.arr[0]

    # logical_max_load 朴素估计的负载，简单将当前请求的输入和输出长度想加得到, 目前已未使用，其值与dynamic_max_load一样
    def set_logical_max_load(self, value):
        self.shared_token_load.arr[1] = value
        return

    def get_logical_max_load(self):
        return self.shared_token_load.arr[1]

    # dynamic_max_load 动态估计的最大负载，考虑请求中途退出的情况，估计的最大token使用量
    def set_dynamic_max_load(self, value):
        self.shared_token_load.arr[2] = value
        self.set_logical_max_load(value)
        self.last_dynamic_max_load_update_time = time.time()
        return

    def get_dynamic_max_load(self):
        return self.shared_token_load.arr[2]

    def need_update_dynamic_max_load(self):
        # 5s 需要进行一次更新
        if time.time() - self.last_dynamic_max_load_update_time >= 5.0:
            return True
        else:
            return False
