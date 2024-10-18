import asyncio
import numpy as np
import rpyc
import torch
import os
from datetime import timedelta
from typing import Dict, List, Tuple
from transformers.configuration_utils import PretrainedConfig
from lightllm.server.router.model_infer.infer_batch import InferBatch
from rpyc.utils.classic import obtain
from lightllm.models.qwen_vl.qwen_visual import QWenVisionTransformer
from lightllm.models.llava.llava_visual import LlavaVisionModel
from lightllm.models.internlm_xcomposer.internlm_visual import InternVisionModel
from lightllm.models.internvl.internvl_visual import InternVLVisionModel
from lightllm.models.qwen2_vl.qwen2_visual import Qwen2VisionTransformerPretrainedModel
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end


class VisualModelRpcServer(rpyc.Service):
    def exposed_init_model(self, kvargs):
        import torch
        import torch.distributed as dist

        self.vit_dp = kvargs["vit_dp"]
        self.vit_tp = kvargs["vit_tp"]
        self.dp_rank_id = kvargs["dp_rank_id"]
        self.tp_rank_id = kvargs["tp_rank_id"]
        client_port = kvargs["client_port"]
        data_type = kvargs["data_type"]
        weight_dir = kvargs["weight_dir"]
        visual_gpu_ids = kvargs["visual_gpu_ids"]
        visual_nccl_port = kvargs["visual_nccl_port"]
        self.vit_rank_id = kvargs["vit_rank_id"]
        
        model_kvargs = {
            "tp_rank_id": self.tp_rank_id,
            "vit_tp": self.vit_tp,
            "weight_dir": weight_dir,
            "client_port": client_port,
            "data_type": data_type,
            "vit_rank_id":self.vit_rank_id,
            "visual_gpu":visual_gpu_ids[self.vit_rank_id]
        }
        if self.vit_tp != 1:
            dist.init_process_group(
                backend="nccl",
                init_method=f'tcp://127.0.0.1:{visual_nccl_port}',# 改这里 tp 才需要nccl， dp不需要， api_server里也要改（需要port应该，nccl_port不需要把？）
                rank=self.tp_rank_id,
                world_size=self.vit_tp,
            )
        print(f"self.tp_rank_id:{self.tp_rank_id}, self.vit_rank_id:{self.vit_rank_id},visual_gpu_ids[self.vit_rank_id] is {visual_gpu_ids[self.vit_rank_id]} ")
        torch.cuda.set_device(visual_gpu_ids[self.vit_rank_id])
        model_cfg, _ = PretrainedConfig.get_config_dict(weight_dir)

        try:
            self.model_type = model_cfg["model_type"]
            if self.model_type == "qwen":
                self.model = QWenVisionTransformer(model_kvargs, **model_cfg["visual"]).eval().bfloat16()
            elif self.model_type == "qwen2_vl":
                self.model = (
                    Qwen2VisionTransformerPretrainedModel(model_kvargs, **model_cfg["vision_config"]).eval().bfloat16()
                )
            elif self.model_type == "llava":
                self.model = LlavaVisionModel(model_kvargs)
            elif self.model_type == "internlmxcomposer2":
                self.model = InternVisionModel(model_kvargs)
            elif self.model_type == "internvl_chat":
                self.model = InternVLVisionModel(model_kvargs)
            else:
                raise Exception(f"can not support {self.model_type} now")

            self.model.load_model(weight_dir)
            self.model = self.model.cuda()
        except Exception as e:
            print("#" * 16)
            print("load model error:", str(e), e, type(e))
            import traceback

            traceback.print_exc()
            raise e

        set_random_seed(2147483647)
        return

    # @calculate_time(show=True, min_cost_ms=150)
    @torch.no_grad()
    def forward(self, images):
        return self.model.encode(images)

    # @calculate_time(show=False, min_cost_ms=300)
    def exposed_encode(self, images):
        return self.forward(images)


class VisualModelRpcClient:
    def __init__(self, model_rpc, vit_tp, rpc_server_process=None):
        self.model: VisualModelRpcServer = model_rpc
        self.vit_tp = vit_tp
        self.rpc_server_process = rpc_server_process
        self.use_rpc = self.vit_tp != 1
        if self.use_rpc:

            def async_wrap(f):
                f = rpyc.async_(f)

                async def _func(*args, **kwargs):
                    ans = f(*args, **kwargs)
                    await asyncio.to_thread(ans.wait)
                    # raise if exception
                    return ans.value

                return _func

            self._init_model = async_wrap(self.model.init_model)
            self._encode = async_wrap(self.model.encode)
        else:
            self._init_model = self.model.exposed_init_model
            self._encode = self.model.exposed_encode
        return

    async def init_model(self, kvargs):
        ans: rpyc.AsyncResult = self._init_model(kvargs)
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def encode(self, uuids):
        ans = self._encode(uuids)
        if self.use_rpc:
            return await ans
        else:
            return ans


def _init_env(port):
    # 注册graceful 退出的处理
    from lightllm.utils.graceful_utils import graceful_registry
    import inspect

    graceful_registry(inspect.currentframe().f_code.co_name)

    from rpyc.utils.server import ThreadedServer

    t = ThreadedServer(VisualModelRpcServer(), port=port, protocol_config={"allow_pickle": True})
    t.start()
    return


async def start_model_process(port, vit_tp):
    if vit_tp == 1:
        return VisualModelRpcClient(VisualModelRpcServer(), vit_tp)
    import multiprocessing

    proc = multiprocessing.Process(target=_init_env, args=(port,))
    proc.start()
    await asyncio.sleep(2)
    repeat_count = 0
    while repeat_count < 20:
        try:
            con = rpyc.connect("localhost", port, config={"allow_pickle": True})
            break
        except BaseException:
            await asyncio.sleep(1)
        repeat_count += 1
    if repeat_count == 20:
        raise Exception("init rpc env error!")

    assert proc.is_alive()
    return VisualModelRpcClient(con.root, vit_tp, rpc_server_process=proc)
