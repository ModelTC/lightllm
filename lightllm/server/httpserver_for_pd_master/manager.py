import sys
import zmq
import zmq.asyncio
import asyncio
import uvloop
import rpyc
import time
import hashlib
import datetime
import aiohttp
import json

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from typing import Union, List, Tuple, Dict
from ..io_struct import FinishStatus, PD_Client_Obj
from ..sampling_params import SamplingParams
from ..multimodal_params import MultimodalParams
from ..req_id_generator import ReqIDGenerator
from fastapi import Request
from lightllm.utils.log_utils import init_logger
from lightllm.server.metrics.manager import MetricClient

logger = init_logger(__name__)


class HttpServerManagerForPDMaster:
    def __init__(
        self,
        args,
        metric_port,
    ):
        self.args = args
        self.metric_client = MetricClient(metric_port)
        self.id_gen = ReqIDGenerator()
        self.prefill_nodes: List[PD_Client_Obj] = []
        self.decode_nodes: List[PD_Client_Obj] = []
        self.url_to_pd_nodes: Dict[str, PD_Client_Obj] = {}

        self.id_to_event: Dict[int, asyncio.Event] = {}
        return

    async def register_pd(self, pd_info_json):
        pd_client = PD_Client_Obj(**pd_info_json)
        self.url_to_pd_nodes[pd_client.client_ip_port] = pd_client
        if pd_client.mode == "prefill":
            self.prefill_nodes.append(pd_client)
        elif pd_client.mode == "decode":
            self.decode_nodes.append(pd_client)
        else:
            assert False

        logger.info(f"mode: {pd_client.mode} url: {pd_client.client_ip_port} registed")
        return

    async def remove_pd(self, pd_info_json):
        pd_client = PD_Client_Obj(**pd_info_json)
        try:
            del self.url_to_pd_nodes[pd_client.client_ip_port]
        except:
            pass
        self.prefill_nodes = [e for e in self.prefill_nodes if e.client_ip_port != pd_client.client_ip_port]
        self.decode_nodes = [e for e in self.decode_nodes if e.client_ip_port != pd_client.client_ip_port]
        logger.info(f"mode: {pd_client.mode} url: {pd_client.client_ip_port} removed")
        return

    async def update_req_status(self, group_request_id):
        try:
            event = self.id_to_event[group_request_id]
            event.set()
            del self.id_to_event[group_request_id]
        except:
            pass
        return

    def tokens(self, prompt: str):
        # to do
        raise NotImplementedError("tokens is not implements")

    async def select_p_d_node(
        self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams
    ):
        import random

        p_node = random.choice(self.prefill_nodes)
        d_node = random.choice(self.decode_nodes)
        return p_node, d_node

    async def generate(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
        request: Request,
    ) -> Tuple[int, str, dict, FinishStatus]:
        start_time = time.time()
        group_request_id = self.id_gen.generate_id()
        sampling_params.group_request_id = group_request_id
        # 记录请求到达的相关信息
        await self._log_req_header(request, group_request_id)
        # 监控
        self.metric_client.counter_inc("lightllm_request_count")
        self.metric_client.histogram_observe("lightllm_request_max_new_tokens", sampling_params.max_new_tokens)

        p_node, d_node = await self.select_p_d_node(prompt, sampling_params, multimodal_params)

        results_generator = self._wait_to_token_package(
            p_node.to_llm_url(), d_node.to_llm_url(), start_time, prompt, sampling_params, multimodal_params, request
        )
        async for sub_req_id, request_output, metadata, finish_status in results_generator:
            yield sub_req_id, request_output, metadata, finish_status
        return

    async def _log_req_header(self, request: Request, group_request_id: int):
        x_request_id = request.headers.get("X-Request-Id", "")
        x_session_id = request.headers.get("X-Session-Id", "")
        format_in_time = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            f"recieved req X-Request-Id:{x_request_id} "
            f"X-Session-Id:{x_session_id} start_time:{format_in_time} "
            f"lightllm_req_id:{group_request_id} "
        )
        return

    async def _to_req_info(
        self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams
    ):
        req = {
            "inputs": prompt,
            "parameters": sampling_params.to_origin_dict(),
            "multimodal_params": multimodal_params.to_origin_dict(),
        }
        return req

    async def fetch_stream(
        self,
        p_url,
        d_url,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
    ):
        group_request_id = sampling_params.group_request_id
        event = asyncio.Event()
        self.id_to_event[group_request_id] = event

        try:
            old_max_new_tokens = sampling_params.max_new_tokens
            sampling_params.max_new_tokens = 1
            if old_max_new_tokens == 1:
                sampling_params.move_kv_to_decode_node = False
            else:
                sampling_params.move_kv_to_decode_node = True

            req = await self._to_req_info(prompt, sampling_params, multimodal_params)
            async with aiohttp.ClientSession() as session:
                async with session.post(p_url, json=req) as response:
                    if response.status == 200:
                        async for line in response.content:
                            line = line.decode("utf-8").strip()
                            if line.startswith("data:"):
                                data = line[len("data:") :].strip()
                                sub_req_id, request_output, metadata, finish_status = json.loads(data)
                                if old_max_new_tokens != 1:
                                    finish_status = FinishStatus.NO_FINISH
                                else:
                                    finish_status = FinishStatus(finish_status)
                                yield sub_req_id, request_output, metadata, finish_status
                    else:
                        logger.error(f"fetch_stream error: {response.status}")
                        raise Exception(f"group_req_id {group_request_id} connection error: {response}")

            # 如果只需要一个输出 token，prefill 完就直接结束掉吧
            if old_max_new_tokens == 1:
                return

            try:
                await asyncio.wait_for(event.wait(), timeout=60)
            except asyncio.TimeoutError:
                logger.warning(f"group_request_id: {group_request_id} time out err")
                # raise Exception(f"group_request_id: {group_request_id} time out err, maybe kv move get questions")

            sampling_params.move_kv_to_decode_node = None
            sampling_params.max_new_tokens = old_max_new_tokens - 1
            req = await self._to_req_info(prompt, sampling_params, multimodal_params)
            async with aiohttp.ClientSession() as session:
                async with session.post(d_url, json=req) as response:
                    if response.status == 200:
                        async for line in response.content:
                            line = line.decode("utf-8").strip()
                            if line.startswith("data:"):
                                data = line[len("data:") :].strip()
                                sub_req_id, request_output, metadata, finish_status = json.loads(data)
                                yield sub_req_id, request_output, metadata, FinishStatus(finish_status)
                    else:
                        logger.error(f"fetch_stream error: {response.status}")
                        raise Exception(f"group_req_id {group_request_id} connection error: {response}")
        finally:
            try:
                del self.id_to_event[group_request_id]
            except:
                pass
        return

    async def _wait_to_token_package(
        self,
        p_url,
        d_url,
        start_time,
        prompt: str,
        sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
        request: Request,
    ):
        out_token_counter = 0
        first_token_cost_ms = sys.float_info.max
        group_request_id = sampling_params.group_request_id
        unfinished_count = sampling_params.best_of

        async for sub_req_id, out_str, metadata, finish_status in self.fetch_stream(
            p_url, d_url, prompt, sampling_params, multimodal_params
        ):
            if await request.is_disconnected():
                await self.abort(group_request_id)
                raise Exception(f"req_id {group_request_id} disconnected")
            prompt_tokens = metadata["prompt_tokens"]
            out_token_counter += 1
            first_token_cost_ms = min((time.time() - start_time) * 1000, first_token_cost_ms)
            yield sub_req_id, out_str, metadata, finish_status
            if finish_status.is_finished():
                unfinished_count -= 1
            if unfinished_count == 0:
                break

        total_cost_time_ms = (time.time() - start_time) * 1000
        mean_per_token_cost_time_ms = (total_cost_time_ms - first_token_cost_ms) / out_token_counter
        x_request_id = request.headers.get("X-Request-Id", "")
        x_session_id = request.headers.get("X-Session-Id", "")
        prompt_cache_len = metadata.pop("prompt_cache_len", 0)
        prompt_cache_ratio = prompt_cache_len / prompt_tokens
        format_start_time = datetime.datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            f"X-Request-Id:{x_request_id} "
            f"X-Session-Id:{x_session_id} start_time:{format_start_time} "
            f"lightllm_req_id:{group_request_id} first_token_cost:{first_token_cost_ms}ms "
            f"total_cost_time:{total_cost_time_ms}ms,out_token_counter:{out_token_counter} "
            f"mean_per_token_cost_time: {mean_per_token_cost_time_ms}ms "
            f"prompt_token_num:{prompt_tokens} "
            f"prompt_cache_len:{prompt_cache_len} "
            f"prompt_cache_ratio:{prompt_cache_ratio} "
        )
        self.metric_client.histogram_observe("lightllm_request_inference_duration", total_cost_time_ms / 1000.0)
        self.metric_client.histogram_observe(
            "lightllm_request_mean_time_per_token_duration", mean_per_token_cost_time_ms / 1000.0
        )
        self.metric_client.histogram_observe("lightllm_request_first_token_duration", first_token_cost_ms / 1000.0)
        self.metric_client.histogram_observe("lightllm_request_generated_tokens", out_token_counter)
        self.metric_client.counter_inc("lightllm_request_success")
        return

    async def abort(self, group_request_id):
        logger.warning(f"aborted group_request_id {group_request_id}")
        return

    async def handle_loop(self):
        while True:
            # 可以做一个定时任务
            await asyncio.sleep(10)
        return
