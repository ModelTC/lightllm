import sys
from typing import AsyncGenerator
import zmq
import zmq.asyncio
import asyncio
import uvloop
import rpyc
import time
import hashlib
import datetime

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from ..tokenizer import get_tokenizer
from ..io_struct import BatchStrOut, AbortReq, FinishStatus, IdleReq
from ..embed_cache.utils import get_shm_name_data, create_shm
from ..req_id_generator import convert_sub_id_to_group_id
from ..sampling_params import SamplingParams
from lightllm.utils.log_utils import init_logger
from lightllm.server.metrics.manager import MetricClient

logger = init_logger(__name__)


class HttpServerManager:
    def __init__(
        self,
        args,
        push_to_router_urls,
        cache_url,
        pull_from_detokenization_urls,
        visual_url,
        metric_url,
        enable_multimodal,
    ):
        self.args = args
        context = zmq.asyncio.Context(len(push_to_router_urls) + len(pull_from_detokenization_urls) + 1)

        self.send_to_router_sockets = [context.socket(zmq.PUSH) for _ in range(len(push_to_router_urls))]
        for i, url in enumerate(push_to_router_urls):
            logger.info(f"connect to send_to_router {url}")
            self.send_to_router_sockets[i].connect(f"tcp://{url}")

        self.enable_multimodal = enable_multimodal
        if self.enable_multimodal:
            cache_host, cache_port = cache_url.split(":")
            self.cache_client = rpyc.connect(cache_host, int(cache_port))
            self.send_to_visual = context.socket(zmq.PUSH)
            self.send_to_visual.connect(f"tcp://{visual_url}")

        self.recv_from_detokenization_sockets = [
            context.socket(zmq.PULL) for _ in range(len(pull_from_detokenization_urls))
        ]
        for i, url in enumerate(pull_from_detokenization_urls):
            logger.info(f"bind to recv_from_detokenization {url}")
            self.recv_from_detokenization_sockets[i].bind(f"tcp://{url}")

        assert len(self.send_to_router_sockets) >= len(
            self.recv_from_detokenization_sockets
        ), "router num < detokenization num"
        assert (
            len(self.send_to_router_sockets) % len(self.recv_from_detokenization_sockets) == 0
        ), "router num % detokenization num != 0"

        self.tokenizer = get_tokenizer(args.model_dir, args.tokenizer_mode, trust_remote_code=args.trust_remote_code)

        self.req_id_to_out_inf = {}  # value type (out_str, metadata, finished, event)

        self.max_req_input_len = args.max_req_input_len
        self.max_req_total_len = args.max_req_total_len
        self.metric_client = MetricClient(metric_url)
        return

    async def wait_model_init(self):
        detokenization_num = len(self.recv_from_detokenization_sockets)
        router_num = len(self.send_to_router_sockets)
        for detok_idx in range(detokenization_num):
            for router_idx in range(router_num):
                rec_ans = await self.recv_from_detokenization_sockets[detok_idx].recv_pyobj()
                assert isinstance(rec_ans, IdleReq), f"error recv type {type(rec_ans)}"
        logger.info("all prefill/normal instances are ready")

    # connect cache server, calculate md5, alloc resource, return uuid
    async def _alloc_resource(self, data, num):
        md5sum = hashlib.md5(data).hexdigest()
        wait_time = 1
        while True:
            record = self.cache_client.root.alloc(md5sum, num)
            # hit or new
            if record:
                uid = record["id"]
                if not self.cache_client.root.get_item_data(uid):
                    create_shm(get_shm_name_data(uid), data)
                    self.cache_client.root.set_item_data(uid)
                return record
            # cache full
            else:
                await asyncio.sleep(wait_time)
                wait_time = min(wait_time + 2, 9)

    async def _alloc_multimodal_resources(self, multimodal_params):
        for img in multimodal_params.images:
            record = await self._alloc_resource(img.read(), self.tokenizer.get_image_token_length(img))
            img.uuid = record["id"]
            img.token_id = record["token_id"]
            img.token_num = record["token_num"]

    async def _release_multimodal_resources(self, multimodal_params):
        if multimodal_params is not None:
            for img in multimodal_params.images:
                if img.uuid is not None:
                    self.cache_client.root.release(img.uuid)
                    # 将 uuid 等 赋值为 None, 防止因为abort等异常情况造成重复释放异常
                    img.uuid = None
                    img.token_id = None
                    img.token_num = None

    def tokens(self, prompt):
        prompt_ids = self.tokenizer.encode(prompt)
        return len(prompt_ids)

    async def generate(
        self, prompt, sampling_params: SamplingParams, group_request_id, multimodal_params, request=None
    ):
        # 记录请求到达的相关信息
        if request is not None:
            x_request_id = request.headers.get("X-Request-Id", "")
            x_session_id = request.headers.get("X-Session-Id", "")
            format_in_time = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
            logger.info(
                f"recieved req X-Request-Id:{x_request_id} "
                f"X-Session-Id:{x_session_id} start_time:{format_in_time} "
                f"lightllm_req_id:{group_request_id} "
            )

        # 监控
        self.metric_client.counter_inc("lightllm_request_count")

        sampling_params.stop_sentences_to_token_ids(self.tokenizer)

        # 统计信息变量
        start_time = time.time()
        out_token_counter = 0
        first_token_cost_ms = sys.float_info.max
        is_first_token = True

        if self.enable_multimodal:
            assert len(multimodal_params.images) <= self.args.cache_capacity, "too many images!"
            await self._alloc_multimodal_resources(multimodal_params)
            prompt_ids = self.tokenizer.encode(prompt, multimodal_params)
        else:
            prompt_ids = self.tokenizer.encode(prompt)
        prompt_tokens = len(prompt_ids)
        # 监控
        self.metric_client.histogram_observe("lightllm_request_input_length", prompt_tokens)
        self.metric_client.histogram_observe("lightllm_request_max_new_tokens", sampling_params.max_new_tokens)
        verify_time_begin = time.time()
        if prompt_tokens > self.max_req_input_len:
            # use long_truncation_mode to truncate long input len req.
            if self.args.long_truncation_mode is None:
                raise ValueError(f"the input prompt token len {prompt_tokens} is too long > {self.max_req_input_len}")
            elif self.args.long_truncation_mode == "head":
                prompt_ids = prompt_ids[-self.max_req_input_len :]
                prompt_tokens = len(prompt_ids)
            elif self.args.long_truncation_mode == "center":
                prompt_ids = (
                    prompt_ids[0 : self.max_req_input_len // 2]
                    + prompt_ids[-(self.max_req_input_len - self.max_req_input_len // 2) :]
                )
                prompt_tokens = len(prompt_ids)
                assert prompt_tokens == self.max_req_input_len
            else:
                assert False, "error args"

        req_total_len = prompt_tokens + sampling_params.max_new_tokens
        if req_total_len > self.max_req_total_len:
            raise ValueError(
                f"the req total len (input len + output len) is too long > max_req_total_len:{self.max_req_total_len}"
            )
        verify_time_end = time.time()

        req_status = ReqStatus(group_request_id, multimodal_params, self._assign_router_idx(group_request_id))
        event = req_status.event
        self.req_id_to_out_inf[group_request_id] = req_status

        if self.enable_multimodal:
            self.send_to_visual.send_pyobj(
                (prompt_ids, sampling_params, multimodal_params, group_request_id, start_time)
            )
        else:
            self.send_to_router_sockets[req_status.router_idx].send_pyobj(
                (prompt_ids, sampling_params, multimodal_params, group_request_id, start_time)
            )

        unfinished_count = sampling_params.best_of

        while True:
            try:
                await asyncio.wait_for(event.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass

            if request is not None and await request.is_disconnected():
                await self.abort(group_request_id, req_status)
                raise Exception(f"req_id {group_request_id} disconnected")

            async with req_status.lock:
                event.clear()
                if len(req_status.out_token_info_list) == 0:
                    continue

                for sub_req_id, out_str, metadata, finish_status in req_status.out_token_info_list:
                    metadata["prompt_tokens"] = prompt_tokens
                    out_token_counter += 1
                    first_token_cost_ms = (time.time() - start_time) * 1000 if is_first_token else first_token_cost_ms
                    is_first_token = False

                    yield sub_req_id, out_str, metadata, finish_status
                    # 如果有子请求完成，就更新计数
                    if finish_status.is_finished():
                        unfinished_count -= 1

                    # 所有子请求完成后，就删除占用的资源
                    if unfinished_count == 0:
                        try:
                            del self.req_id_to_out_inf[group_request_id]
                            await self._release_multimodal_resources(multimodal_params)
                        except:
                            pass
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
                        self.metric_client.histogram_observe(
                            "lightllm_request_validation_duration", verify_time_end - verify_time_begin
                        )
                        self.metric_client.histogram_observe(
                            "lightllm_request_inference_duration", total_cost_time_ms / 1000.0
                        )
                        self.metric_client.histogram_observe(
                            "lightllm_request_mean_time_per_token_duration", mean_per_token_cost_time_ms / 1000.0
                        )
                        self.metric_client.histogram_observe(
                            "lightllm_request_first_token_duration", first_token_cost_ms / 1000.0
                        )
                        self.metric_client.histogram_observe("lightllm_request_generated_tokens", out_token_counter)
                        self.metric_client.counter_inc("lightllm_request_success")

                        return
                req_status.out_token_info_list.clear()
        return

    def _assign_router_idx(self, group_request_id):
        return hash(group_request_id) % len(self.send_to_router_sockets)

    async def abort(self, group_request_id, req_status):
        abort_req = AbortReq(group_req_id=group_request_id)
        self.send_to_router_sockets[req_status.router_idx].send_pyobj(abort_req)
        if self.enable_multimodal:
            self.send_to_visual.send_pyobj(abort_req)
        try:
            req = self.req_id_to_out_inf[group_request_id]
            await self._release_multimodal_resources(req.multimodal_params)
            del self.req_id_to_out_inf[group_request_id]
        except:
            pass
        logger.warning(f"aborted group_request_id {group_request_id}")
        return

    async def _recv_from_socket(self, socket):
        message = await socket.recv_pyobj()  # 接收来自 socket 的消息
        return message  # 每次接收到消息就 yield 出去

    async def _recv_from_sockets(self, sockets):
        tasks = {asyncio.create_task(self._recv_from_socket(socket)): socket for socket in sockets}
        while True:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                socket = tasks.pop(task)
                message = await task
                yield message
                new_task = asyncio.create_task(self._recv_from_socket(socket))
                tasks[new_task] = socket

    async def handle_loop(self):
        async for recv_ans in self._recv_from_sockets(self.recv_from_detokenization_sockets):
            assert isinstance(recv_ans, BatchStrOut), f"error recv type {type(recv_ans)}"
            for sub_req_id, text, metadata, finish_status in recv_ans.reqs_infs:
                finish_status = FinishStatus(finish_status)
                group_req_id = convert_sub_id_to_group_id(sub_req_id)
                try:
                    if not finish_status.is_aborted():
                        req_status: ReqStatus = self.req_id_to_out_inf[group_req_id]
                        async with req_status.lock:
                            req_status.out_token_info_list.append((sub_req_id, text, metadata, finish_status))
                            req_status.event.set()
                    else:
                        del self.req_id_to_out_inf[group_req_id]
                except:
                    pass
        return


class ReqStatus:
    def __init__(self, req_id, multimodal_params, router_idx) -> None:
        self.req_id = req_id
        self.multimodal_params = multimodal_params
        self.lock = asyncio.Lock()
        self.event = asyncio.Event()
        self.out_token_info_list = []
        self.router_idx = router_idx
