import copy
import time
import math
import uuid
import uvloop
import asyncio
import rpyc

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import aiohttp
import zmq
import zmq.asyncio
from threading import Event as ThreadEvent, Lock as ThreadLock
from typing import Dict, List, Optional
from ..sampling_params import SamplingParams
from ..io_struct import (
    IdleReq,
    Req,
    NormalReq,
    SPDAssignReq,
    SPDCommitReq,
    SPDPreCommitReq,
    SplitFuseReq,
    TokenHealingReq,
    Batch,
    RouterLoadOut,
)
from ..multimodal_params import MultimodalParams
from .model_infer.model_rpc import start_model_process, ModelRpcClient
from .req_queue import build_req_queue
from rpyc.utils.classic import obtain
from lightllm.utils.infer_utils import calculate_time
from .dynamic_prompt.shared_arr import SharedInt
from .dynamic_prompt.radix_cache import RadixCacheReadOnlyClient
from ..io_struct import BatchTokenIdOut, AbortReq, ReqRunStatus, FinishStatus, ReqDetokenizationState
from .stats import Stats
from .pause_strategy import Fcfs, select_paused_reqs
from ..tokenizer import get_tokenizer
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.token_load import TokenLoad
from lightllm.server.req_id_generator import convert_sub_id_to_group_id
from lightllm.server.metrics.manager import MetricClient

logger = init_logger(__name__)


class RouterManager:
    def __init__(
        self,
        args,
        controller_to_router_url,
        router_to_detokenization_url,
        model_rpc_ports,
        metric_url,
        controller_write_feedback_from_router_url,
        decode_scheduler_url,
        controller_read_load_from_router_url,
    ):
        self.args = args
        self.model_weightdir = args.model_dir
        self.local_world_size = args.tp
        self.model_instance_num = args.model_instance_num
        self.model_instance_id = args.model_instance_id
        self.load_way = args.load_way
        self.mode = args.mode
        self.max_total_token_num = args.max_total_token_num
        # 用共享内存进行共享，router 模块读取进行精确的调度估计
        self.shared_can_use_token_num = SharedInt(f"{args.nccl_port}_mem_manger_can_use_token_num")
        # 初始化 radix_cache_client 用于读取 prompt cache 的管理信息
        self.radix_cache_client = None

        # 共享变量，用于存储router端调度分析得到的机器负载信息
        self.shared_token_load = TokenLoad(f"{str(args.nccl_port)}_shared_token_load")
        self.shared_token_load.set_current_load(0.0)
        self.shared_token_load.set_logical_max_load(0.0)
        self.shared_token_load.set_dynamic_max_load(0.0)

        self.pause_strategy = Fcfs()
        self.running_batch: Batch = None
        self.eos_id = args.eos_id
        self.has_wait_tokens = 0
        self.max_wait_tokens = args.router_max_wait_tokens

        dist_mode = getattr(args, "dist_mode", "normal")
        context = zmq.asyncio.Context(3 + int(dist_mode == "decode"))
        if dist_mode != "decode":
            # decode 模式下不需要接收 httpserver 的请求
            self.recv_from_httpserver = context.socket(zmq.PULL)
            self.recv_from_httpserver.bind(f"tcp://{controller_to_router_url}")

        self.send_to_detokenization = context.socket(zmq.PUSH)
        self.send_to_detokenization.connect(f"tcp://{router_to_detokenization_url}")
        logger.info(f"router connect to send_to_detokenization {router_to_detokenization_url}")
        self.model_rpc_ports = model_rpc_ports

        self.is_splitfuse_mode = args.splitfuse_mode
        self.is_token_healing = self.args.token_healing_mode
        self.splitfuse_block_size = args.splitfuse_block_size

        self.stats_tool = Stats(not args.disable_log_stats, args.log_stats_interval)
        self.metric_client = MetricClient(metric_url)

        assert dist_mode in ["normal", "prefill", "decode"], f"Error dist mode {dist_mode}"
        self.dist_mode = dist_mode
        if self.dist_mode == "decode":
            self.commit_history = {}
            self.commit_amount = 0
            self.commit_lock = ThreadLock()
            self.recv_from_decode_scheduler = context.socket(zmq.PULL)
            self.recv_from_decode_scheduler.bind(f"tcp://{controller_to_router_url}")
            self.send_to_decode_scheduler = context.socket(zmq.PUSH)
            self.send_to_decode_scheduler.connect(f"tcp://{controller_write_feedback_from_router_url}")
            self.thread_event = ThreadEvent()
            self.thread_event.set()
            self.send_load_to_controller = context.socket(zmq.PUSH)
            self.send_load_to_controller.connect(f"tcp://{controller_read_load_from_router_url}")
            self._session = None
        elif self.dist_mode == "prefill":
            self._req_to_target_instance = {}
            self.decode_scheduler_url = f"http://{decode_scheduler_url}"
            self.dropped_reqs = []

            class LastRecord:
                def __init__(self, n):
                    self.record = [0 for _ in range(n)]
                    self.lock = ThreadLock()

                def add(self, i, n):
                    with self.lock:
                        self.record[i] += n

                def read(self, i):
                    with self.lock:
                        d = self.record[i]
                        self.record[i] = 0
                    return d

            self.last_prefill_load = LastRecord(2)
        return

    async def wait_to_model_ready(self):
        # 初始化模型
        self.model_rpcs: List[ModelRpcClient] = []
        logger.info(f"start model rpcs {self.model_rpc_ports}")
        for rank_id in range(self.local_world_size):
            rpc_model = await start_model_process(port=self.model_rpc_ports[rank_id], world_size=self.local_world_size)
            self.model_rpcs.append(rpc_model)

        init_model_ret = []
        for rank_id in range(self.local_world_size):  # async init model process
            kvargs = {
                "args": self.args,
                "rank_id": rank_id + self.local_world_size * self.model_instance_id,
                "world_size": self.local_world_size * self.model_instance_num,
                "model_instance_id": self.model_instance_id,
                "model_instance_num": self.model_instance_num,
                "local_world_size": self.local_world_size,
                "weight_dir": self.model_weightdir,
                "load_way": self.load_way,
                "max_total_token_num": self.max_total_token_num,
                "mode": self.mode,
                "max_req_num": self.args.running_max_req_size + 8,
                "max_seq_length": self.args.max_req_total_len + 8,  # 留一点余量
                "nccl_port": self.args.nccl_port,
                "is_splitfuse_mode": self.is_splitfuse_mode,
                "splitfuse_block_size": self.splitfuse_block_size,
                "is_token_healing": self.args.token_healing_mode,
                "return_all_prompt_logprobs": self.args.return_all_prompt_logprobs,
                "use_reward_model": self.args.use_reward_model,
                "use_dynamic_prompt_cache": self.args.use_dynamic_prompt_cache,
                "data_type": self.args.data_type,
                "eos_id": self.eos_id,
                "beam_mode": self.args.beam_mode,
                "diverse_mode": self.args.diverse_mode,
                "graph_max_batch_size": self.args.graph_max_batch_size,
                "graph_max_len_in_batch": self.args.graph_max_len_in_batch,
                "disable_cudagraph": self.args.disable_cudagraph,
                "mem_fraction": self.args.mem_fraction,
                "batch_max_tokens": self.args.batch_max_tokens,
            }
            init_model_ret.append(self.model_rpcs[rank_id].init_model(kvargs))

        await asyncio.gather(*init_model_ret)
        if self.max_total_token_num is None:
            self.max_total_token_num = await self.model_rpcs[0].get_max_total_token_num()
            self.args.max_total_token_num = self.max_total_token_num
        if self.args.use_dynamic_prompt_cache:
            self.radix_cache_client = RadixCacheReadOnlyClient(
                str(self.args.nccl_port), self.max_total_token_num, tp_id=0
            )
        self.req_queue = build_req_queue(self.args, self)
        logger.info(f"use req queue {self.req_queue.__class__.__name__}")
        self.send_to_detokenization.send_pyobj(IdleReq(self.dist_mode))
        return

    async def push_load_to_controller(self):
        if self.dist_mode == "decode":
            load = self._dynamic_max_load()
            req = (len(self.running_batch.reqs) if self.running_batch is not None else 0) + len(
                self.req_queue.waiting_req_list
            )
        elif self.dist_mode == "normal" or self.dist_mode == "prefill":
            return
        else:
            raise ValueError(f"Error dist mode {self.dist_mode}")
        info = RouterLoadOut(self.model_instance_id, self.dist_mode, load, req)
        self.send_load_to_controller.send_pyobj(info)

    def add_req(
        self,
        prompt_ids: List[int],
        sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
        group_req_id: int,
        start_time: float,
    ):
        req_group = []
        for i in range(sampling_params.best_of):
            if self.is_splitfuse_mode:
                req = SplitFuseReq(
                    group_req_id + i,
                    copy.deepcopy(prompt_ids),
                    sampling_params,
                    multimodal_params,
                    self.splitfuse_block_size,
                )
            elif self.is_token_healing:
                req = TokenHealingReq(group_req_id + i, copy.deepcopy(prompt_ids), sampling_params, multimodal_params)
            else:
                req = NormalReq(group_req_id + i, copy.deepcopy(prompt_ids), sampling_params, multimodal_params)
            req.start_time = start_time
            req_group.append(req)

        self.req_queue.extend(req_group)
        logger.info(f"req_id: {group_req_id} add to req_queue prompt_ids: {prompt_ids}")
        self.send_to_detokenization.send_pyobj(
            ReqDetokenizationState(
                group_req_id,
                prompt_ids,
                sampling_params.max_new_tokens,
                sampling_params.ignore_eos,
                sampling_params.skip_special_tokens,
                sampling_params.add_spaces_between_special_tokens,
                sampling_params.print_eos_token,
                sampling_params.best_of,
            )
        )
        return

    async def abort(self, group_req_id):
        if self.running_batch is not None:
            for req in self.running_batch.reqs:
                if convert_sub_id_to_group_id(req.request_id) == group_req_id:
                    req.finish_status = FinishStatus.FINISHED_ABORT
        for req in self.req_queue.waiting_req_list:
            if convert_sub_id_to_group_id(req.request_id) == group_req_id:
                req.finish_status = FinishStatus.FINISHED_ABORT
        return

    async def loop_for_fwd(
        self,
    ):
        counter_count = 0
        while True:
            await self._step()
            counter_count += 1
            if self.running_batch is not None:
                if counter_count % 50 == 0:
                    token_ratio1 = self.get_used_tokens() / self.max_total_token_num
                    token_ratio2 = (
                        self.max_total_token_num - self.shared_can_use_token_num.get_value()
                    ) / self.max_total_token_num
                    logger.debug(
                        f"current batch size: {len(self.running_batch.reqs)} \n"
                        f"paused req num: {len(self.req_queue.pause_req_dict)} \n"
                        f"token used ratio: {token_ratio1} not contain prompt cache tree unrefed tokens\n"
                        f"token used ratio: {token_ratio2} contain prompt cache tree unrefed tokens"
                    )
                    self.shared_token_load.set_current_load(token_ratio1)
                    self.req_queue.update_token_load(self.running_batch)
                    pass
                self.stats_tool.print_stats()
                self.metric_client.gauge_set("lightllm_batch_current_size", len(self.running_batch.reqs))
                self.metric_client.gauge_set("lightllm_batch_pause_size", len(self.req_queue.pause_req_dict))
                self.metric_client.gauge_set("lightllm_queue_size", len(self.req_queue.waiting_req_list))
                self.metric_client.gauge_set(
                    "lightllm_batch_current_max_tokens",
                    int(self.shared_token_load.get_dynamic_max_load() * self.max_total_token_num),
                )
            else:
                self.shared_token_load.set_dynamic_max_load(0.0)
                self.shared_token_load.set_current_load(0.0)
                if counter_count % 300 == 0:
                    self.metric_client.gauge_set("lightllm_batch_current_size", 0.0)
                    self.metric_client.gauge_set("lightllm_batch_pause_size", 0.0)
                    self.metric_client.gauge_set("lightllm_queue_size", 0.0)
                    self.metric_client.gauge_set("lightllm_batch_current_max_tokens", 0.0)

            if self.running_batch is None:
                await asyncio.sleep(0.01)  # 10ms

    async def _step(self):
        if self.dist_mode == "normal":
            await self._normal_step()
        elif self.dist_mode == "prefill":
            await self._prefill_only_step()
        elif self.dist_mode == "decode":
            await self._decode_only_step()
        else:
            raise ValueError(f"Error dist mode {self.dist_mode}")

    async def _normal_step(self):
        """
        事件处理循环
        """
        # 删除所有已经 finished 的 req
        # 当前无运行请求时
        if self.running_batch is None:
            new_batch = self.req_queue.generate_new_batch(self.running_batch)
            if new_batch is not None:
                self.metric_client.histogram_observe("lightllm_batch_next_size", len(new_batch.reqs))
                for req in new_batch.reqs:
                    self.metric_client.histogram_observe(
                        "lightllm_request_queue_duration_bucket", time.time() - req.start_time
                    )
                self.stats_tool.count_prompt_tokens(new_batch)
                self.running_batch = new_batch
                await self._prefill_batch(self.running_batch)
                self._filter_runing_batch()
                self.has_wait_tokens = 0
            return

        # 有运行请求，但是已经到了可以调度新的请求合并推理的时机
        if self.has_wait_tokens >= self.max_wait_tokens:
            new_mini_batch = self.req_queue.generate_new_batch(self.running_batch)
            self.has_wait_tokens = 0
            if new_mini_batch is not None:
                self.stats_tool.count_prompt_tokens(new_mini_batch)
                await self._prefill_batch(new_mini_batch)
                if not new_mini_batch.is_clear():
                    await self._merge_batch(self.running_batch, new_mini_batch)
                    self.running_batch.merge(new_mini_batch)
                return

        # 正常 decode 阶段， 如果可以直接decode就直接decode，否则通过暂停策略暂停一些请求
        # 释放一些管理的 token
        if self._can_decode(self.running_batch):
            self.stats_tool.count_output_tokens(self.running_batch)
            await self._decode_batch(self.running_batch)
            self._filter_runing_batch()
            self.has_wait_tokens += 1
            return
        else:
            # pause strategy
            paused_reqs = select_paused_reqs(
                self.running_batch, self.pause_strategy, self.req_queue, self.max_total_token_num
            )
            await self._pause_reqs(self.running_batch, paused_reqs)
            logger.debug(f"pasued req num: {len(self.req_queue.pause_req_dict)}")
            self.has_wait_tokens = 0
            return
        return

    async def _prefill_only_step(self):
        assert self.running_batch is None, "prefill runnig batch must be none, as the prefill is ONE STEP "
        new_batch = self.req_queue.generate_new_batch(self.running_batch)
        if new_batch is not None:
            self.running_batch = new_batch
            await self._request_for_decode_scheduler(new_batch)
            if self.dropped_reqs:
                self.req_queue.back_to_wait_list(self.dropped_reqs)
                self.dropped_reqs = []
            await self._init_batch(new_batch)
            await self._prefill_communicating_with_decode_scheduler(new_batch)
            self.stats_tool.count_prompt_tokens(new_batch)
            await self._prefill_batch(new_batch)
            self._filter_runing_batch()
            await self._remove_batch(new_batch)
            self.has_wait_tokens = 0
            self.running_batch = None
        return

    async def _prefill_communicating_with_decode_scheduler(self, batch: Batch):
        req_ids = [req.request_id for req in batch.reqs]
        target_instance_id = self._req_to_target_instance[req_ids[0]]
        tasks = []
        for idx, rpc in enumerate(self.model_rpcs):
            tasks.append(
                rpc.send_request(
                    batch.batch_id,
                    target_instance_id * self.local_world_size + idx,
                )
            )
        ans = await asyncio.gather(*tasks)
        assert all([a == ans[0] for a in ans]), f"Error at communicating with decode scheduler {ans}"
        batch.set_p2p_idx(ans[0])

    async def _decode_only_step(self):
        if self.running_batch is None:
            new_batch = self.req_queue.generate_new_batch(self.running_batch)
            if new_batch is None:
                return
            else:
                await self._init_batch(new_batch)
                self.running_batch = new_batch
        else:
            if self.has_wait_tokens >= self.max_wait_tokens:
                new_mini_batch = self.req_queue.generate_new_batch(self.running_batch)
                self.has_wait_tokens = 0
                if new_mini_batch is not None:
                    await self._init_batch(new_mini_batch)
                    await self._merge_batch(self.running_batch, new_mini_batch)
                    self.running_batch.merge(new_mini_batch)

        assert self._can_decode(self.running_batch), "can not decode"
        self.stats_tool.count_output_tokens(self.running_batch)
        await self._decode_batch(self.running_batch)
        self._filter_runing_batch()
        self.has_wait_tokens += 1

    async def recv_req(self, req_info_list: list, commit_id: str):
        new_reqs = []
        for req_info in req_info_list:
            prompt_ids = req_info["input_id"]
            sampling_params = SamplingParams(**req_info["sampling_param"])
            multimodal_params = MultimodalParams(**req_info["multimodal_params"])
            req_id = req_info["request_id"]
            cur_kv_len = len(prompt_ids) + 1
            req = NormalReq(req_id, prompt_ids, sampling_params, multimodal_params)
            req.cur_kv_len = cur_kv_len
            req.is_recv = True
            if req_info["req_status"] == "ReqRunStatus.PAUSED_AND_OFFLOAD":
                req.req_status = ReqRunStatus.PAUSED_AND_OFFLOAD
            else:
                raise ValueError(f"Error at recv_req {req_info['req_status']}")
            self.req_queue.pause_req_dict[req.request_id] = req
            self.req_queue.append(req)
            new_reqs.append(req)

    async def _request_for_decode_scheduler(self, batch: Batch):
        cum_tokens = []
        needed_tokens = 0
        req_info = []
        for req in batch.reqs:
            needed_tokens += len(req.prompt_ids) + req.sample_params.max_new_tokens
            cum_tokens.append(needed_tokens)
            org = req.req_status
            req.req_status = ReqRunStatus.PAUSED_AND_OFFLOAD
            req_info.append(req.to_json())
            req.req_status = org
        assign_req = SPDAssignReq(cum_tokens, self.model_instance_id, batch.batch_id, req_info)
        while True:
            async with self._session.post(
                self.decode_scheduler_url,
                json=assign_req.to_http_obj(),
            ) as resp:
                if resp.status != 200:
                    logger.error(
                        f"Error when query the decode scheduler, status code: {resp.status}, req id: {req.request_id}"
                    )
                    continue
                resp_data = await resp.json()
                target_rank = resp_data.get("target_instance_id", -1)
                req_num = resp_data.get("req_num", -1)
                if target_rank == -1:
                    logger.info("no target rank, wait for 1s")
                    await asyncio.sleep(1)
                    continue
                self.dropped_reqs = batch.reqs[req_num:]
                batch.reqs = batch.reqs[:req_num]
                for req in batch.reqs:
                    self._req_to_target_instance[req.request_id] = target_rank
                break

    async def _init_batch(self, batch: Batch):
        reqs = [r.to_rpc_obj() for r in batch.reqs]
        rets = [self.model_rpcs[tp_rank].init_batch(batch.batch_id, reqs) for tp_rank in range(self.local_world_size)]
        ans = await asyncio.gather(*rets)
        if self.local_world_size != 1:
            req_to_req_status = obtain(ans[0])
        else:
            req_to_req_status = ans[0]

        self._update_init_status_to_batch(batch, req_to_req_status)
        for req in batch.reqs:
            prompt_cache_len = req.cur_kv_len
            prompt_cache_ratio = req.cur_kv_len / req.input_len
            req.prompt_cache_len = prompt_cache_len
            self.metric_client.histogram_observe("lightllm_cache_length", prompt_cache_len)
            self.metric_client.histogram_observe("lightllm_cache_ratio", prompt_cache_ratio)
            # logger.info(
            #     f"lightllm_req_id:{req.request_id} "
            #     f"prompt_cache_len:{prompt_cache_len} "
            #     f"prompt_cache_ratio:{prompt_cache_ratio} "
            # )
        logger.debug(f"Init Batch: {batch.simple_log()} \n")
        return

    async def _prefill_batch(self, batch: Batch):
        start_time = time.time()
        self.metric_client.counter_inc("lightllm_batch_inference_count", "prefill")
        if self.dist_mode == "normal":
            await self._init_batch(batch)
        if not self.is_splitfuse_mode:
            # 在 非 splitfuse 模式下，才需要真的执行 prefill 的操作。
            rets = [self.model_rpcs[tp_rank].prefill_batch(batch.batch_id) for tp_rank in range(self.local_world_size)]
            ans = await asyncio.gather(*rets)
            if self.local_world_size != 1:
                req_to_out_status = obtain(ans[0])
            else:
                req_to_out_status = ans[0]

            self._update_out_status_to_batch(batch, req_to_out_status)
            await self.push_load_to_controller()
            unfinished_req_ids, finished_req_ids = batch.mark_and_get_finished_req_and_preupdate_status()
            self._send_to_detokenization_proc(batch, req_to_out_status)
            batch.filter_out_finished_req(unfinished_req_ids, finished_req_ids)
            await self._handle_finish_req(batch, unfinished_req_ids, finished_req_ids)
        self.metric_client.histogram_observe(
            "lightllm_batch_inference_duration_bucket", time.time() - start_time, "prefill"
        )
        return

    async def _decode_batch(self, batch: Batch):
        start_time = time.time()
        self.metric_client.counter_inc("lightllm_batch_inference_count", "decode")
        rets = [self.model_rpcs[tp_rank].decode_batch(batch.batch_id) for tp_rank in range(self.local_world_size)]
        ans = await asyncio.gather(*rets)
        if self.local_world_size != 1:
            req_to_out_status = obtain(ans[0])
        else:
            req_to_out_status = ans[0]

        self._update_out_status_to_batch(batch, req_to_out_status)
        await self.push_load_to_controller()
        unfinished_req_ids, finished_req_ids = batch.mark_and_get_finished_req_and_preupdate_status()
        self._send_to_detokenization_proc(batch, req_to_out_status)
        batch.filter_out_finished_req(unfinished_req_ids, finished_req_ids)
        await self._handle_finish_req(batch, unfinished_req_ids, finished_req_ids)
        self.metric_client.histogram_observe(
            "lightllm_batch_inference_duration_bucket", time.time() - start_time, "decode"
        )
        return

    async def _filter_batch(self, batch: Batch, unfinished_req_ids, finished_req_ids: List):
        rets = [
            self.model_rpcs[tp_rank].filter_batch(batch.batch_id, unfinished_req_ids, finished_req_ids)
            for tp_rank in range(self.local_world_size)
        ]
        await asyncio.gather(*rets)
        return

    async def _merge_batch(self, batch1, batch2):
        rets = [
            self.model_rpcs[tp_rank].merge_batch(batch1.batch_id, batch2.batch_id)
            for tp_rank in range(self.local_world_size)
        ]
        await asyncio.gather(*rets)
        return

    async def _remove_batch(self, batch):
        rets = [self.model_rpcs[tp_rank].remove_batch(batch.batch_id) for tp_rank in range(self.local_world_size)]
        await asyncio.gather(*rets)
        return

    async def _pause_reqs(self, batch: Batch, pasue_reqs):
        pasue_reqs_info = [(r.request_id, r.req_status) for r in pasue_reqs]
        rets = [
            self.model_rpcs[tp_rank].pause_reqs(batch.batch_id, pasue_reqs_info)
            for tp_rank in range(self.local_world_size)
        ]
        await asyncio.gather(*rets)
        return

    async def _handle_finish_req(self, batch: Batch, unfinished_req_ids, finished_req_ids):
        if len(finished_req_ids) != 0:
            if batch.is_clear():
                await self._remove_batch(batch)
            else:
                await self._filter_batch(batch, unfinished_req_ids, finished_req_ids)
        return

    def _filter_runing_batch(self):
        if self.running_batch is not None and self.running_batch.is_clear():
            self.running_batch = None
            return

    def _update_init_status_to_batch(self, batch: Batch, req_to_req_status):
        self._update_out_status_to_batch(batch, req_to_req_status)
        return

    def _update_out_status_to_batch(self, batch: Batch, req_to_out_status):
        new_batch_decode_need_tokens = 0  # 只有在 splitfuse 模式下有意义
        for req_id, (
            req_status,
            cur_kv_len,
            cur_output_len,
            token_info_list,
            finish_status_value,
            extral_info,
        ) in req_to_out_status.items():
            req: Req = batch.id_to_reqs[req_id]
            req.req_status = req_status
            req.cur_kv_len = cur_kv_len
            req.cur_output_len = cur_output_len
            # 暂时不维护 output_ids 和 output_metadata_list
            # for (new_token_id, new_gen_metadata) in token_info_list:
            #     req.output_ids.append(new_token_id)
            #     req.output_metadata_list.append(new_gen_metadata)
            # 当没有被 aborted 的时候，才更新请求状态。
            if not req.finish_status.is_aborted():
                req.finish_status = FinishStatus(finish_status_value)
            new_batch_decode_need_tokens += req.get_decode_need_tokens()

        batch.batch_decode_need_tokens = new_batch_decode_need_tokens
        return

    def _can_decode(self, batch: Batch):
        return batch.batch_decode_need_tokens + self.get_used_tokens() <= self.max_total_token_num

    def _send_to_detokenization_proc(self, batch: Batch, req_ans):
        batch_out = BatchTokenIdOut()
        for req_id, (_, _, _, token_info_list, _, _) in req_ans.items():
            req = batch.id_to_reqs[req_id]
            for idx, (new_token_id, new_gen_metadata) in enumerate(token_info_list):
                # req.finish_status 传输 value值 不传送对象，可以减少序列化对象的大小。
                new_gen_metadata["prompt_cache_len"] = req.prompt_cache_len
                if idx == len(token_info_list) - 1:
                    batch_out.reqs_infs.append((req_id, new_token_id, new_gen_metadata, req.finish_status.value))
                else:
                    batch_out.reqs_infs.append((req_id, new_token_id, new_gen_metadata, FinishStatus.NO_FINISH))
        self.send_to_detokenization.send_pyobj(batch_out)
        return

    def get_used_tokens(self):
        if self.args.use_dynamic_prompt_cache:
            return (
                self.max_total_token_num
                - self.shared_can_use_token_num.get_value()
                - self.radix_cache_client.get_unrefed_tokens_num()
            )
        else:
            return self.max_total_token_num - self.shared_can_use_token_num.get_value()

    def _commit_req(self, data: SPDPreCommitReq):
        try:
            self.commit_lock.acquire(blocking=True, timeout=10)
            self.commit_history[data.commit_id] = data.total_tokens
            self.commit_amount += data.total_tokens
        except Exception as e:
            logger.error(f"Error at commit req {data.commit_id} {data.total_tokens} {e}")
            raise e
        finally:
            self.commit_lock.release()

    def _uncommit_req(self, event_id):
        try:
            self.commit_lock.acquire(blocking=True, timeout=10)
            needed_token = self.commit_history.pop(event_id)
            self.commit_amount -= needed_token
        except Exception as e:
            raise e
        finally:
            self.commit_lock.release()

    def _can_commit(self):
        used = (
            math.floor(
                self.shared_token_load.get_dynamic_max_load() * self.max_total_token_num
                + self.get_used_tokens()
                - (0 if self.running_batch is None else self.running_batch.total_tokens())
            )
            + self.commit_amount
        )
        return used <= 0.999 * self.max_total_token_num

    def _dynamic_max_load(self):
        r"""
        used tokens = (prompt + generated tokens) of running batch + transferred commit tokens
        dyna max of running batch = sum of (max - generated)
        dyna max of commit = max
        """
        used_tokens = self.get_used_tokens()
        if self.running_batch is None:
            running_batch_dynamic_max = 0
        else:
            total = 0
            for req in self.running_batch.reqs:
                total += req.sample_params.max_new_tokens - req.cur_output_len + 1
            running_batch_dynamic_max = total
        commit_dynamic_max = self.commit_amount
        return used_tokens + running_batch_dynamic_max + commit_dynamic_max

    async def task_loop(self):
        if self.dist_mode == "decode":
            await self._decode_communicating_with_decode_scheduler_loop()
        elif self.dist_mode == "prefill":
            await self._loop_for_netio_req()

    async def _init_async_resources(self):
        if self.dist_mode == "prefill":
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600))
        await self.push_load_to_controller()

    async def _decode_communicating_with_decode_scheduler_loop(self):
        data = {}
        while True:
            data = await self.recv_from_decode_scheduler.recv_pyobj()
            if isinstance(data, SPDPreCommitReq):
                if self._dynamic_max_load() + data.total_tokens <= self.max_total_token_num:
                    self._commit_req(data)
                    self.send_to_decode_scheduler.send_pyobj(True)
                else:
                    self.send_to_decode_scheduler.send_pyobj(False)
            elif isinstance(data, SPDCommitReq):
                self.send_to_decode_scheduler.send_pyobj(True)
                commit_id = data.commit_id
                source_instance = data.source_instance
                assert source_instance >= 0, f"req from {source_instance}"
                tasks = []
                for idx, rpc in enumerate(self.model_rpcs):
                    tasks.append(
                        rpc.recv_request(
                            data.req_info,
                            source_instance * self.local_world_size + idx,
                        )
                    )
                await asyncio.gather(*tasks)
                await self.recv_req(data.req_info, commit_id)
                self._uncommit_req(str(commit_id))
            else:
                raise ValueError(f"Error data type {data} {type(data)}")

    async def _loop_for_netio_req(self):
        while True:
            recv_req = await self.recv_from_httpserver.recv_pyobj()
            if isinstance(recv_req, tuple) and len(recv_req) == 5:
                prompt_ids, sampling_params, multimodal_params, group_req_id, start_time = recv_req
                self.add_req(prompt_ids, sampling_params, multimodal_params, group_req_id, start_time)
            elif isinstance(recv_req, AbortReq):
                abort_req = recv_req
                group_req_id = abort_req.group_req_id
                await self.abort(group_req_id)
                self.send_to_detokenization.send_pyobj(abort_req)
            else:
                assert False, f"Error Req Inf {recv_req}"

    def clean_up(self):
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.kill()
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.join()
        return


def start_router_process(
    args,
    controller_to_router_url,
    router_to_detokenization_url,
    model_rpc_ports,
    metric_url,
    controller_write_feedback_from_router_url,
    decode_scheduler_url,
    controller_read_load_from_router_url,
    pipe_writer,
):
    # 注册graceful 退出的处理
    from lightllm.utils.graceful_utils import graceful_registry
    import inspect
    import setproctitle

    dist_mode = getattr(args, "dist_mode", "normal")
    model_instance_id = args.model_instance_id
    setproctitle.setproctitle(f"lightllm:router:{dist_mode}:{model_instance_id}")

    graceful_registry(inspect.currentframe().f_code.co_name)

    try:
        router = RouterManager(
            args,
            controller_to_router_url,
            router_to_detokenization_url,
            model_rpc_ports,
            metric_url,
            controller_write_feedback_from_router_url,
            decode_scheduler_url,
            controller_read_load_from_router_url,
        )

        asyncio.run(router.wait_to_model_ready())
    except:
        import traceback
        import sys

        etype, evalue, tb = sys.exc_info()
        err_str = "\n".join(traceback.format_exception(etype, evalue, tb))
        pipe_writer.send(err_str)
        router.clean_up()
        raise

    pipe_writer.send("init ok")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(router._init_async_resources())
    loop.create_task(router.loop_for_fwd())
    loop.run_until_complete(router.task_loop())
    return
