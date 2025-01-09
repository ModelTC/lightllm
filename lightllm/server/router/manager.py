import copy
import time
import uuid
import uvloop
import asyncio
import torch
import rpyc
import pickle

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import zmq
import zmq.asyncio
import torch.multiprocessing as mp
from typing import Dict, List, Optional
from .batch import Batch
from .model_infer.model_rpc import start_model_process, ModelRpcClient
from .req_queue import build_req_queue
from lightllm.utils.infer_utils import calculate_time
from lightllm.server.core.objs.io_objs import GroupReqIndexes
from lightllm.server.core.objs import ShmReqManager
from .dynamic_prompt.radix_cache import RadixCacheReadOnlyClient
from .stats import Stats
from .pause_strategy import Fcfs, select_paused_reqs
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.token_load import TokenLoad
from lightllm.server.metrics.manager import MetricClient
from lightllm.common.basemodel.infer_lock import g_router_lock
from lightllm.common.mem_manager import ReadOnlyStaticsMemoryManager

logger = init_logger(__name__)


class RouterManager:
    def __init__(self, args, router_port, detokenization_port, model_rpc_ports, metric_port):
        self.args = args
        self.model_weightdir = args.model_dir
        self.world_size = args.tp
        self.dp_size = args.dp
        self.load_way = args.load_way
        self.mode = args.mode
        self.max_total_token_num = args.max_total_token_num
        self.shm_req_manager = ShmReqManager()
        # 用共享内存进行共享，router 模块读取进行精确的调度估计
        self.read_only_statics_mem_manager = ReadOnlyStaticsMemoryManager(args.nccl_port, args.tp)
        # 初始化 radix_cache_client 用于读取 prompt cache 的管理信息
        self.radix_cache_client = None

        # 共享变量，用于存储router端调度分析得到的机器负载信息
        self.shared_token_load = TokenLoad(f"{str(args.nccl_port)}_shared_token_load", self.dp_size)
        for dp_index in range(self.dp_size):
            self.shared_token_load.set_estimated_peak_token_count(0, dp_index)
            self.shared_token_load.set_frozened_token_count(0, dp_index)
            self.shared_token_load.set_current_load(0.0, dp_index)
            self.shared_token_load.set_logical_max_load(0.0, dp_index)
            self.shared_token_load.set_dynamic_max_load(0.0, dp_index)

        self.pause_strategy = Fcfs()
        self.running_batch: Batch = None
        self.eos_id = args.eos_id
        self.has_wait_tokens = 0
        self.max_wait_tokens = args.router_max_wait_tokens

        context = zmq.asyncio.Context(2)
        self.recv_from_httpserver = context.socket(zmq.PULL)
        self.recv_from_httpserver.bind(f"{args.zmq_mode}127.0.0.1:{router_port}")

        self.send_to_detokenization = context.socket(zmq.PUSH)
        self.send_to_detokenization.connect(f"{args.zmq_mode}127.0.0.1:{detokenization_port}")
        self.model_rpc_ports = model_rpc_ports

        self.is_splitfuse_mode = args.splitfuse_mode
        self.is_token_healing = self.args.token_healing_mode
        self.splitfuse_block_size = args.splitfuse_block_size

        self.stats_tool = Stats(not args.disable_log_stats, args.log_stats_interval)
        self.metric_client = MetricClient(metric_port)
        self.is_pd_run_mode = self.args.run_mode in ["prefill", "decode"]
        # p d 分离模式下，需要调度锁来同步调度端和推理端的一些数据操作
        # 主要是为了防止调度失误，造成 OOM 等错误
        self.router_lock = mp.Lock()
        g_router_lock.obj = self.router_lock
        return

    async def wait_to_model_ready(self):
        # 初始化模型
        self.model_rpcs: List[ModelRpcClient] = []
        # 用于 kv move 管理进程 和 推理进程进行task信息的交互。
        self.info_queue: mp.Queue = mp.Queue()
        self.mem_queues: List[torch.multiprocessing.Queue] = [
            torch.multiprocessing.Queue() for _ in range(self.world_size)
        ]
        for rank_id in range(self.world_size):
            rpc_model = await start_model_process(
                args=self.args,
                port=self.model_rpc_ports[rank_id],
                world_size=self.world_size,
                info_queue=self.info_queue,
                mem_queue=self.mem_queues[rank_id],
                router_lock=self.router_lock,
            )
            self.model_rpcs.append(rpc_model)

        init_model_ret = []
        for rank_id in range(self.world_size):  # async init model process
            kvargs = {
                "args": self.args,
                "rank_id": rank_id,
                "world_size": self.world_size,
                "dp_size": self.dp_size,
                "weight_dir": self.model_weightdir,
                "load_way": self.load_way,
                "max_total_token_num": self.max_total_token_num,
                "mode": self.mode,
                "max_req_num": self.args.running_max_req_size + 8,
                "max_seq_length": self.args.max_req_total_len + 8,  # 留一点余量
                "nccl_port": self.args.nccl_port,
                "is_first_token_constraint_mode": self.args.first_token_constraint_mode,
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
                "quant_type": self.args.quant_type,
                "quant_cfg": self.args.quant_cfg,
                "pd_rpyc_port": self.args.pd_tp_infer_rpyc_ports[rank_id],  # 非 pd 模式可以不设置
            }
            init_model_ret.append(self.model_rpcs[rank_id].init_model(kvargs))

        await asyncio.gather(*init_model_ret)
        if self.max_total_token_num is None:
            self.max_total_token_num = await self.model_rpcs[0].get_max_total_token_num()
            self.args.max_total_token_num = self.max_total_token_num
        if self.args.use_dynamic_prompt_cache:
            self.radix_cache_client = RadixCacheReadOnlyClient(
                str(self.args.nccl_port), self.max_total_token_num, tp_size=self.world_size
            )
        self.req_queue = build_req_queue(self.args, self, self.dp_size)
        logger.info(f"use req queue {self.req_queue.__class__.__name__}")

        if self.args.run_mode == "prefill":
            # 启动 prefill kv move 管理进程
            from lightllm.server.router.model_infer.mode_backend.continues_batch.pd_mode.prefill_node_impl import (
                start_prefill_kv_move_manager_process,
            )

            start_prefill_kv_move_manager_process(self.args, self.info_queue, self.mem_queues)

        if self.args.run_mode == "decode":
            # 启动 decode kv move 管理进程
            from lightllm.server.router.model_infer.mode_backend.continues_batch.pd_mode.decode_node_impl import (
                start_decode_kv_move_manager_process,
            )

            start_decode_kv_move_manager_process(self.args, self.info_queue, self.mem_queues)

        return

    def add_req(self, group_req_indexes: GroupReqIndexes):
        req_group = []
        for req_index in group_req_indexes.shm_req_indexes:
            req = self.shm_req_manager.get_req_obj_by_index(req_index)
            req.multimodal_params = group_req_indexes.multimodal_params
            req.start_time = group_req_indexes.time_mark
            req_group.append(req)

        self.req_queue.extend(req_group)
        try:
            self.send_to_detokenization.send_pyobj(
                group_req_indexes,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        except BaseException as e:
            logger.exception(str(e))

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
                    for dp_index in range(self.dp_size):
                        token_ratio1 = self.get_used_tokens(dp_index) / self.max_total_token_num
                        token_ratio2 = (
                            self.max_total_token_num
                            - self.read_only_statics_mem_manager.get_unrefed_token_num(dp_index)
                        ) / self.max_total_token_num
                        d_i = dp_index
                        logger.debug(
                            f"dp_i {d_i} current batch size: {len(self.running_batch.reqs)} \n"
                            f"dp_i {d_i} paused req num: {self.req_queue.get_paused_req_num()} \n"
                            f"dp_i {d_i} token used ratio: {token_ratio1} not contain prompt cache tree unrefed token\n"
                            f"dp_i {d_i} token used ratio: {token_ratio2} contain prompt cache tree unrefed token"
                        )
                self.req_queue.update_token_load(self.running_batch, force_update=False)
                self.stats_tool.print_stats()
                self.metric_client.gauge_set("lightllm_batch_current_size", len(self.running_batch.reqs))
                self.metric_client.gauge_set("lightllm_batch_pause_size", self.req_queue.get_paused_req_num())
                self.metric_client.gauge_set("lightllm_queue_size", self.req_queue.get_wait_req_num())
                self.metric_client.gauge_set(
                    "lightllm_batch_current_max_tokens",
                    int(
                        sum(self.shared_token_load.get_dynamic_max_load(d_i) for d_i in range(self.dp_size))
                        * self.max_total_token_num
                    ),
                )
            else:
                self.req_queue.update_token_load(self.running_batch, force_update=True)
                if counter_count % 300 == 0:
                    self.metric_client.gauge_set("lightllm_batch_current_size", 0.0)
                    self.metric_client.gauge_set("lightllm_batch_pause_size", 0.0)
                    self.metric_client.gauge_set("lightllm_queue_size", 0.0)
                    self.metric_client.gauge_set("lightllm_batch_current_max_tokens", 0.0)

            if self.running_batch is None:
                await asyncio.sleep(0.01)  # 10ms

    async def _step(self):
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
                self.has_wait_tokens = self.max_wait_tokens
            return

        # 有运行请求，但是已经到了可以调度新的请求合并推理的时机
        if self.has_wait_tokens >= self.max_wait_tokens:
            new_mini_batch = self.req_queue.generate_new_batch(self.running_batch)
            self.has_wait_tokens = 0
            if new_mini_batch is not None:
                self.has_wait_tokens = self.max_wait_tokens
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
            logger.debug(f"pasued req num: {self.req_queue.get_paused_req_num()}")
            self.has_wait_tokens = 0
            return
        return

    async def _init_batch(self, batch: Batch):
        reqs = [r.to_router_rpc_obj() for r in batch.reqs]
        rets = [self.model_rpcs[tp_rank].init_batch(batch.batch_id, reqs) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        self._update_init_status_to_batch(batch)
        logger.debug(f"Init Batch: {batch.simple_log()} \n")
        return

    async def _prefill_batch(self, batch: Batch):
        start_time = time.time()
        self.metric_client.counter_inc("lightllm_batch_inference_count", "prefill")
        await self._init_batch(batch)
        if not self.is_splitfuse_mode:
            # 在 非 splitfuse 模式下，才需要真的执行 prefill 的操作。
            rets = [self.model_rpcs[tp_rank].prefill_batch(batch.batch_id) for tp_rank in range(self.world_size)]
            await asyncio.gather(*rets)
            self._update_out_status_to_batch(batch)
            unfinished_req_ids, finished_req_ids = batch.mark_and_get_finished_req_and_preupdate_status()
            self._send_to_detokenization_proc(batch)
            batch.filter_out_finished_req(unfinished_req_ids, finished_req_ids, self.shm_req_manager)
            await self._handle_finish_req(batch, unfinished_req_ids, finished_req_ids)
        self.metric_client.histogram_observe(
            "lightllm_batch_inference_duration_bucket", time.time() - start_time, "prefill"
        )
        return

    async def _decode_batch(self, batch: Batch):
        start_time = time.time()
        self.metric_client.counter_inc("lightllm_batch_inference_count", "decode")
        rets = [self.model_rpcs[tp_rank].decode_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)

        self._update_out_status_to_batch(batch)
        unfinished_req_ids, finished_req_ids = batch.mark_and_get_finished_req_and_preupdate_status()
        self._send_to_detokenization_proc(batch)
        batch.filter_out_finished_req(unfinished_req_ids, finished_req_ids, self.shm_req_manager)
        await self._handle_finish_req(batch, unfinished_req_ids, finished_req_ids)
        self.metric_client.histogram_observe(
            "lightllm_batch_inference_duration_bucket", time.time() - start_time, "decode"
        )
        return

    async def _filter_batch(self, batch: Batch, unfinished_req_ids, finished_req_ids: List):
        rets = [
            self.model_rpcs[tp_rank].filter_batch(batch.batch_id, unfinished_req_ids, finished_req_ids)
            for tp_rank in range(self.world_size)
        ]
        await asyncio.gather(*rets)
        return

    async def _merge_batch(self, batch1, batch2):
        rets = [
            self.model_rpcs[tp_rank].merge_batch(batch1.batch_id, batch2.batch_id) for tp_rank in range(self.world_size)
        ]
        await asyncio.gather(*rets)
        return

    async def _remove_batch(self, batch):
        rets = [self.model_rpcs[tp_rank].remove_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _pause_reqs(self, batch: Batch, pasue_reqs):
        pasue_reqs_info = [(r.request_id, r.req_status) for r in pasue_reqs]
        rets = [
            self.model_rpcs[tp_rank].pause_reqs(batch.batch_id, pasue_reqs_info) for tp_rank in range(self.world_size)
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

    def _update_init_status_to_batch(self, batch: Batch):
        self._update_out_status_to_batch(batch)
        return

    def _update_out_status_to_batch(self, batch: Batch):
        new_batch_decode_need_tokens = [0 for _ in range(self.dp_size)]  # 只有在 splitfuse 模式下有意义

        for req in batch.reqs:
            req_dp_index = req.sample_params.suggested_dp_index
            new_batch_decode_need_tokens[req_dp_index] += req.get_decode_need_tokens()

        batch.batch_decode_need_tokens = new_batch_decode_need_tokens
        return

    def _can_decode(self, batch: Batch):
        # p d 分离模式下，目前只能使用保守调度，保证请求放入进行decode的时候
        # 显存token肯定是够用的。
        # deepseekv2 dp 模式下,采用保守调度，也肯定够用
        if self.is_pd_run_mode or self.dp_size > 1:
            return True

        # 下面的判定条件，只在 dp 为 1 的情况下启用
        assert self.dp_size == 1
        return batch.batch_decode_need_tokens[0] + self.get_used_tokens(0) <= self.max_total_token_num

    def _send_to_detokenization_proc(self, batch: Batch):
        # 发个空包触发一下detokenization
        self.send_to_detokenization.send_pyobj(None, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def get_used_tokens(self, dp_index):
        if self.args.use_dynamic_prompt_cache:
            return (
                self.max_total_token_num
                - self.read_only_statics_mem_manager.get_unrefed_token_num(dp_index)
                - self.radix_cache_client.get_unrefed_tokens_num(dp_index)
            )
        else:
            return self.max_total_token_num - self.read_only_statics_mem_manager.get_unrefed_token_num(dp_index)

    async def loop_for_netio_req(self):
        while True:
            recv_req: GroupReqIndexes = await self.recv_from_httpserver.recv_pyobj()
            if isinstance(recv_req, GroupReqIndexes):
                self.add_req(recv_req)
            else:
                assert False, f"Error Req Inf {recv_req}"

    def clean_up(self):
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.kill()
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.join()
        return


def start_router_process(args, router_port, detokenization_port, model_rpc_ports, metric_port, pipe_writer):
    # 注册graceful 退出的处理
    from lightllm.utils.graceful_utils import graceful_registry
    import inspect

    graceful_registry(inspect.currentframe().f_code.co_name)

    try:
        router = RouterManager(
            args,
            router_port=router_port,
            detokenization_port=detokenization_port,
            model_rpc_ports=model_rpc_ports,
            metric_port=metric_port,
        )

        asyncio.run(router.wait_to_model_ready())
    except:
        import traceback
        import sys

        etype, evalue, tb = sys.exc_info()
        err_str = "\n".join(traceback.format_exception(etype, evalue, tb))
        logger.error(err_str)
        pipe_writer.send(err_str)
        router.clean_up()
        raise

    pipe_writer.send("init ok")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(router.loop_for_fwd())
    loop.run_until_complete(router.loop_for_netio_req())
    return
