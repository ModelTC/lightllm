import os
import torch
import copy
import bisect
from collections import OrderedDict
from typing import Optional
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.distributed import dist_group_manager, lightllm_capture_graph, CustomProcessGroup
from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from .infer_struct import InferStateInfo

logger = init_logger(__name__)


class CudaGraph:
    # CudaGraph forward pass for the decoding stage.

    def __init__(self, max_batch_size=8, max_len_in_batch=8192):
        self.graph = OrderedDict()  # for LRU

        self.mempool = torch.cuda.graph_pool_handle() if torch.cuda.is_available() else None
        self.max_batch_size = max_batch_size
        self.graph_max_len_in_batch = max_len_in_batch
        self.args = get_env_start_args()
        self.enable_decode_microbatch_overlap = self.args.enable_decode_microbatch_overlap
        self.max_graph_pool_size = self.args.max_graph_pool_size

        # gen cuda graph batch_sizes
        # cuda graph gen for batch size = [1, 2, 3, ..., graph_split_batch_size]
        # and [graph_split_batch_size + graph_grow_step_size,
        # graph_split_batch_size + 2 * graph_grow_step_size,  ...,  self.max_batch_size]
        graph_split_batch_size = self.args.graph_split_batch_size
        max_batch_size = self.max_batch_size
        graph_grow_step_size = self.args.graph_grow_step_size

        batch_sizes = [i for i in range(1, graph_split_batch_size + 1)]
        for _batch_size in range(graph_split_batch_size + graph_grow_step_size, max_batch_size, graph_grow_step_size):
            batch_sizes.append(_batch_size)

        batch_sizes = list(set([e for e in batch_sizes if e < max_batch_size]))
        batch_sizes.append(max_batch_size)
        batch_sizes.sort()

        self.cuda_graph_batch_sizes = batch_sizes
        assert batch_sizes[-1] == self.max_batch_size
        logger.info(f"cuda graph batch_sizes: {self.cuda_graph_batch_sizes}")

    def can_run(self, batch_size, max_len_in_batch):
        return batch_size <= self.max_batch_size and max_len_in_batch <= self.graph_max_len_in_batch

    def get_graph(self, batch_size):
        # We assume batch_size has already been adjusted to the closest supported graph batch size
        # If the graph already exists, get it and move it to the most recently used position.
        if batch_size in self.graph:
            find_graph = self.graph.pop(batch_size)  # Dequeue the graph
            self.graph[batch_size] = find_graph  # Enqueue the graph for LRU
            return find_graph
        else:
            return None

    def evict_oldest_graph(self):
        if self.graph:
            oldest_batch_size, oldest_graph = self.graph.popitem(last=False)
            del oldest_graph
            logger.info(f"Evicted CUDA graph for batch size: {oldest_batch_size}")

    def find_closest_graph_batch_size(self, batch_size):
        index = bisect.bisect_left(self.cuda_graph_batch_sizes, batch_size)
        if index < len(self.cuda_graph_batch_sizes):
            find_batch_size = self.cuda_graph_batch_sizes[index]
            return find_batch_size
        else:
            return None

    def _capture_decode(self, decode_func, input_ids: torch.Tensor, infer_state: InferStateInfo):
        dist_group: CustomProcessGroup = infer_state.dist_group
        if len(self.graph) >= self.max_graph_pool_size:
            self.evict_oldest_graph()

        graph_obj = torch.cuda.CUDAGraph()
        batch_size = input_ids.shape[0]
        infer_state.max_len_in_batch = self.graph_max_len_in_batch
        infer_state.total_token_num = self.graph_max_len_in_batch * batch_size
        # warmup
        # 因为有些推理过程的代码，会通过判断infer_state中是否存在某些属性来在一层上
        # 做一些初始化的操作，后续层可以复用这些计算的结果，如
        # lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding.py
        # 中做的一些操作，所以在 warmup 的时候，需要调用infer_state的copy函数做一个
        # 浅拷贝，不然后续传入到cuda graph捕获过程中后，infer_state因为提前拥有了这些属性，
        # 导致不会重新初始化，这样捕获过程中会不能捕获这些临时添加到 infer_state 管理对象
        # 中的 tensor。
        for _ in range(1):
            torch.cuda.synchronize()
            decode_func(input_ids, copy.copy(infer_state))
            torch.cuda.synchronize()

        with lightllm_capture_graph(dist_group):
            with torch.cuda.graph(graph_obj, pool=self.mempool):
                model_output = decode_func(input_ids, infer_state)
        # We assume batch_size has already been adjusted to the closest supported graph batch size
        self.graph[batch_size] = (graph_obj, input_ids, infer_state, model_output)
        graph_obj.replay()
        return model_output

    def _capture_decode_overlap(
        self,
        decode_func,
        input_ids: torch.Tensor,
        infer_state: InferStateInfo,
        input_ids1: torch.Tensor,
        infer_state1: InferStateInfo,
    ):
        dist_group: CustomProcessGroup = infer_state.dist_group
        if len(self.graph) >= self.max_graph_pool_size:
            self.evict_oldest_graph()

        dist_group1 = infer_state1.dist_group
        graph_obj = torch.cuda.CUDAGraph()
        batch_size = input_ids.shape[0]
        infer_state.max_len_in_batch = self.graph_max_len_in_batch
        infer_state.total_token_num = self.graph_max_len_in_batch * batch_size
        infer_state1.max_len_in_batch = self.graph_max_len_in_batch
        infer_state1.total_token_num = self.graph_max_len_in_batch * batch_size
        # warmup
        for _ in range(1):
            torch.cuda.synchronize()
            decode_func(input_ids, copy.copy(infer_state), input_ids1, copy.copy(infer_state1))
            torch.cuda.synchronize()
        with lightllm_capture_graph(dist_group1):
            with lightllm_capture_graph(dist_group):
                with torch.cuda.graph(graph_obj, pool=self.mempool):
                    model_output, model_output1 = decode_func(input_ids, infer_state, input_ids1, infer_state1)
        # We assume batch_size has already been adjusted to the closest supported graph batch size
        self.graph[batch_size] = (
            graph_obj,
            input_ids,
            infer_state,
            input_ids1,
            infer_state1,
            model_output,
            model_output1,
        )
        graph_obj.replay()
        return model_output, model_output1

    def capture_decode(
        self,
        decode_func,
        input_ids: torch.Tensor,
        infer_state: InferStateInfo,
        input_ids1: Optional[torch.Tensor] = None,
        infer_state1: Optional[torch.Tensor] = None,
    ):
        """
        Capture the cuda graph for the decoding stage.
        input_ids1 and infer_state1 is used for the overlap.
        """
        if self.enable_decode_microbatch_overlap:
            return self._capture_decode_overlap(decode_func, input_ids, infer_state, input_ids1, infer_state1)
        else:
            assert input_ids1 is None and infer_state1 is None
            return self._capture_decode(decode_func, input_ids, infer_state)

    def _replay(self, input_ids: torch.Tensor, infer_state: InferStateInfo):
        batch_size = input_ids.shape[0]
        graph_obj, graph_input_ids, graph_infer_state, graph_output = self.graph[batch_size]
        graph_input_ids.copy_(input_ids)
        graph_infer_state.copy_for_cuda_graph(infer_state)
        graph_obj.replay()
        return graph_output

    def _replay_overlap(
        self,
        input_ids: torch.Tensor,
        infer_state: InferStateInfo,
        input_ids1: torch.Tensor,
        infer_state1: InferStateInfo,
    ):
        batch_size = input_ids.shape[0]
        (
            graph_obj,
            graph_input_ids,
            graph_infer_state,
            graph_input_ids1,
            graph_infer_state1,
            graph_model_output,
            graph_model_output1,
        ) = self.graph[batch_size]
        graph_input_ids.copy_(input_ids)
        graph_infer_state.copy_for_cuda_graph(infer_state)
        graph_input_ids1.copy_(input_ids1)
        graph_infer_state1.copy_for_cuda_graph(infer_state1)
        graph_obj.replay()
        return graph_model_output, graph_model_output1

    def replay(self, input_ids, infer_state, input_ids1=None, infer_state1=None):
        if self.enable_decode_microbatch_overlap:
            return self._replay_overlap(input_ids, infer_state, input_ids1, infer_state1)
        else:
            assert input_ids1 is None and infer_state1 is None
            return self._replay(input_ids, infer_state)

    @torch.no_grad()
    def warmup(self, model):
        logger.info("Begin capture cudagraph, use the --disable_cudagraph to disable it.")
        # for typing easy
        from .basemodel import TpPartBaseModel

        model: TpPartBaseModel = model

        # decode cuda graph init
        for batch_size in self.cuda_graph_batch_sizes[::-1]:
            seq_len = 2
            total_token_num = batch_size * seq_len
            max_len_in_batch = self.graph_max_len_in_batch
            input_ids = torch.tensor([1 for _ in range(batch_size)], dtype=torch.int32, device="cuda")
            mem_indexes = model.mem_manager.alloc(len(input_ids)).cuda()
            b_req_idx = torch.tensor(
                [model.req_manager.HOLD_REQUEST_ID for _ in range(batch_size)], dtype=torch.int32, device="cuda"
            )
            b_seq_len = torch.empty(batch_size, dtype=torch.int32, device="cuda")
            b_seq_len.fill_(seq_len)

            model_input = ModelInput(
                batch_size=batch_size,
                total_token_num=total_token_num,
                max_len_in_batch=max_len_in_batch,
                input_ids=input_ids,
                mem_indexes=mem_indexes,
                b_req_idx=b_req_idx,
                b_seq_len=b_seq_len,
                is_prefill=False,
                **model._gen_special_model_input(batch_size),
            )
            model_output: ModelOutput = model.forward(model_input)
            del model_output
            del input_ids
            del mem_indexes
            del b_req_idx
            del b_seq_len

            model.mem_manager.free_all()
            model.req_manager.free_all()
            # release local tensors
            for var_name, var_value in list(locals().items()):
                if isinstance(var_value, torch.Tensor):
                    del locals()[var_name]
            torch.cuda.empty_cache()

        logger.info(
            f"Capture cudagraph success, batch_size <={self.max_batch_size} "
            f"and max_len_in_batch <= {self.graph_max_len_in_batch} will infer with cudagraph."
        )

    @torch.no_grad()
    def warmup_overlap(self, model):
        logger.info("Begin capture overlap cudagraph, use the --disable_cudagraph to disable it.")
        # for typing easy
        from .basemodel import TpPartBaseModel

        model: TpPartBaseModel = model

        for batch_size in self.cuda_graph_batch_sizes[::-1]:
            decode_batches = []
            for micro_batch_index in [0, 1]:
                # dummy decoding, capture the cudagraph
                seq_len = 2
                total_token_num = batch_size * seq_len
                max_len_in_batch = self.graph_max_len_in_batch
                input_ids = torch.tensor([1 for _ in range(batch_size)], dtype=torch.int32, device="cuda")
                mem_indexes = model.mem_manager.alloc(len(input_ids)).cuda()
                b_req_idx = torch.tensor(
                    [model.req_manager.HOLD_REQUEST_ID for _ in range(batch_size)], dtype=torch.int32, device="cuda"
                )
                b_seq_len = torch.empty(batch_size, dtype=torch.int32, device="cuda")
                b_seq_len.fill_(seq_len)

                micro_batch = ModelInput(
                    is_prefill=False,
                    batch_size=batch_size,
                    total_token_num=total_token_num,
                    max_len_in_batch=max_len_in_batch,
                    input_ids=input_ids,
                    mem_indexes=mem_indexes,
                    b_req_idx=b_req_idx,
                    b_seq_len=b_seq_len,
                    **model._gen_special_model_input(batch_size),
                )
                decode_batches.append(micro_batch)
                del micro_batch

                for var_name, var_value in list(locals().items()):
                    if isinstance(var_value, torch.Tensor):
                        del locals()[var_name]
                torch.cuda.empty_cache()

            _, _ = model.microbatch_overlap_decode(decode_batches[0], decode_batches[1])

            model.mem_manager.free_all()
            model.req_manager.free_all()

            del decode_batches

            # release local tensors
            for var_name, var_value in list(locals().items()):
                if isinstance(var_value, torch.Tensor):
                    del locals()[var_name]
            torch.cuda.empty_cache()

        logger.info(
            f"Capture overlap cudagraph success, batch_size <={self.max_batch_size} "
            f"and max_len_in_batch <= {self.graph_max_len_in_batch} will infer with cudagraph."
        )
