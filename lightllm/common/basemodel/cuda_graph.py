import os
import torch
import copy
import bisect
from typing import Optional
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.distributed import dist_group_manager, lightllm_capture_graph, CustomProcessGroup
from lightllm.common.basemodel.microbatch_overlap_objs import DecodeMicroBatch
from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.common.spec_info import SpeculativeDecodeAlgorithm
from .infer_struct import InferStateInfo


logger = init_logger(__name__)


class CudaGraph:
    # CudaGraph forward pass for the decoding stage.

    def __init__(self, max_batch_size=8, max_len_in_batch=8192):
        self.graph = {}
        self.mempool = torch.cuda.graph_pool_handle() if torch.cuda.is_available() else None
        self.max_batch_size = max_batch_size
        self.graph_max_len_in_batch = max_len_in_batch
        self.args = get_env_start_args()
        self.enable_decode_microbatch_overlap = self.args.enable_decode_microbatch_overlap

        # gen cuda graph batch_sizes
        # cuda graph gen for batch size = [1, 2, 3, ..., graph_split_batch_size]
        # and [graph_split_batch_size + graph_grow_step_size,
        # graph_split_batch_size + 2 * graph_grow_step_size,  ...,  self.max_batch_size]
        graph_split_batch_size = self.args.graph_split_batch_size
        max_batch_size = self.max_batch_size
        graph_grow_step_size = self.args.graph_grow_step_size

        batch_sizes = [i for i in range(1, min(graph_split_batch_size, max_batch_size) + 1)]
        for _batch_size in range(
            graph_split_batch_size + graph_grow_step_size, max_batch_size + 1, graph_grow_step_size
        ):
            batch_sizes.append(_batch_size)
        self.cuda_graph_batch_sizes = batch_sizes
        assert batch_sizes[-1] == self.max_batch_size
        logger.info(f"cuda graph batch_sizes: {self.cuda_graph_batch_sizes}")

    def can_run(self, batch_size, max_len_in_batch):
        return batch_size <= self.max_batch_size and max_len_in_batch <= self.graph_max_len_in_batch

    def need_capture(self, batch_size):
        find_batch_size = self._find_closest_graph_batch_size(batch_size)
        if find_batch_size is not None:
            return find_batch_size not in self.graph
        else:
            assert False, "dead code"

    def _find_closest_graph_batch_size(self, batch_size):
        index = bisect.bisect_left(self.cuda_graph_batch_sizes, batch_size)
        if index < len(self.cuda_graph_batch_sizes):
            find_batch_size = self.cuda_graph_batch_sizes[index]
            return find_batch_size
        else:
            return None

    def _capture_decode(self, decode_func, input_ids: torch.Tensor, infer_state: InferStateInfo):
        dist_group: CustomProcessGroup = infer_state.dist_group
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
                    predict_logics, predict_logics1 = decode_func(input_ids, infer_state, input_ids1, infer_state1)
        self.graph[batch_size] = (
            graph_obj,
            input_ids,
            infer_state,
            input_ids1,
            infer_state1,
            predict_logics,
            predict_logics1,
        )
        graph_obj.replay()
        return predict_logics, predict_logics1

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
        find_batch_size = self._find_closest_graph_batch_size(batch_size)
        need_mask_fill = batch_size < find_batch_size
        graph_obj, graph_input_ids, graph_infer_state, graph_output = self.graph[find_batch_size]
        graph_input_ids[0:batch_size].copy_(input_ids)
        # fill a valid token_id
        if need_mask_fill:
            graph_input_ids[batch_size:].fill_(1)
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
        find_batch_size = self._find_closest_graph_batch_size(batch_size)
        need_mask_fill = batch_size < find_batch_size
        (
            graph_obj,
            graph_input_ids,
            graph_infer_state,
            graph_input_ids1,
            graph_infer_state1,
            graph_predict_logics,
            graph_predict_logics1,
        ) = self.graph[find_batch_size]
        graph_input_ids[0:batch_size].copy_(input_ids)
        if need_mask_fill:
            graph_input_ids[batch_size:].fill_(1)
        graph_infer_state.copy_for_cuda_graph(infer_state)
        graph_input_ids1[0:batch_size].copy_(input_ids1)
        if need_mask_fill:
            graph_input_ids1[batch_size:].fill_(1)
        graph_infer_state1.copy_for_cuda_graph(infer_state1)
        graph_obj.replay()
        return graph_predict_logics, graph_predict_logics1

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

        # prefill init padding req.
        predict_id = self._warmup_prefill(model)

        # decode cuda graph init
        for batch_size in self.cuda_graph_batch_sizes[::-1]:
            seq_len = 2
            total_token_num = batch_size * seq_len
            max_len_in_batch = self.graph_max_len_in_batch
            input_ids = torch.tensor([predict_id for _ in range(batch_size)], dtype=torch.int32, device="cuda")
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

        predict_id = self._warmup_prefill(model)

        for batch_size in self.cuda_graph_batch_sizes[::-1]:
            decode_batches = []
            for micro_batch_index in [0, 1]:
                # dummy decoding, capture the cudagraph
                seq_len = 2
                total_token_num = batch_size * seq_len
                max_len_in_batch = self.graph_max_len_in_batch
                input_ids = torch.tensor([predict_id for _ in range(batch_size)], dtype=torch.int32, device="cuda")
                mem_indexes = model.mem_manager.alloc(len(input_ids)).cuda()
                b_req_idx = torch.tensor(
                    [model.req_manager.HOLD_REQUEST_ID for _ in range(batch_size)], dtype=torch.int32, device="cuda"
                )
                b_seq_len = torch.empty(batch_size, dtype=torch.int32, device="cuda")
                b_seq_len.fill_(seq_len)

                micro_batch = ModelInput(
                    batch_size=batch_size,
                    total_token_num=total_token_num,
                    max_len_in_batch=max_len_in_batch,
                    input_ids=input_ids,
                    mem_indexes=mem_indexes,
                    b_req_idx=b_req_idx,
                    b_seq_len=b_seq_len,
                )
                decode_batches.append(micro_batch)

                for var_name, var_value in list(locals().items()):
                    if isinstance(var_value, torch.Tensor):
                        del locals()[var_name]
                torch.cuda.empty_cache()

            _, _ = model.microbatch_overlap_decode(decode_batches[0], decode_batches[1])

            model.mem_manager.free_all()
            model.req_manager.free_all()

            # release local tensors
            for var_name, var_value in list(locals().items()):
                if isinstance(var_value, torch.Tensor):
                    del locals()[var_name]
            torch.cuda.empty_cache()

        logger.info(
            f"Capture overlap cudagraph success, batch_size <={self.max_batch_size} "
            f"and max_len_in_batch <= {self.graph_max_len_in_batch} will infer with cudagraph."
        )

    def _warmup_prefill(self, model) -> int:
        from .basemodel import TpPartBaseModel

        model: TpPartBaseModel = model

        # prefill init padding req.
        prefill_input_len = 1
        batch_size = 1
        dummy_input_ids = torch.ones((batch_size,), dtype=torch.int32, device="cuda")
        b_req_idx = torch.tensor(
            [model.req_manager.HOLD_REQUEST_ID for _ in range(batch_size)], dtype=torch.int32, device="cuda"
        )
        mem_indexes = torch.tensor(
            [model.mem_manager.HOLD_TOKEN_MEMINDEX for _ in range(batch_size)], dtype=torch.int32, device="cuda"
        )
        b_seq_len = torch.ones(batch_size, dtype=torch.int32, device="cuda")
        b_ready_cache_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        total_token_num = prefill_input_len * batch_size
        dummy_hidden_states = None
        if model.spec_algo.is_mtp_module():
            dummy_hidden_states = torch.randn(
                total_token_num, model.config["hidden_size"], dtype=model.data_type, device="cuda"
            )
        model_input = ModelInput(
            batch_size=batch_size,
            total_token_num=total_token_num,
            max_len_in_batch=prefill_input_len,
            input_ids=dummy_input_ids,
            mem_indexes=mem_indexes,
            b_req_idx=b_req_idx,
            b_seq_len=b_seq_len,
            b_ready_cache_len=b_ready_cache_len,
            is_prefill=True,
            multimodal_params=[],
            hidden_states=dummy_hidden_states,
        )

        model_output: ModelOutput = model.forward(model_input)
        del dummy_input_ids
        del b_req_idx
        del mem_indexes
        del b_seq_len
        del b_ready_cache_len
        prob_out = torch.softmax(model_output.logits, dim=-1)
        del model_output
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        del prob_out
        predict_ids = predict_ids.detach().cpu().numpy()
        predict_id = int(predict_ids[0][0])
        torch.cuda.empty_cache()
        return predict_id
