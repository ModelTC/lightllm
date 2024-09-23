import os
import torch


class CudaGraph:
    # CudaGraph forward pass for the decoding stage.

    def __init__(self, max_batch_size=16, max_len_in_batch=8192):
        self.graph = {}
        self.mempool = torch.cuda.graph_pool_handle() if torch.cuda.is_available() else None
        self.max_batch_size = max_batch_size
        # self.graph_max_len_in_batch = max_len_in_batch

    def need_capture(self, batch_size):
        return batch_size not in self.graph

    def capture_decode(self, decode_func, input_ids, infer_state):
        graph_obj = torch.cuda.CUDAGraph()
        batch_size = input_ids.shape[0]
        # infer_state.max_len_in_batch = self.graph_max_len_in_batch
        for _ in range(2):
            torch.cuda.synchronize()
            tmp_ = decode_func(input_ids, infer_state)
            torch.cuda.synchronize()
        with torch.cuda.graph(graph_obj, pool=self.mempool):
            predict_logics = decode_func(input_ids, infer_state)
        self.graph[batch_size] = (graph_obj, input_ids, infer_state, predict_logics)
        graph_obj.replay()
        return predict_logics
    
    def replay(self, input_ids, infer_state):
        batch_size = input_ids.shape[0]
        graph_obj, graph_input_ids, graph_infer_state, graph_predict_logics = self.graph[batch_size]
        graph_input_ids.copy_(input_ids)
        graph_infer_state.copy_(infer_state)
        graph_obj.replay()
        return graph_predict_logics