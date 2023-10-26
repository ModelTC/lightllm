import asyncio
import rpyc
import torch
import traceback
import numpy as np
from datetime import timedelta
from typing import Dict, List, Tuple
from transformers.configuration_utils import PretrainedConfig
from lightllm.server.router.model_infer.infer_batch import InferBatch
from rpyc.utils.classic import obtain

from lightllm.models.bloom.model import BloomTpPartModel
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.llama_quantized.model import LlamaTpPartModelQuantized
from lightllm.models.llama_ppl.model import LlamaPPlTpPartModel
from lightllm.models.llama2_ppl.model import Llama2PPlTpPartModel
from lightllm.models.llama2.model import Llama2TpPartModel
from lightllm.models.starcoder.model import StarcoderTpPartModel
from lightllm.models.starcoder_ppl.model import StarcoderPPlTpPartModel
from lightllm.models.qwen.model import QWenTpPartModel
from lightllm.models.baichuan7b.model import Baichuan7bTpPartModel
from lightllm.models.baichuan13b.model import Baichuan13bTpPartModel
from lightllm.models.chatglm2.model import ChatGlm2TpPartModel
from lightllm.models.internlm.model import InternlmTpPartModel
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.common.configs.config import setting
from .post_process import sample
from .infer_batch import requests_mapping

class ModelRpcServer(rpyc.Service):

    def exposed_init_model(self, rank_id, world_size, weight_dir, max_total_token_num, load_way, mode):
        import torch
        import torch.distributed as dist
        if world_size != 1:
            trans_list = [obtain(e) for e in (rank_id, world_size, weight_dir, max_total_token_num, load_way, mode)]
            rank_id, world_size, weight_dir, max_total_token_num, load_way, mode = trans_list

        self.tp_rank = rank_id
        self.world_size = world_size
        self.load_way = load_way
        self.mode = mode
        self.cache = {}

        dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{setting["nccl_port"]}', rank=rank_id, world_size=world_size)
        torch.cuda.set_device(rank_id)

        model_cfg, _ = PretrainedConfig.get_config_dict(
            weight_dir
        )

        try:
            self.model_type = model_cfg["model_type"]
            if self.model_type == "bloom":
                self.model = BloomTpPartModel(rank_id, world_size, weight_dir, max_total_token_num, load_way, mode)
            elif self.model_type == "llama":
                if "num_key_value_heads" in model_cfg.keys():
                    if "ppl" not in mode:
                        self.model = Llama2TpPartModel(rank_id, world_size, weight_dir, max_total_token_num, load_way, mode)
                    else:
                        self.model = Llama2PPlTpPartModel(rank_id, world_size, weight_dir, max_total_token_num, load_way, mode)
                else:
                    if "ppl" not in mode:
                        if 'int8weight' in mode or 'int4weight' in mode:
                            self.model = LlamaTpPartModelQuantized(rank_id, world_size, weight_dir, max_total_token_num, load_way, mode)
                        else:
                            self.model = LlamaTpPartModel(rank_id, world_size, weight_dir, max_total_token_num, load_way, mode)
                    else:
                        self.model = LlamaPPlTpPartModel(rank_id, world_size, weight_dir, max_total_token_num, load_way, mode)
            elif self.model_type == "qwen":
                self.model = QWenTpPartModel(rank_id, world_size, weight_dir, max_total_token_num, load_way, mode)
            elif self.model_type == "baichuan":
                if model_cfg['hidden_size'] == 4096:
                    self.model = Baichuan7bTpPartModel(rank_id, world_size, weight_dir, max_total_token_num, load_way, mode)
                elif model_cfg["hidden_size"] == 5120:
                    self.model = Baichuan13bTpPartModel(rank_id, world_size, weight_dir, max_total_token_num, load_way, mode)
                else:
                    raise Exception('can not support baichuan format')
            elif self.model_type == 'gpt_bigcode':
                if "ppl" not in mode:
                    self.model = StarcoderTpPartModel(rank_id, world_size, weight_dir, max_total_token_num, load_way, mode)
                else:
                    self.model = StarcoderPPlTpPartModel(rank_id, world_size, weight_dir, max_total_token_num, load_way, mode)
            elif self.model_type == 'chatglm':
                self.model = ChatGlm2TpPartModel(rank_id, world_size, weight_dir, max_total_token_num, load_way, mode)
            elif self.model_type == 'internlm':
                self.model = InternlmTpPartModel(rank_id, world_size, weight_dir, max_total_token_num, load_way, mode)
            else:
                raise Exception(f"can not support {self.model_type} now")
        except Exception as e:
            print("#" * 16)
            print("load model error:", str(e), e, type(e))
            raise e
        
        set_random_seed(2147483647)
        return
    # @calculate_time(show=True, min_cost_ms=0.1)
    def exposed_add_batch(self, batch_id, reqs, dtype):
        if self.world_size != 1:
            batch_id, reqs, dtype = obtain(batch_id), obtain(reqs), obtain(dtype)
        import torch
        if dtype == "fp16":
            dtype = torch.float16
        else:
            assert False, "error dtype"
        batch_data = InferBatch.init_batch(batch_id, reqs, dtype, torch.cuda.current_device(), self.model.req_manager, self.model.vocab_size)
        self.cache[batch_id] = batch_data
        return
    
    @calculate_time(show=False, min_cost_ms=300)
    def exposed_prefill_batch(self, batch_id):
        return self.forward(batch_id, is_prefill=True)

    @calculate_time(show=True, min_cost_ms=200)
    def exposed_decode_batch(self, batch_id):
        return self.forward(batch_id, is_prefill=False)

    # @calculate_time(show=True, min_cost_ms=0.1)
    def exposed_filter_batch(self, batch_id, req_id_list, finished_req_id_list):
        if self.world_size != 1:
            batch_id, req_id_list, finished_req_id_list = obtain(batch_id), obtain(req_id_list), obtain(finished_req_id_list)
        # print("filter old size:", len(batch.reqs), "new size:", len(req_id_list))
        batch = self.cache.pop(batch_id)
        filter_batch = batch.filter(req_id_list, finished_req_id_list)
        del batch
        self.cache[batch_id] = filter_batch
        return

    def exposed_restore_reqs(self, batch_id, req_id_list):
        if self.world_size != 1:
            batch_id, req_id_list = obtain(batch_id), obtain(req_id_list)
        batch1 = self.cache.pop(batch_id)
        batch2 = batch1.restore_reqs(req_id_list)
        self.cache[batch_id] = batch2
        del batch1
        return

    def exposed_stop_reqs(self, batch_id, req_id_list):
        if self.world_size != 1:
            batch_id, req_id_list = obtain(batch_id), obtain(req_id_list)
        batch1 = self.cache.pop(batch_id)
        batch2 = batch1.stop_reqs(req_id_list)
        self.cache[batch_id] = batch2
        del batch1
        return

    # @calculate_time(show=True, min_cost_ms=0.1)
    def exposed_merge_batch(self, batch_id1, batch_id2):
        batch1 = self.cache.pop(batch_id1)
        batch2 = self.cache.pop(batch_id2)
        m_batch = InferBatch.merge(batch1, batch2)
        del batch1
        del batch2
        self.cache[batch_id1] = m_batch
        return

    # @calculate_time(show=True, min_cost_ms=10)
    def exposed_remove_batch(self, batch_id):
        batch = self.cache.pop(batch_id)
        batch.free_self()
        del batch
        # torch.cuda.empty_cache()
        return
    
    def prepare_inputs(self, batch, is_prefill):
        nopad_total_token_num = 0
        nopad_max_len_in_batch = 0
        b_start_loc = 0
        input_ids = []
        nopad_b_req_idx = []
        nopad_b_start_loc = []
        nopad_b_seq_len = []
        for request_id in batch.request_ids:
            req = requests_mapping[request_id]
            nopad_total_token_num += req.b_seq_len
            nopad_max_len_in_batch = max(nopad_max_len_in_batch, req.b_seq_len)
            nopad_b_req_idx.append(req.b_req_idx)
            nopad_b_start_loc.append(b_start_loc)
            if req.offload:
                nopad_b_seq_len.append(req.prompt_len)
                input_ids.append(req.prompt_token_ids)
            else:
                nopad_b_seq_len.append(req.b_seq_len)
                input_ids.append(req.input_id)
            b_start_loc += req.b_seq_len
        if is_prefill:
            if len(input_ids) > 1:
                input_ids = np.concatenate(input_ids, dtype=np.int64)
            else:
                input_ids = input_ids[0]
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device='cuda')
        nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device='cuda')
        nopad_b_start_loc = torch.tensor(nopad_b_start_loc, dtype=torch.int32, device='cuda')
        nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device='cuda')
        kwargs = {
            "batch_size": len(batch),
            "total_token_num": nopad_total_token_num,
            "max_len_in_batch": nopad_max_len_in_batch,
            "input_ids": input_ids,
            "b_loc": batch.req_manager.req_to_token_indexs,
            "b_loc_idx": nopad_b_req_idx,
            "b_start_loc": nopad_b_start_loc,
            "b_seq_len": nopad_b_seq_len,
            "is_prefill": is_prefill            
        }
        return kwargs

    # @calculate_time(show=True, min_cost_ms=150)
    def forward(self, batch_id, is_prefill):
        batch: InferBatch = self.cache.pop(batch_id)
        kwargs = self.prepare_inputs(batch, is_prefill)

        logits = self.model.forward(**kwargs)
        next_token_ids, next_token_probs = sample(logits, batch)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
        output_dict = {}
        new_input_ids = []    
        for i, (r, all_input_ids, next_token_id, next_token_logprob) in enumerate(zip(batch.requests, batch.all_input_ids, next_token_ids, next_token_logprobs)):
            # all_input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long, device="cuda")
            all_input_ids.append(int(next_token_id))
            # all_input_ids_tensor = None
            new_input_ids.append(next_token_id)
            requests_mapping[r].out_token_id_count[next_token_id] += 1
            if not requests_mapping[r].offload:
                requests_mapping[r].input_id = next_token_id
                requests_mapping[r].b_seq_len += 1
            else:
                requests_mapping[r].offload = False
            metadata = {
                'id': int(next_token_id),
                'logprob': float(next_token_logprob),
            }
            output_dict[r] = (int(next_token_id), metadata)
        
        self.cache[batch.batch_id] = batch
        return output_dict


class ModelRpcClient:
    def __init__(self, model_rpc, world_size, rpc_server_process=None):
        self.model: ModelRpcServer = model_rpc
        self.world_size = world_size
        self.rpc_server_process = rpc_server_process
        self.use_rpc = self.world_size != 1
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
            self._add_batch = async_wrap(self.model.add_batch)
            self._prefill_batch = async_wrap(self.model.prefill_batch)
            self._decode_batch = async_wrap(self.model.decode_batch)
            self._stop_reqs = async_wrap(self.model.stop_reqs)
            self._restore_reqs = async_wrap(self.model.restore_reqs)
            self._filter_batch = async_wrap(self.model.filter_batch)
            self._merge_batch = async_wrap(self.model.merge_batch)
            self._remove_batch = async_wrap(self.model.remove_batch)
        else:
            self._init_model = self.model.exposed_init_model
            self._add_batch = self.model.exposed_add_batch
            self._prefill_batch = self.model.exposed_prefill_batch
            self._decode_batch = self.model.exposed_decode_batch
            self._stop_reqs = self.model.exposed_stop_reqs
            self._restore_reqs = self.model.exposed_restore_reqs
            self._filter_batch = self.model.exposed_filter_batch
            self._merge_batch = self.model.exposed_merge_batch
            self._remove_batch = self.model.exposed_remove_batch
        return

    async def init_model(self, rank_id, world_size, weight_dir, max_total_token_num, load_way, mode):
        ans : rpyc.AsyncResult = self._init_model(rank_id, world_size, weight_dir, max_total_token_num, load_way, mode)
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def init_batch(self, batch_id, reqs):
        ans = self._add_batch(batch_id, reqs, "fp16")
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def prefill_batch(self, batch_id):
        ans = self._prefill_batch(batch_id)
        if self.use_rpc:
            return await ans
        else:
            return ans

    async def decode_batch(self, batch_id):
        ans = self._decode_batch(batch_id)
        if self.use_rpc:
            return await ans
        else:
            return ans

    async def filter_batch(self, batch_id, req_id_list, finished_req_id_list):
        ans = self._filter_batch(batch_id, req_id_list, finished_req_id_list)
        if self.use_rpc:
            await ans
            return
        else:
            return 

    async def stop_reqs(self, batch_id, req_id_list):
        ans = self._stop_reqs(batch_id, req_id_list)
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def restore_reqs(self, batch_id, req_id_list):
        ans = self._restore_reqs(batch_id, req_id_list)
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def merge_batch(self, batch_id1, batch_id2):
        ans = self._merge_batch(batch_id1, batch_id2)
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def remove_batch(self, batch_id):
        ans = self._remove_batch(batch_id)
        if self.use_rpc:
            await ans
            return
        else:
            return


def _init_env(port):
    from rpyc.utils.server import ThreadedServer
    t = ThreadedServer(ModelRpcServer(), port=port, protocol_config={"allow_pickle": True})
    t.start()
    return


async def start_model_process(port, world_size):
    # 单卡时不使用 rpc
    if world_size == 1:
        return ModelRpcClient(ModelRpcServer(), world_size)
    
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
    return ModelRpcClient(con.root, world_size, rpc_server_process=proc)
