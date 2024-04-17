import torch
from transformers.configuration_utils import PretrainedConfig
from lightllm.models.mixtral.model import MixtralTpPartModel
from lightllm.models.qwen2.model import Qwen2TpPartModel
from lightllm.server.router.model_infer.infer_batch import InferBatch

from lightllm.models.bloom.model import BloomTpPartModel
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.llama_wquant.model import LlamaTpPartModelWQuant
from lightllm.models.llama_awquant.model import LlamaTpPartModelAWQuant
from lightllm.models.llama_quik.model import LlamaTpPartModelQuik
from lightllm.models.starcoder.model import StarcoderTpPartModel
from lightllm.models.starcoder_wquant.model import StarcoderTpPartModelWQuant
from lightllm.models.starcoder2.model import Starcoder2TpPartModel
from lightllm.models.qwen.model import QWenTpPartModel
from lightllm.models.qwen_wquant.model import QWenTpPartModelWQuant
from lightllm.models.baichuan7b.model import Baichuan7bTpPartModel
from lightllm.models.baichuan13b.model import Baichuan13bTpPartModel
from lightllm.models.baichuan2_7b.model import Baichuan2_7bTpPartModel
from lightllm.models.baichuan2_13b.model import Baichuan2_13bTpPartModel
from lightllm.models.chatglm2.model import ChatGlm2TpPartModel
from lightllm.models.internlm.model import InternlmTpPartModel
from lightllm.models.stablelm.model import StablelmTpPartModel
from lightllm.models.internlm2.model import Internlm2TpPartModel
from lightllm.models.internlm_wquant.model import InternlmTpPartModelWQuant
from lightllm.models.internlm2_wquant.model import Internlm2TpPartModelWQuant
from lightllm.models.yi.model import YiTpPartModel
from lightllm.models.mistral.model import MistralTpPartModel
from lightllm.models.minicpm.model import MiniCPMTpPartModel
from lightllm.models.llava.model import LlavaTpPartModel
from lightllm.models.qwen_vl.model import QWenVLTpPartModel
from lightllm.models.internlm_xcomposer.model import InternlmComposerTpPartModel
from lightllm.models.gemma_2b.model import Gemma_2bTpPartModel
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.infer_utils import calculate_time
from .pre_process import prepare_decode_inputs, prepare_prefill_inputs, splitfuse_prepare_decode_inputs
from .post_process import sample
from .infer_batch import requests_mapping
from .infer_batch import InferReq
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache


def load_llm_model(model_kvargs):
    weight_dir = model_kvargs["weight_dir"]
    mode = model_kvargs["mode"]
    model_cfg, _ = PretrainedConfig.get_config_dict(weight_dir)
    is_weight_only_quant = any("w6a16" in mode_ or "w8a16" in mode_ or "w4a16" in mode_ for mode_ in mode)

    is_multimodal = False
    try:
        model_type = model_cfg.get("model_type", "")
        if model_type == "bloom":
            model = BloomTpPartModel(model_kvargs)
        elif model_type == "llama":
            if is_weight_only_quant:
                model = LlamaTpPartModelWQuant(model_kvargs)
            elif any("w8a8" in mode_ for mode_ in mode):
                model = LlamaTpPartModelAWQuant(model_kvargs)
            elif any('quik_activation_weight' in mode_ for mode_ in mode):
                # Supports both w4a4 and w8a8 modes, with automatic mode selection upon model loading.
                model = LlamaTpPartModelQuik(model_kvargs)
            else:
                model = LlamaTpPartModel(model_kvargs)
        elif model_type == "qwen":
            if "visual" in model_cfg:
                model = QWenVLTpPartModel(model_kvargs)
                is_multimodal = True
            elif is_weight_only_quant:
                model = QWenTpPartModelWQuant(model_kvargs)
            else:
                model = QWenTpPartModel(model_kvargs)
        elif model_type == "baichuan":
            if model_cfg["hidden_size"] == 4096:
                if model_cfg["architectures"][0] == "BaichuanForCausalLM":
                    model = Baichuan2_7bTpPartModel(model_kvargs)
                else:
                    model = Baichuan7bTpPartModel(model_kvargs)
            elif model_cfg["hidden_size"] == 5120:
                if model_cfg["architectures"][0] == "BaichuanForCausalLM":
                    model = Baichuan2_13bTpPartModel(model_kvargs)
                else:
                    model = Baichuan13bTpPartModel(model_kvargs)
            else:
                raise Exception("can not support baichuan format")
        elif model_type == "gpt_bigcode":
            if is_weight_only_quant:
                model = StarcoderTpPartModelWQuant(model_kvargs)
            else:
                model = StarcoderTpPartModel(model_kvargs)
        elif model_type == "starcoder2":
            model = Starcoder2TpPartModel(model_kvargs)
        elif model_type == "chatglm":
           model = ChatGlm2TpPartModel(model_kvargs)
        elif model_type == "internlm":
            if is_weight_only_quant:
                model = InternlmTpPartModelWQuant(model_kvargs)
            else:
                model = InternlmTpPartModel(model_kvargs)
        elif model_type == "internlm2":
            if is_weight_only_quant:
                model = Internlm2TpPartModelWQuant(model_kvargs)
            else:
                model = Internlm2TpPartModel(model_kvargs)
        elif model_type == "Yi":
            model = YiTpPartModel(model_kvargs)
        elif model_type == "mistral":
            model = MistralTpPartModel(model_kvargs)
        elif model_type == "stablelm":
            model = StablelmTpPartModel(model_kvargs)
        elif model_type == "mixtral":
            model = MixtralTpPartModel(model_kvargs)
        elif model_type == "minicpm" or model_cfg["architectures"][0] == "MiniCPMForCausalLM":
            model = MiniCPMTpPartModel(model_kvargs)
        elif model_type == "llava":
            model = LlavaTpPartModel(model_kvargs)
            is_multimodal = True
        elif model_type == "internlmxcomposer2":
            model = InternlmComposerTpPartModel(model_kvargs)
            is_multimodal = True
        elif model_type == "qwen2":
            model = Qwen2TpPartModel(model_kvargs)
        elif model_type == "gemma":
            model = Gemma_2bTpPartModel(model_kvargs)
        else:
            raise Exception(f"can not support {model_type} now")
        return model, is_multimodal

    except Exception as e:
        print(f"load model error: {str(e)} {e} {type(e)}")
        import traceback
        traceback.print_exc()
        raise e


class ModelBaseServer:
    def exposed_init_model(self, kvargs):
        import torch
        import torch.distributed as dist

        world_size = kvargs["world_size"]
        self.is_multimodal = False
        self.tp_rank = kvargs["rank_id"]
        self.world_size = kvargs["world_size"]
        self.load_way = kvargs["load_way"]
        self.mode = kvargs["mode"]
        self.is_splitfuse_mode = kvargs.get("is_splitfuse_mode", False)
        self.splitfuse_block_size = kvargs.get("splitfuse_block_size", None)
        self.return_all_prompt_logprobs = kvargs.get("return_all_prompt_logprobs", False)
        self.use_dynamic_prompt_cache = kvargs.get("use_dynamic_prompt_cache", False)
        self.eos_id = kvargs.get("eos_id", [2])

        self.cache = {}
        self.logger = init_logger(__name__)

        weight_dir = kvargs["weight_dir"]
        max_total_token_num = kvargs["max_total_token_num"]

        dist.init_process_group(
            "nccl", init_method=f'tcp://127.0.0.1:{kvargs["nccl_port"]}', rank=self.tp_rank, world_size=world_size
        )
        torch.cuda.set_device(self.tp_rank)

        self.radix_cache = (
            RadixCache(str(kvargs["nccl_port"]), max_total_token_num, self.tp_rank)
            if self.use_dynamic_prompt_cache
            else None
        )

        model_kvargs = {
            "tp_rank": self.tp_rank,
            "world_size": self.world_size,
            "weight_dir": weight_dir,
            "max_total_token_num": max_total_token_num,
            "load_way": self.load_way,
            "mode": self.mode,
            "max_req_num": kvargs.get("max_req_num", 1000),
            "max_seq_length": kvargs.get("max_seq_length", 1024 * 5),
            "return_all_prompt_logprobs": self.return_all_prompt_logprobs,
            "use_dynamic_prompt_cache": self.use_dynamic_prompt_cache,
        }
        self.model, self.is_multimodal = load_llm_model(model_kvargs)
        set_random_seed(2147483647)

        return

    # @calculate_time(show=True, min_cost_ms=0.1)
    def exposed_add_batch(self, batch_id, reqs, dtype):
        import torch

        if dtype == "fp16":
            dtype = torch.float16
        else:
            assert False, "error dtype"
        batch_data = InferBatch.init_batch(
            batch_id,
            reqs,
            dtype,
            torch.cuda.current_device(),
            self.model.req_manager,
            self.model.vocab_size,
            self.radix_cache,
        )
        self.cache[batch_id] = batch_data

        # 将更新后的状态返回给调用方用于router中请求的状态
        ans = {}
        for req_id in batch_data.request_ids:
            req_obj: InferReq = requests_mapping[req_id]
            ans[req_id] = (req_obj.req_status, req_obj.cur_kv_len)
        return ans

    @calculate_time(show=False, min_cost_ms=300)
    def exposed_prefill_batch(self, batch_id):
        return self.forward(batch_id, is_prefill=True)

    @calculate_time(show=True, min_cost_ms=200)
    def exposed_decode_batch(self, batch_id):
        if self.is_splitfuse_mode:
            return self.splitfuse_forward(batch_id)
        else:
            return self.forward(batch_id, is_prefill=False)

    # @calculate_time(show=True, min_cost_ms=0.1)
    def exposed_filter_batch(self, batch_id, req_id_list, finished_req_id_list):
        # print("filter old size:", len(batch.reqs), "new size:", len(req_id_list))
        batch = self.cache.pop(batch_id)
        filter_batch = batch.filter(req_id_list, finished_req_id_list)
        del batch
        self.cache[batch_id] = filter_batch
        return

    def exposed_pause_reqs(self, batch_id, req_list):
        batch1 = self.cache.pop(batch_id)
        batch2 = batch1.pause_reqs(req_list)
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

    # @calculate_time(show=True, min_cost_ms=150)
    def forward(self, batch_id, is_prefill):
        # special code for return all prompt_logprobs
        if self.return_all_prompt_logprobs and is_prefill:
            return self._prefill_to_return_all_prompt_logprobs(batch_id)

        output_dict = {}
        batch: InferBatch = self.cache.pop(batch_id)
        if is_prefill:
            kwargs, run_reqs = prepare_prefill_inputs(
                batch, self.radix_cache, self.model.mem_manager, self.is_multimodal
            )
        else:
            kwargs, run_reqs = prepare_decode_inputs(batch, self.radix_cache, self.model.mem_manager)

        logits = self.model.forward(**kwargs)
        next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        for req_obj, next_token_id, next_token_logprob in zip(run_reqs, next_token_ids, next_token_logprobs):
            # prefill and decode is same
            req_obj.cur_kv_len = len(req_obj.input_token_ids)
            req_obj.input_token_ids.append(next_token_id)
            req_obj.out_token_id_count[next_token_id] += 1
            metadata = {
                "id": int(next_token_id),
                "logprob": float(next_token_logprob),
            }
            output_dict[req_obj.r_id] = (
                req_obj.req_status,
                req_obj.cur_kv_len,
                int(next_token_id),
                metadata,
            )  # 状态， cur_kv_len, token_id, metadata

        self.cache[batch.batch_id] = batch
        return output_dict

    @torch.no_grad()
    def _prefill_to_return_all_prompt_logprobs(self, batch_id):
        # 在 return all_prompt_logprobs 的模式下，不能启用 dynamic prompt cache
        assert self.radix_cache is None
        output_dict = {}
        batch: InferBatch = self.cache.pop(batch_id)
        kwargs, run_reqs = prepare_prefill_inputs(batch, self.radix_cache, self.model.mem_manager)

        prompt_all_logits = self.model.forward(**kwargs)
        input_ids = kwargs["input_ids"]
        b_start_loc = kwargs["b_start_loc"]
        b_seq_len = kwargs["b_seq_len"]
        last_index = torch.cumsum(b_seq_len, dim=0, dtype=torch.long) - 1
        logits = prompt_all_logits[last_index, :]

        next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        b_start_loc = b_start_loc.cpu().numpy()
        b_seq_len = b_seq_len.cpu().numpy()
        for req_obj, next_token_id, next_token_logprob, start_loc, seq_len in zip(
            run_reqs, next_token_ids, next_token_logprobs, b_start_loc, b_seq_len
        ):
            # prefill and decode is same
            req_obj.cur_kv_len = len(req_obj.input_token_ids)
            req_obj.input_token_ids.append(next_token_id)
            req_obj.out_token_id_count[next_token_id] += 1
            metadata = {
                "id": int(next_token_id),
                "logprob": float(next_token_logprob),
            }

            cur_ids: torch.Tensor = input_ids[start_loc : start_loc + seq_len]
            cur_logits = prompt_all_logits[start_loc : start_loc + seq_len]
            cur_logprobs = torch.log_softmax(cur_logits, dim=-1, dtype=torch.float)[0:-1, :]
            cur_logprobs = torch.gather(cur_logprobs, dim=1, index=cur_ids[1:].view(-1, 1)).detach().cpu().numpy()

            cur_ids = cur_ids.cpu().numpy()
            all_prompts = []
            for index in range(len(cur_ids) - 1):
                tmp_dict = {int(cur_ids[index + 1]): float(cur_logprobs[index, 0])}
                all_prompts.append([int(cur_ids[index]), tmp_dict])

            metadata["prompt_logprobs"] = all_prompts
            metadata["prompt_token_ids"] = [int(e) for e in cur_ids]
            output_dict[req_obj.r_id] = (
                req_obj.req_status,
                req_obj.cur_kv_len,
                int(next_token_id),
                metadata,
            )  # 状态， cur_kv_len, token_id, metadata

        self.cache[batch.batch_id] = batch
        return output_dict

    # @calculate_time(show=True, min_cost_ms=200)
    def splitfuse_forward(self, batch_id):
        output_dict = {}
        batch: InferBatch = self.cache.pop(batch_id)
        kwargs, decode_reqs, prefill_reqs = splitfuse_prepare_decode_inputs(
            batch, self.splitfuse_block_size, self.radix_cache, self.model.mem_manager
        )
        decode_req_num = len(decode_reqs)
        all_reqs = decode_reqs
        all_reqs.extend(prefill_reqs)

        logits = self.model.splitfuse_forward(**kwargs)
        next_token_ids, next_token_probs = sample(logits, all_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        index = 0
        for req_obj, next_token_id, next_token_logprob in zip(all_reqs, next_token_ids, next_token_logprobs):
            if index < decode_req_num:
                req_obj.cur_kv_len = len(req_obj.input_token_ids)
                req_obj.input_token_ids.append(next_token_id)
                req_obj.out_token_id_count[next_token_id] += 1
                metadata = {
                    "id": int(next_token_id),
                    "logprob": float(next_token_logprob),
                }
                output_dict[req_obj.r_id] = (
                    req_obj.req_status,
                    req_obj.cur_kv_len,
                    int(next_token_id),
                    metadata,
                )  # 状态， cur_kv_len, token_id, metadata
            else:
                old_input_token_size = len(req_obj.input_token_ids)
                split_len = min(old_input_token_size - req_obj.cur_kv_len, self.splitfuse_block_size)
                if req_obj.cur_kv_len + split_len == old_input_token_size:
                    # 有输出
                    req_obj.cur_kv_len = old_input_token_size
                    req_obj.input_token_ids.append(next_token_id)
                    req_obj.out_token_id_count[next_token_id] += 1
                    metadata = {
                        "id": int(next_token_id),
                        "logprob": float(next_token_logprob),
                    }
                    output_dict[req_obj.r_id] = (req_obj.req_status, req_obj.cur_kv_len, int(next_token_id), metadata)
                elif req_obj.cur_kv_len + split_len < old_input_token_size:
                    # 没输出
                    req_obj.cur_kv_len = req_obj.cur_kv_len + split_len
                    output_dict[req_obj.r_id] = (req_obj.req_status, req_obj.cur_kv_len, None, None)
                else:
                    assert False, "error state"
            index += 1

        self.cache[batch.batch_id] = batch
        return output_dict