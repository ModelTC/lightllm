
import unittest
import random
import torch
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.req_manager import ReqManager
from lightllm.models.llama.triton_kernel.gqa_flash_decoding_vsm import gqa_token_decode_attention_flash_decoding_vsm
from lightllm.models.llama.triton_kernel.gqa_flash_decoding import gqa_token_decode_attention_flash_decoding
from lightllm.utils.infer_utils import benchmark_time

class TestVSMGQADecoding(unittest.TestCase):
    def test_vsm_gqa_decoding_able_to_run(self):
        # set seed 
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        bs_list = range(1, 40)
        group_size_list = [8, 16]
        seq_len_list = [256, 512, 1024, 2048]
        q_head_dim_list = [64, 128]
        q_head_num_list = [8, 16, 32]

        for bs in bs_list:
            for group_size in group_size_list:
                for seq_len in seq_len_list:
                    for q_head_dim in q_head_dim_list:
                        for q_head_num in q_head_num_list:
                            if q_head_num < group_size:
                                continue
                            kv_head_num = q_head_num // group_size
                            q_head_dim = q_head_dim
                            kv_head_dim = q_head_dim
                            seq_len = (torch.zeros(bs, dtype=torch.int32) + seq_len).to(torch.int32)
                            total_token_in_the_batch = seq_len.sum().item()
                            rounded_total_token_in_the_batch = (total_token_in_the_batch + 128 - 1) // 128 * 128

                            q_shape = [bs, q_head_num, q_head_dim]
                            kv_shape = [rounded_total_token_in_the_batch, kv_head_num, kv_head_dim]
                            qkv_dtype = torch.float16

                            q, k, v = torch.randn(q_shape, dtype=qkv_dtype, device="cuda"), torch.randn(kv_shape, dtype=qkv_dtype, device="cuda"), torch.randn(kv_shape, dtype=qkv_dtype, device="cuda")

                            req_to_token_index = torch.zeros((bs, 2048)) - 1
                            token_index = torch.arange(rounded_total_token_in_the_batch)
                            

                            total_count = 0
                            for i in range(bs):
                                req_to_token_index[i, :seq_len[i]] = token_index[total_count:total_count + seq_len[i]]
                                total_count += seq_len[i]

                            req_to_token_index = req_to_token_index.long().cuda()
                            
                            b_req_idx = torch.arange(bs, device="cuda")
                            state = InferStateInfo()
                            state.req_manager = ReqManager(bs, 2048, None)
                            state.b_req_idx = b_req_idx.cuda()
                            state.b_seq_len = seq_len.cuda()
                            state.max_len_in_batch = 2048
                            state.batch_size = bs
                            state.q_head_num = q_head_num
                            state.q_head_dim = q_head_dim
                            state.kv_head_num = kv_head_num
                            state.softmax_scale = 1 / (q_head_dim ** 0.5)
                            state.total_token_num = torch.tensor([total_token_in_the_batch], dtype=torch.int32).cuda() 
                            benchmark_time(gqa_token_decode_attention_flash_decoding_vsm, q, k, v, state, warmup=0, repeat=1)

if __name__ == "__main__":
    unittest.main()