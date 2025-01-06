import unittest
import random
import torch
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.req_manager import ReqManager
from lightllm.models.llama.triton_kernel.vsm_gqa_flash_decoding import vsm_gqa_flash_decoding
from lightllm.models.llama.triton_kernel.gqa_flash_decoding import gqa_token_decode_attention_flash_decoding
from lightllm.utils.infer_utils import benchmark_time

class TestVSMGQADecoding(unittest.TestCase):
    def test_vsm_gqa_decoding(self):
        # set seed 
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        for bs in range(1, 40):
            for group_size in [8, 16]:
                for seq_len in [256, 512, 1024, 2048]:
                    for q_head_dim in [64, 128]:
                        for q_head_num in [8, 16, 32]:
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
                            shuffle_token_index = token_index[torch.randperm(rounded_total_token_in_the_batch)]

                            total_count = 0
                            for i in range(bs):
                                req_to_token_index[i, :seq_len[i]] = shuffle_token_index[total_count:total_count + seq_len[i]]
                                total_count += seq_len[i]

                            req_to_token_index = req_to_token_index.long().cuda()
                            
                            b_req_idx = torch.arange(bs, device="cuda")
                            state = InferStateInfo()
                            state.req_manager = ReqManager(bs, 2048, None)
                            state.b_req_idx = b_req_idx.cuda()
                            state.b_seq_len = seq_len.cuda()
                            state.max_len_in_batch = 2048
                            state.batch_size = bs
                            state.total_token_num = torch.tensor([total_token_in_the_batch], dtype=torch.int32).cuda() 
                            # vsm_gqa_flash_decoding(q, state, k, v, total_token_in_the_batch, q_head_dim, q_head_num, kv_head_dim, kv_head_num)
                            try:
                                time_vsm = benchmark_time(vsm_gqa_flash_decoding, q, state, k, v, q_head_dim, q_head_num, kv_head_dim, kv_head_num)
                            except Exception as e:
                                print(f'q_shape: {q_shape} kv_shape: {kv_shape}')
                                raise e
                            # gqa_token_decode_attention_flash_decoding(q, state, q_head_num, q_head_dim, k, v)
                            time_ori = benchmark_time(gqa_token_decode_attention_flash_decoding, q, state, q_head_num, q_head_dim, k, v)
                            print(f'accleartion: {time_ori / time_vsm} b_seq_len: avg={seq_len.float().mean().item()} max={seq_len.max().item()} min={seq_len.min().item()} std={seq_len.float().std().item()}')

if __name__ == "__main__":
    unittest.main()
