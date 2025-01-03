import unittest
import random
import torch
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.req_manager import ReqManager
from lightllm.models.llama.triton_kernel.vsm_gqa_flash_decoding import vsm_gqa_flash_decoding

class TestVSMGQADecoding(unittest.TestCase):
    def test_vsm_gqa_decoding(self):
        for _ in range(10):
            for group_size in [8, 16]:
                bs = torch.randint(10, 40, (1,)).item()
                kv_head_num = 2 ** (torch.randint(3, 6, (1,)).item())
                q_head_dim = 128
                q_head_num = kv_head_num // group_size
                kv_head_dim = 128
                seq_len = torch.randint(128, 2048, (bs,)).item()
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
                state.b_req_idx = b_req_idx
                state.b_seq_len = seq_len 
                vsm_gqa_flash_decoding(q, state, k, v, total_token_in_the_batch, q_head_dim, q_head_num, kv_head_dim)

if __name__ == "__main__":
    unittest.main()
