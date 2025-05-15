import torch

import triton
import triton.language as tl
import copy


@triton.jit
def _fwd_kernel_copy_kv_index_to_req(
    req_to_token_indexs, b_req_idx, b_seq_len, memindex, stride_req_to_token_b, stride_req_to_token_s
):
    cur_seq = tl.program_id(0)
    cur_index = tl.program_id(1)
    batch_size = tl.num_programs(1)
    cur_req_idx = tl.load(b_req_idx + cur_index)
    cur_token_index = tl.load(memindex + cur_index + batch_size * cur_seq)
    cur_seq_len = tl.load(b_seq_len + cur_index) + cur_seq
    dest_offset = req_to_token_indexs + cur_req_idx * stride_req_to_token_b + (cur_seq_len - 1) * stride_req_to_token_s
    tl.store(dest_offset, cur_token_index)
    return


@torch.no_grad()
def copy_kv_index_to_req(req_to_token_indexs, b_req_idx, b_seq_len, memindex, decode_len=1):
    batch_size = b_seq_len.shape[0]
    assert b_seq_len.shape[0] * decode_len == memindex.shape[0] and b_req_idx.shape[0] == b_seq_len.shape[0]
    grid = (
        decode_len,
        batch_size,
    )
    num_warps = 1
    _fwd_kernel_copy_kv_index_to_req[grid](
        req_to_token_indexs,
        b_req_idx,
        b_seq_len,
        memindex,
        req_to_token_indexs.stride(0),
        req_to_token_indexs.stride(1),
        num_warps=num_warps,
        num_stages=1,
    )

    return


if __name__ == "__main__":
    for decode_len in [1, 2]:
        max_request_num = 100
        max_sequence_length = 1000
        req_to_token_indexs = torch.zeros((max_request_num + 1, max_sequence_length), dtype=torch.int32, device="cuda")
        bs = 8
        b_req_idx = torch.randint(low=0, high=max_request_num - 1, size=(bs,)).cuda()
        b_seq_len = torch.randint(low=1, high=max_sequence_length, size=(bs,)).cuda()
        memindex = torch.randint(low=0, high=10000, size=(bs * decode_len,)).cuda()
        copy_kv_index_to_req(req_to_token_indexs, b_req_idx, b_seq_len, memindex, decode_len)

        for i in range(bs):
            for j in range(decode_len):
                if req_to_token_indexs[b_req_idx[i]][b_seq_len[i] + j - 1] != memindex[j * bs + i]:
                    print("ERROR")
                    exit(1)

    print("PASS")
