def init_req_to_token_indexes(req_to_token_indexs, b_req_idx, b_seq_len, b_prompt_cache_len, max_len_in_batch, alloc_mem_index):
    start_index = 0
    b_seq_len_numpy = b_seq_len.cpu().numpy()
    b_req_idx_numpy = b_req_idx.cpu().numpy()
    for i in range(len(b_seq_len)):
        cur_seq_len = b_seq_len_numpy[i]
        req_to_token_indexs[b_req_idx_numpy[i], b_prompt_cache_len[i] : cur_seq_len + b_prompt_cache_len[i]] = alloc_mem_index[start_index:start_index + cur_seq_len]
        start_index += cur_seq_len
    return