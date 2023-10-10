def init_bloc(b_loc, b_loc_idx, b_seq_len, max_len_in_batch, alloc_mem_index):
    start_index = 0
    b_seq_len_numpy = b_seq_len.cpu().numpy()
    for i in range(len(b_seq_len)):
        cur_seq_len = b_seq_len_numpy[i]
        b_loc[b_loc_idx[i], 0:cur_seq_len] = alloc_mem_index[start_index:start_index + cur_seq_len]
        start_index += cur_seq_len
    return