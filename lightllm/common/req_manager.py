import torch
    
class ReqManager:
    def __init__(self, max_request_num, max_sequence_length, mem_manager):
        self.req_state = torch.zeros((max_request_num,), dtype=torch.bool, device="cuda")
        self.req_to_token_indexs = torch.zeros((max_request_num, max_sequence_length), dtype=torch.int32, device="cuda")
        self.can_use_req_size = max_request_num
        self.mem_manager = mem_manager

    def alloc(self, need_size):
        if need_size > self.can_use_req_size:
            print(f'Insufficient requested capacity, remaining {self.can_use_req_size}')
            return None
        select_index = torch.nonzero(self.req_state==0).reshape(-1)[:need_size]
        self.req_state[select_index] = 1
        self.can_use_req_size -= len(select_index)
        return select_index
    
    def free_req(self, free_index, free_seq_len):
        self.can_use_req_size += 1
        self.req_state[free_index] = 0
        if self.can_use_req_size == len(self.req_state):
            print(f"freed all request size {self.can_use_req_size}")
        self.mem_manager.free(self.req_to_token_indexs[free_index][:free_seq_len])
    
    def free(self, free_index, free_seq_len):
        """_summary_

        Args:
            free_index (torch.Tensor): _description_
        """

        self.can_use_req_size += free_index.shape[0]
        self.req_state[free_index] = 0
        if self.can_use_req_size == len(self.req_state):
            print(f"freed all request size {self.can_use_req_size}")
        remove_index = []
        for (idx, seq_len) in zip(free_index, free_seq_len):
            remove_index.append(self.req_to_token_indexs[idx][:seq_len])
        remove_index = torch.cat(remove_index, dim=-1)
        self.mem_manager.free(remove_index)
        return

    def free_all(self):
        self.can_use_req_size = len(self.req_state)
        self.req_state[:] = 0
    
    