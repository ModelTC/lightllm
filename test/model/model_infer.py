import numpy as np
from multiprocessing import Queue
import multiprocessing
import os


def test_model_inference(world_size, model_class, batch_size, input_len, output_len, extra_model_kvargs):
    ans_queue = Queue()
    workers = []
    for rank_id in range(world_size):
        model_kvargs = {
            "tp_rank": rank_id,
            "world_size": world_size,
            "load_way": "HF",
            "max_total_token_num": None,
            "mem_faction": 0.8,
            "max_req_num": max(batch_size, 1000),
            "batch_max_tokens": input_len,
            "run_mode": "normal",
            "max_seq_length": (input_len + output_len),
        }
        model_kvargs.update(extra_model_kvargs)

        proc = multiprocessing.Process(
            target=tppart_model_infer, args=(model_class, model_kvargs, batch_size, input_len, output_len, ans_queue)
        )
        proc.start()
        workers.append(proc)

    for proc in workers:
        proc.join()

    assert not ans_queue.empty()
    while not ans_queue.empty():
        assert ans_queue.get()
    return


def tppart_model_infer(model_class, model_kvargs, batch_size, input_len, output_len, ans_queue):
    import torch
    from lightllm.distributed import custom_comm_ops
    from lightllm.utils.dist_utils import set_current_device_id

    import torch.distributed as dist

    rank_id = model_kvargs["tp_rank"]
    world_size = model_kvargs["world_size"]

    torch.cuda.set_device(rank_id)
    set_current_device_id(rank_id)
    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:28765", rank=rank_id, world_size=world_size)

    custom_comm_ops.set_custom_reduce()
    custom_comm_ops.set_custom_gather()
    dist.barrier()

    torch.cuda.empty_cache()

    model_part = model_class(model_kvargs)
    # warm up
    test_data = np.vstack([np.arange(5, input_len + 5) for _ in range(batch_size)])
    test_data[:, 0] = 100000
    test_data = test_data.reshape(-1)
    test_data = torch.from_numpy(test_data).cuda()

    b_req_idx = torch.tensor(
        [model_part.req_manager.alloc() for _ in range(batch_size)], dtype=torch.int32, device="cuda"
    )
    b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_ready_cache_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        b_start_loc[i] = i * input_len
        b_seq_len[i] = input_len

    total_token_num = input_len * batch_size
    mem_indexes = model_part.req_manager.mem_manager.alloc(test_data.shape[0]).cuda()

    logics = model_part.forward(
        batch_size,
        total_token_num,
        input_len,
        test_data,
        mem_indexes,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        b_ready_cache_len=b_ready_cache_len,
        is_prefill=True,
    )
    prob_out = torch.softmax(logics, dim=-1)
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    predict_ids = predict_ids.detach().cpu().numpy()

    for i in range(output_len):
        b_start_loc = b_start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        total_token_num += batch_size
        b_seq_len += 1
        mem_indexes = model_part.req_manager.mem_manager.alloc(predict_ids.shape[0]).cuda()
        logics = model_part.forward(
            batch_size,
            total_token_num,
            input_len + i + 1,
            torch.from_numpy(predict_ids).cuda().reshape(-1),
            mem_indexes,
            b_req_idx,
            b_start_loc,
            b_seq_len,
            is_prefill=False,
        )
        prob_out = torch.softmax(logics, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach().cpu().numpy()

    model_part.mem_manager.free_all()
    model_part.req_manager.free_all()

    b_req_idx = None
    b_start_loc = None
    b_seq_len = None

    dist.barrier()
    import time

    torch.cuda.synchronize()
    start_time = time.time()

    prefill_start_time = time.time()

    b_req_idx = torch.tensor(
        [model_part.req_manager.alloc() for _ in range(batch_size)], dtype=torch.int32, device="cuda"
    )
    b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        b_start_loc[i] = i * input_len
        b_seq_len[i] = input_len

    total_token_num = batch_size * input_len
    mem_indexes = model_part.req_manager.mem_manager.alloc(test_data.shape[0]).cuda()
    logics = model_part.forward(
        batch_size,
        total_token_num,
        input_len,
        test_data,
        mem_indexes,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        b_ready_cache_len=b_ready_cache_len,
        is_prefill=True,
    )
    prob_out = torch.softmax(logics, dim=-1)
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    predict_ids = predict_ids.detach().cpu().numpy()

    torch.cuda.synchronize()
    if rank_id == 0:
        print("prefill time cost:", (time.time() - prefill_start_time) * 1000)

    for i in range(output_len):
        torch.cuda.synchronize()
        step_start = time.time()
        b_start_loc = b_start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        total_token_num += batch_size
        b_seq_len += 1
        mem_indexes = model_part.req_manager.mem_manager.alloc(predict_ids.shape[0]).cuda()
        logics = model_part.forward(
            batch_size,
            total_token_num,
            input_len + i + 1,
            torch.from_numpy(predict_ids).cuda().reshape(-1),
            mem_indexes,
            b_req_idx,
            b_start_loc,
            b_seq_len,
            is_prefill=False,
        )
        prob_out = torch.softmax(logics, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach().cpu().numpy()
        torch.cuda.synchronize()
        if i % 100 == 0 or i == output_len - 1:
            if rank_id == 0:
                print(i, "step cost time:", (time.time() - step_start) * 1000)

    torch.cuda.synchronize()
    end_time = time.time()

    if rank_id == 0:
        print("time total cost(ms):", (end_time - start_time) * 1000)
    ans_queue.put(True)

    return
