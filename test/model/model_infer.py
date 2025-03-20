import os
import numpy as np
from multiprocessing import Queue
import multiprocessing
from transformers import PretrainedConfig
from lightllm.utils.dist_utils import init_distributed_env
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.models.deepseek2.model import Deepseek2TpPartModel


def test_model_inference(args, model_class):
    ans_queue = Queue()
    workers = []
    dp_size = args.get("dp", 1)
    if dp_size > 1:
        os.environ["ENABLE_DP"] = "1"

    for rank_id in range(args.tp):
        model_kvargs = {
            "args": args,
            "nccl_host": args.nccl_host,
            "data_type": args.data_type,
            "nccl_port": args.nccl_port,
            "rank_id": rank_id,
            "world_size": args.tp,
            "dp_size": dp_size,
            "weight_dir": args.model_dir,
            "load_way": "HF",
            "max_total_token_num": args.batch_size * (args.input_len + args.output_len + 1),
            "graph_max_len_in_batch": args.max_req_total_len,
            "graph_max_batch_size": args.graph_max_batch_size,
            "mem_faction": args.mem_fraction,
            "max_req_num": max(args.batch_size, 2048),
            "batch_max_tokens": args.batch_size * args.input_len,
            "run_mode": "normal",
            "max_seq_length": (args.input_len + args.output_len),
            "disable_cudagraph": True if args.profile else False,
        }
        proc = multiprocessing.Process(
            target=tppart_model_infer,
            args=(model_class, model_kvargs, args.batch_size, args.input_len, args.output_len, ans_queue),
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
    args = get_env_start_args()
    import triton.profiler as proton
    import torch
    from lightllm.distributed import dist_group_manager
    from lightllm.utils.dist_utils import set_current_device_id

    import torch.distributed as dist

    init_distributed_env(model_kvargs)

    if model_class == Deepseek2TpPartModel:
        model_cfg, _ = PretrainedConfig.get_config_dict(model_kvargs["weight_dir"])
        dist_group_manager.set_deepep(model_cfg["n_routed_experts"])
    dist.barrier()

    torch.cuda.empty_cache()

    model_part = model_class(model_kvargs)

    # warm up
    # test_data = np.vstack([np.arange(5, input_len + 5) for _ in range(batch_size)])
    test_data = np.vstack([np.random.randint(0, 50256, input_len) for _ in range(batch_size)])
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

    rank_id = model_kvargs["rank_id"]
    if rank_id == 0:
        if args.profile:
            proton.start(name="forward_prefill", context="python")

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
        if args.profile:
            proton.finalize()
        print("prefill time cost:", (time.time() - prefill_start_time) * 1000)

    if rank_id == 0:
        if args.profile:
            proton.start(name="forward_decode", context="python")

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
        if args.profile:
            proton.finalize()
            # triton version need >= 3.2.0
            # pip install llnl-hatchet
            # proton-viewer -m time/ms,time/% forward_prefill.hatchet
            # proton-viewer -m time/ms,time/% forward_decode.hatchet
        print("time total cost(ms):", (end_time - start_time) * 1000)
    ans_queue.put(True)

    return
