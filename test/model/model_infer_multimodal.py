import numpy as np
from multiprocessing import Queue
import multiprocessing

def test_multimodal_inference(world_size, model_dir, model_class, batch_size, input_len, output_len, extra_embed_len, offset):
    ans_queue = Queue()
    workers = []
    for rank_id in range(world_size):
        proc = multiprocessing.Process(target=tppart_multimodal_infer, args=(rank_id, world_size, ans_queue, model_dir, model_class, batch_size, input_len, output_len, extra_embed_len, offset))
        proc.start()
        workers.append(proc)

    for proc in workers:
        proc.join()

    assert not ans_queue.empty()
    while not ans_queue.empty():
        assert ans_queue.get()
    return 


# for multimodal, input_embeds is concated by the text embeds and extra embeds
def gen_input_embeds(batch_size, model, input_len, extra_embed_len, offset):
    import torch
    all_input_ids = []
    all_input_embeds = []

    for i in range(batch_size):
        # shape = [input_len]
        input_ids = torch.from_numpy(np.arange(5, input_len + 5).reshape(-1)).cuda()
        # shape = [input_len, dim_size]
        text_embeds = model.get_input_embeddings()(input_ids)

        # generate random extra embeds
        dim_size = text_embeds.shape[1]
        # shape = [extra_embed_len, dim_size]
        extra_embeds = torch.rand(
            size=(extra_embed_len, dim_size),
            dtype=text_embeds.dtype,
            device=text_embeds.device
        ) - 0.5
        # shape = [input_len + extra_embed_len, dim_size]
        input_embeds = torch.cat([text_embeds[:offset], extra_embeds, text_embeds[offset:]], dim=0)
        all_input_embeds.append(input_embeds)

        # length of input_embeds should == length of input_ids, pad the input_ids
        # shape = [extra_embed_len]
        pad_ids = torch.zeros(
            size=(extra_embed_len,),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        # shape = [input_len + extra_embed_len] 
        input_ids = torch.cat([input_ids[:offset], pad_ids, input_ids[offset:]], dim=0)
        all_input_ids.append(input_ids)

    inputs_ids = torch.cat(all_input_ids)
    inputs_embeds = torch.cat(all_input_embeds)
    return inputs_ids, inputs_embeds 


def tppart_multimodal_infer(rank_id, world_size, ans_queue, model_dir, model_class, batch_size, input_len, output_len, extra_embed_len, offset):
    import torch
    import torch.distributed as dist
    dist.init_process_group('nccl', init_method='tcp://127.0.0.1:28765', rank=rank_id, world_size=world_size)
    torch.cuda.set_device(rank_id)

    import torch.distributed as dist
    dist.barrier()
    torch.cuda.empty_cache()

    model_part = model_class(dist.get_rank(), 
                             dist.get_world_size(), 
                             max_total_token_num= batch_size * (input_len + output_len + extra_embed_len), 
                             weight_dir=model_dir, 
                             load_way="HF")
    # warm up
    test_data, test_embeds = gen_input_embeds(
        batch_size, model_part, input_len, extra_embed_len, offset)
    # after gen_input_embeds, real input_len is plus by extra_embed_len
    input_len += extra_embed_len

    b_loc = torch.zeros(batch_size, input_len + output_len, dtype=torch.long, device="cuda")
    b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        b_loc[i, 0:input_len] = i * input_len + torch.arange(0, input_len, dtype=torch.int32, device="cuda")
        b_start_loc[i] = i * input_len
        b_seq_len[i] = input_len

    total_token_num = input_len * batch_size
    logics = model_part.forward(batch_size, 
                                total_token_num, 
                                input_len, 
                                test_data,
                                b_loc,
                                b_start_loc,
                                b_seq_len,
                                is_prefill=True,
                                inputs_embeds=test_embeds)
    prob_out = torch.softmax(logics, dim=-1)
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    predict_ids = predict_ids.detach().cpu().numpy()

    for i in range(output_len):
        b_loc[:, input_len + i] = total_token_num + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        b_start_loc = b_start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        total_token_num += batch_size
        b_seq_len += 1
        logics = model_part.forward(batch_size, total_token_num, input_len + i + 1, torch.from_numpy(
            predict_ids).cuda().reshape(-1), b_loc, b_start_loc, b_seq_len, is_prefill=False)
        prob_out = torch.softmax(logics, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach().cpu().numpy()
    
    max_len_in_batch = input_len + output_len
    for i in range(batch_size):
        model_part.mem_manager.free(b_loc[i, max_len_in_batch - b_seq_len[i]:max_len_in_batch])
    if rank_id == 0:
        print("can use mem size:", model_part.mem_manager.can_use_mem_size)
        
    b_loc = None
    b_start_loc = None
    b_seq_len = None
    
    dist.barrier()
    import time
    torch.cuda.synchronize()
    start_time = time.time()

    prefill_start_time = time.time()

    b_loc = torch.zeros(batch_size, input_len + output_len, dtype=torch.long, device="cuda")
    b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        b_start_loc[i] = i * input_len
        b_seq_len[i] = input_len

    total_token_num = batch_size * input_len
    logics = model_part.forward(batch_size, total_token_num, input_len, test_data,
        b_loc, b_start_loc, b_seq_len, is_prefill=True, inputs_embeds=test_embeds)
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

        logics = model_part.forward(batch_size, total_token_num, input_len + i + 1, torch.from_numpy(
            predict_ids).cuda().reshape(-1), b_loc, b_start_loc, b_seq_len, is_prefill=False)
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


