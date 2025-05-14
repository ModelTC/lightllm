import torch
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity

data_o = torch.zeros((128 * 1024), dtype=torch.int32, device="cuda")
in_data = list(range(0, 1000))
in_datas = [list(range(0, 1000)) for _ in range(100)]
import time

test1 = torch.zeros((128*1024), dtype=torch.int32, device="cpu", pin_memory=False)

data_o[0].fill_(10)

tmp = torch.zeros((100, 100), dtype=torch.int32, device="cuda")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=False,
    profile_memory=False,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile/profile.file"),
) as prof:
    for _ in range(100):
        # data_o[0].fill_(10)
        test1.pin_memory()

        # tmp[[1, 2], [3, 4]] += 1

        # test1.pin_memory().cuda(non_blocking=True)
        # a = torch.from_numpy(in_data)
        # a.cuda(non_blocking=True)

    # for _ in range(100):
    #     # data_o[0].fill_(10)
    #     a = torch.from_numpy(in_data)
    #     a.cuda(non_blocking=True)

    # torch.cuda.synchronize()
    pass
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=16), flush=True)

a = torch.tensor(in_data + [13221, ], dtype=torch.int32, device="cpu").pin_memory().cuda(non_blocking=True)

torch.cuda.synchronize()

start = time.time()

for i in range(6):
    # a = torch.from_numpy(in_data).pin_memory().cuda(non_blocking=True)
    a = torch.tensor(in_datas[i], dtype=torch.int32, device="cpu").pin_memory().cuda(non_blocking=True)
    # torch.bincount(a, minlength=0)
    # print(torch.bincount(a, minlength=0).shape)
    # a.cuda(non_blocking=True)

torch.cuda.synchronize()
print(f"{time.time() - start} s")

print(data_o[0:10])