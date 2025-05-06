import cython
import time

def test_cython():
    start = time.time()
    a = 0
    for i in range(100000):
        a += i
    end = time.time()
    print("Python time:", end - start, "Result:", a)
    

@cython.cfunc
def test_cython1():
    a: cython.int = 0
    size: cython.int = 100000
    i: cython.int = 0

    # for i in range(size):
    #     a += i
    while i < size:
        a += i
        i += 1
    
    


test_cython()
start = time.time()
test_cython1()
end = time.time()
print("Python time1:", end - start)
test_cython()

start = time.time()
test_cython1()
end = time.time()
print("Python time1:", end - start)


import torch

buffer_size = 1000 * 1000 
pinned_buffer = torch.empty(buffer_size, dtype=torch.float32, device="cpu", pin_memory=True)
pinned_buffer[:] = 0.0

numpy_array = pinned_buffer.numpy()
numpy_array[0] = 100
print(pinned_buffer)


LOADWORKER=8 python -m lightllm.server.api_server --port 8018 --model_dir /dev/shm/Qwen3-30B-A3B/ --tp 8 --graph_max_batch_size 16  --enable_fa3