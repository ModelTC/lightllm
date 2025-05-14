import time
import requests
import json
import threading


class RequestThread(threading.Thread):
    def __init__(self, url, headers, data):
        threading.Thread.__init__(self)
        self.url = url
        self.headers = headers
        self.data = data

    def run(self):
        response = requests.post(self.url, headers=self.headers, data=json.dumps(self.data))
        if response.status_code == 200:
            print(response.json())
        else:
            print("Error:", response.status_code, response.text)


url = "http://localhost:8088/generate"
headers = {"Content-Type": "application/json"}

for i in range(1):
    data = {
        "inputs": "San Francisco is a",
        # 'temperature': 0.1,
        "parameters": {
            "do_sample": False,
        },
    }
    thread = RequestThread(url, headers, data)
    thread.start()

time.sleep(2)

for i in range(16):
    data = {
        "inputs": "San Francisco is a",
        "parameters": {
            "do_sample": False,
            "ignore_eos": True,
            "max_new_tokens": 200,
        },
    }
    thread = RequestThread(url, headers, data)
    thread.start()

"""
LOADWORKER=8 python -m lightllm.server.api_server --port 8019 --model_dir /mtc/baishihao/Qwen3-30B-A3B-FP8 --use_tcp_store_server_to_init_nccl --tp 2 --graph_max_batch_size 16 --nccl_host 10.120.114.75  --nccl_port 8111 --enable_fa3 --dp 2 | tee log.txt
"""

LOADWORKER=8 python -m lightllm.server.api_server --port 8019 --model_dir /mtc/baishihao/Qwen3-30B-A3B-FP8 --tp 2 --graph_max_batch_size 16 --nccl_host 10.120.114.75  --nccl_port 8111 --enable_fa3 --dp 2 | tee log.txt

"""
MOE_MODE=EP LOADWORKER=18 python -m lightllm.server.api_server --port 8019 --model_dir /dev/shm/DeepSeek-R1 \
--tp 8 \
--dp 8 \
--max_total_token_num 100000 \
--graph_max_batch_size 16 \
--batch_max_tokens 4096 \
--enable_flashinfer_prefill \
--enable_flashinfer_decode  \
--enable_prefill_microbatch_overlap \
--disable_aggressive_schedule \
--ep_redundancy_expert_config_path /mtc/wzj/lightllm/test/test_redundancy_expert_config.json \
--auto_update_redundancy_expert
"""