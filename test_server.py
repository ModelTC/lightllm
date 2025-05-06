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
            print('Error:', response.status_code, response.text)


url = 'http://127.0.0.1:8018/generate'
headers = {'Content-Type': 'application/json'}

for i in range(1):
    data = {
        'inputs': 'San Francisco is a',
        # 'temperature': 0.1,
        'parameters' : {
            'do_sample': False,
        }
    }
    thread = RequestThread(url, headers, data)
    thread.start()

time.sleep(2)

for i in range(15):
    data = {
        'inputs': '中国人',
        'parameters': {
            'do_sample': False,
            'ignore_eos': False,
            'max_new_tokens': 200,
            'stop_sequences': [],
        }
    }
    thread = RequestThread(url, headers, data)
    thread.start()

"""
LOADWORKER=8 python -m lightllm.server.api_server --sampling_backend sglang_kernel --port 8013 --model_dir /mtc/models/DeepSeek-V2-Lite-Chat --tp 2 --nccl_port 32131 --graph_max_batch_size 16
LOADWORKER=8 python -m lightllm.server.api_server --port 8012 --model_dir /mtc/models/bloom --tp 1 --nccl_port 32131 --graph_max_batch_size 16

"""