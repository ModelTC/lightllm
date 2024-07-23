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


url = 'http://localhost:8888/generate'
headers = {'Content-Type': 'application/json'}

for i in range(1):
    data = {
        "inputs": """<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the correct answer to this question: trans-cinnamaldehyde was treated with methylmagnesium bromide, forming product 1.\n\n1 was treated with pyridinium chlorochromate, forming product 2.\n\n3 was treated with (dimethyl(oxo)-l6-sulfaneylidene)methane in DMSO at elevated temperature, forming product 3.\n\nhow many carbon atoms are there in product 3?\nChoices:\n(A)12\n(B)14\n(C)11\n(D)10\nFormat your response as follows: "The correct answer is (insert answer here)"<|im_end|>\n<|im_start|>assistant\n""",
        "parameters": {
            "temperature": 1,
            "max_new_tokens": 300,
            "stop_sequences": [
            "<|endofblock|>",
            "<|endofblock|><|im_end|>",
            "<|endoftext|>",
            "<|im_start|>"
            ],
            # "repetition_penalty": 1.05,
            "top_k": 1,
            "best_of": 1,
            "do_sample": True
        }
    }
    thread = RequestThread(url, headers, data)
    thread.start()
"""
time.sleep(2)

for i in range(20):
    data = {
        'inputs': 'San Francisco is a',
        'parameters': {
            'do_sample': False,
            'ignore_eos': True,
            'max_new_tokens': 200,
        }
    }
    thread = RequestThread(url, headers, data)
    thread.start()
"""