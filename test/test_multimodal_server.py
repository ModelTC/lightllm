import base64
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


def image_to_base64(image):
    with open(image, "rb") as fin:
        encoded_string = base64.b64encode(fin.read()).decode("utf-8")
    return encoded_string


question = "Describe this picture to me."
question = f"user\nYou are an AI assistant whose name is goodAI. \
            <start_of_image>{question}\n"

url = "http://localhost:9999/generate"
headers = {"Content-Type": "application/json"}

for i in range(1):
    b64 = image_to_base64("test/test.jpg")
    dct = {"type": "base64", "data": b64}
    data = {
        "inputs": question,
        "parameters": {"temperature": 0.0, "do_sample": False, "max_new_tokens": 200},
        "multimodal_params": {
            "images": [
                dct,
            ]
        },
    }
    thread = RequestThread(url, headers, data)
    thread.start()
