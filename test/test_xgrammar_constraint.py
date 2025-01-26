import time
import requests
import json
import threading

"""
python -m lightllm.server.api_server --model_dir /mnt/nvme0/chenjunyi/models/nb10_w8/  \
                                     --host 0.0.0.0                 \
                                     --port 9999                   \
                                     --tp 1                        \
                                     --nccl_port 65535                \
				                     --max_req_total_len 200000 \
                                     --max_total_token_num 400000 \
                                     --data_type bf16   \
                                     --trust_remote_code  \
                                     --output_constraint_mode xgrammar
"""


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


url = "http://localhost:9999/generate"
headers = {"Content-Type": "application/json"}
json_grammar_ebnf_str = r"""
root ::= basic_array | basic_object
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*) ".0"?
basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= (([\"] basic_string_1 [\"]))
basic_string_1 ::= "" | [^"\\\x00-\x1F] basic_string_1 | "\\" escape basic_string_1
escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= "[" ("" | ws basic_any (ws "," ws basic_any)*) ws "]"
basic_object ::= "{" ("" | ws basic_string ws ":" ws basic_any ( ws "," ws basic_string ws ":" ws basic_any)*) ws "}"
ws ::= [ \n\t]*
"""

for i in range(1):
    data = {
        "inputs": "Introduce yourself in JSON briefly.",
        # 'temperature': 0.1,
        "parameters": {
            "do_sample": False,
            "guided_grammar": json_grammar_ebnf_str,
            "max_new_tokens": 200,
        },
    }
    thread = RequestThread(url, headers, data)
    thread.start()

time.sleep(2)

for i in range(20):
    data = {
        "inputs": "12-(25+16)*7=",
        "parameters": {
            "do_sample": False,
            "ignore_eos": True,
            "max_new_tokens": 200,
            "guided_grammar": r"""root ::= (expr "=" term)+
expr  ::= term ([-+*/] term)*
term  ::= num | "(" expr ")"
num   ::= [0-9]+""",
        },
    }
    thread = RequestThread(url, headers, data)
    thread.start()
