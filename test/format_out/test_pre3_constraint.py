import time
import requests
import json
import threading
from transformers import AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("/data/nvme1/models/llama3-8b-instruct")

ds = load_dataset("NousResearch/json-mode-eval")
prompt = ds["train"]["prompt"]


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


url = "http://localhost:8888/generate"
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
json_grammar_ebnf_file = "/data/nvme1/chenjunyi/project/lightllm/test/format_out/json_grammar.ebnf"
chain_of_thought_ebnf_file = "/data/nvme1/chenjunyi/project/lightllm/test/format_out/chain_of_thought.ebnf"

# system_prompt = open("./test/format_out/system.md", "r").read()
# user_input = open("./test/format_out/user.md", "r").read()

cot_system_prompt = """ Question: 9.11 and 9.9 -- which is bigger?
Answer: {"reasoning":[{"reasoning_step":"Both 9.11 and 9.9 are decimal numbers."},
{"reasoning_step":"When comparing decimal numbers, we look at the numbers after the decimal point."},
{"reasoning_step":"In this case, 9.11 has the number 1 after the decimal point, while 9.9 has the number 9."},
{"reasoning_step":"Since 1 is greater than 9, 9.11 is greater than 9.9."}],"conclusion":"9.11 is bigger."}
Following the example above, answer the question.
"""

# messages = [
#     {"role": "user", "content": prompt[0]},
# ]
messages = []
for i in range(80):
    messages.append([{"role": "user", "content": prompt[i]}])

cot_question = [
    # {"role": "system", "content": cot_system_prompt},
    {"role": "user", "content": "Question: 8.11 and 8.9 -- which is bigger? Answer:"}
]

# inputs = tokenizer.apply_chat_template(cot_question, tokenize=False)
inputs = [tokenizer.apply_chat_template(messages[i], tokenize=False) for i in range(len(messages))]
# x = 7
# print(inputs[x])
# print()
for i in range(10):
    data = {
        "inputs": inputs[i],
        # 'temperature': 0.1,
        "parameters": {
            "do_sample": False,
            # "guided_grammar": json_grammar_ebnf_file,
            "max_new_tokens": 150,
        },
    }
    thread = RequestThread(url, headers, data)
    thread.start()

# time.sleep(2)

# for i in range(20):
#     data = {
#         "inputs": "12-(25+16)*7=",
#         "parameters": {
#             "do_sample": False,
#             "ignore_eos": True,
#             "max_new_tokens": 200,
#             "guided_grammar": r"""root ::= (expr "=" term)+
# expr  ::= term ([-+*/] term)*
# term  ::= num | "(" expr ")"
# num   ::= [0-9]+""",
#         },
#     }
#     thread = RequestThread(url, headers, data)
#     thread.start()
