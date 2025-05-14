from jinja2 import Template
import json
import requests


def render_messages(template: str, messages: list, add_generation_prompt=True, tools=None):
    """
    Renders the given template with the provided messages.

    Args:
        template (str): The Jinja2 template string.
        messages (list): A list of dictionaries representing messages.

    Returns:
        str: The rendered string.
    """
    jinja_template = Template(template)

    if tools:
        return jinja_template.render(messages=messages, add_generation_prompt=add_generation_prompt)
    return jinja_template.render(messages=messages, add_generation_prompt=add_generation_prompt)

# Define the template
template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='', is_first_sp=true) %}{%- for message in messages %}{%- if message['role'] == 'system' %}{%- if ns.is_first_sp %}{% set ns.system_prompt = ns.system_prompt + message['content'] %}{% set ns.is_first_sp = false %}{%- else %}{% set ns.system_prompt = ns.system_prompt + '\\n\\n' + message['content'] %}{%- endif %}{%- endif %}{%- endfor %}{{ bos_token }}{{ ns.system_prompt }}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and 'tool_calls' in message %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls'] %}{%- if not ns.is_first %}{%- if message['content'] is none %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- else %}{{'<｜Assistant｜>' + message['content'] + '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- endif %}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- endif %}{%- endfor %}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- if message['role'] == 'assistant' and 'tool_calls' not in message %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n'}}{% endif %}"
from transformers import AutoTokenizer

path = "/mtc/baishihao/Qwen3-30B-A3B-FP8/"

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

def qwen3_render(messages, enable_thinking=True):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking# Switches between thinking and non-thinking modes. Default is True.
    )
    return text

def test_chat():
    # Example messages
    messages = [
        # {"role": "user", "content": "你觉得商汤科技怎么样？"},
        {"role": "user", "content": "你觉得deepseek怎么样？"},
        # {"role": "user", "content": "你好啊"},
        #{"role": "user", "content": "1+1=？"},
        # {"role": "user", "content": "欣小萌为什么是人脉区up主"},
    ]
    #messages = [{"role": "user", "content": "18.3.19 $\\\\star \\\\star$ Find all positive integer triples $(a, b, c)$ that satisfy $a^{2}+b^{2}+c^{2}=2005$ and $a \\\\leqslant b \\\\leqslant c$."}]

    # Render the messages using the template
    result = render_messages(template, messages, add_generation_prompt=True)
    #result = qwen3_render(messages, enable_thinking=True).replace("assistant\n", "assistant\n<think>")
    print(result)
    # Print the rendered result
    url = "http://0.0.0.0:8019/generate"

    # 请求头
    headers = {
        "Content-Type": "application/json"
    }

    # 请求数据
    payload = {
        "inputs": result,
        "parameters": {
            "temperature": 0.7,
            "max_new_tokens": 1000,
            "stop": [
            # "<｜begin▁of▁sentence｜>",
            # "<｜end▁of▁sentence｜>"
            "<|im_end|>"
            ],
            "top_p": 0.7,
            "repetition_penalty": 1.,
            # "top_k": 1,
            "do_sample": True,
            "return_full_text": False,
            "best_of":1,
            "skip_special_tokens": False,
            "print_eos_token": True,
    }
    }


    print("input: ", json.dumps(payload))

    # 发送 POST 请求
    response = requests.post(url, headers=headers, data=json.dumps(payload))


    # 打印响应
    if response.status_code == 200:
        print("Response:", response.json())
    else:
        print("Error:", response.status_code, response.text)



if __name__ == "__main__":
    # test chat
    test_chat()