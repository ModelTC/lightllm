import os
import itertools
from easydict import EasyDict
import sys
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from benchmark_prompt_cache import run


# settings
models = {
    "llama-7b": "http://localhost:8080",
    "llama-13b": "http://localhost:8081"
}
warm_up = True
num_workers = [1, 8, 16, 24, 32]
first_input_lens = [16000, 64000]
subsequent_input_lens = [128]
output_lens = [128]
num_turns = [5, 10]
num_users = [60]
result_dir = "./llama"
result_path = os.path.join(result_dir, f"summary_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.md")
heads = [
    "model_name", "num_workers", "first_input_len", "subsequent_input_len", "output_len", "num_turns", "num_users",
    "prefill_throughput(tokens/s)", "decode_throughput(tokens/s)", "total_throughput(tokens/s)",
    "qps", "首字延迟 第25% 分位数值(ms)", "首字延迟 第50% 分位数值(ms)", "首字延迟 第75% 分位数值(ms)",
    "首字延迟 第99% 分位数值(ms)", "首字延迟 第100% 分位数值(ms)", "包间延迟 第25% 分位数值(ms)",
    "包间延迟 第50% 分位数值(ms)", "包间延迟 第75% 分位数值(ms)", "包间延迟 第99% 分位数值(ms)",
    "包间延迟 第100% 分位数值(ms)",
]

# run test
results = []
for num_worker, first_input_len, subsequent_input_len, output_len, num_turn, num_user in itertools.product(
        num_workers, first_input_lens, subsequent_input_lens, output_lens, num_turns, num_users):
    for model_name, model_url in models.items():
        args = EasyDict({
            "model_name": model_name,
            "model_url": model_url,
            "num_workers": num_worker,
            "first_input_len": first_input_len,
            "subsequent_input_len": subsequent_input_len,
            "output_len": output_len,
            "num_turns": num_turn,
            "num_users": num_user,
            "result_dir": result_dir,
            "print": False,
            "cache": True,
            "use_cache": False
        })
        if warm_up:
            warm_up = False
            args["cache"] = False
            result = run(args)
            args["cache"] = True
        result = run(args)
        results.append(args | result)

with open(result_path, "w") as md_file:
    md_file.write('|')
    for head in heads:
        md_file.write(head + "|")
    md_file.write('\r\n')
    md_file.write('|')
    for _ in range(len(heads)):
        md_file.write('------|')
    md_file.write('\r\n')
    for result in results:
        md_file.write('|')
        for head in heads:
            md_file.write(str(result[head]) + "|")
        md_file.write('\r\n')