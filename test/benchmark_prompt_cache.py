import requests
import json
import operator
from functools import reduce
import argparse
import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from dataclasses import dataclass, field
import uuid
import pickle


num_workers = 15

def generate_stream(args):
    the_word = '龙'
    prefix = str(uuid.uuid4())[:8]
    first_prompt = prefix + the_word * args.first_input_len
    subsequent_prompt = the_word * args.subsequent_input_len
    prompt = first_prompt
    
    results = []
    responses = []
    has_error = False
    headers = {"Content-Type": "application/json"}
    for i in range(args.num_turns):
        try:
            start_time = time.time()
            r = requests.post(f"{args.address}/generate_stream",
                headers=headers,
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": args.output_len
                    }
                },
                stream=True
            )
            ans = ""
            last_time = start_time
            for chunk in r.iter_lines():
                if chunk == b'':
                    continue
                t = time.time()
                responses.append({
                    'latency': t - last_time,
                })
                data = json.loads(chunk.decode()[5:].strip())
                if data['generated_text']:
                    ans = data['generated_text']
                last_time = t
            prompt += ans + subsequent_prompt
        except Exception as e:
            print(e)
            has_error = True
            break
        finally:
            results.append({
                'start_time': start_time,
                'end_time': time.time(),
                'input': prompt,
                'max_new_tokens': args.output_len,
                'responses': responses,
                'has_error': has_error
            })
    return results

def conclusion_and_show(results):
    first_token_latency = []
    per_token_latency = []
    output_total_tokens = 0
    error_count = 0
    start_times = []
    end_times = []
    for e in results:
        if e['has_error']:
            error_count += 1
        else:
            start_times.append(e["start_time"])
            end_times.append(e["end_time"])
            tokens = e['responses']
            if len(tokens) == 0:
                error_count += 1
            else:
                first_token_latency.append(tokens[0]["latency"] * 1000) # ms
                per_token_latency.extend([e["latency"] * 1000 for e in tokens[1:]]) # ms
                output_total_tokens += len(tokens)

    print("test total_count", len(results))
    print("test error_count", error_count)
    print("output_total_tokens", output_total_tokens)
    print("qps ", len(results) / (np.max(end_times) - np.min(start_times)))

    percentiles = [25, 50, 75, 99, 100]
    values = np.percentile(first_token_latency, percentiles)
    for percentile, value in zip(percentiles, values):
        print("首字延迟 第{}% 分位数值：{:.2f}".format(percentile, value))

    values = np.percentile(per_token_latency, percentiles)
    for percentile, value in zip(percentiles, values):
        print("包间延迟 第{}% 分位数值：{:.2f}".format(percentile, value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", type=str, default="http://localhost:8080", help="server address")
    parser.add_argument("--model_name", type=str, default="model", help="for result file name")
    parser.add_argument("--num_workers", type=int, default=5, help="number of concurrent requests")
    parser.add_argument("--first_input_len", type=int, default=512, help="input length of the first turn of dialogue")
    parser.add_argument("--subsequent_input_len", type=int, default=512, help="input length of subsequent conversations")
    parser.add_argument("--output_len", type=int, default=128)
    parser.add_argument("--num_turns", type=int, default=10, help="number of dialogue turns per user")
    parser.add_argument("--num_users", type=int, default=10, help="number of users")
    parser.add_argument("--result_dir", type=str, default="./results", help="directory to save results")
    
    args = parser.parse_args()
    model_name = args.model_name
    num_workers = args.num_workers
    first_input_len = args.first_input_len
    subsequent_input_len = args.subsequent_input_len
    output_len = args.output_len
    num_turns = args.num_turns
    num_users = args.num_users

    result_file = f"{model_name}_{num_workers}_{first_input_len}_{subsequent_input_len}_{output_len}_{num_turns}_{num_users}.txt"
    result_path = os.path.join(args.result_dir, result_file)

    if os.path.isfile(result_path):
        with open(result_path, 'rb') as file:
            results = pickle.load(file)
    else:
        os.makedirs(args.result_dir, exist_ok=True)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(generate_stream, [args] * num_users), total=num_users, desc="running tests"))
            results = reduce(operator.add, results)
    
        with open(result_path, 'wb') as file:
            pickle.dump(results, file)

    conclusion_and_show(results)   
