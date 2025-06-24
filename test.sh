python3 test/benchmark_client.py \
  --url http://10.120.178.74:60011/generate \
  --num_clients 100 \
  --tokenizer_path /mtc/wufeiyang/Qwen2.5-72B-Instruct \
  --input_num 2000 \
  --input_len 1024 \
  --output_len 16384 \
  --server_api lightllm \
  --dump_file result.json \
  --seed 42