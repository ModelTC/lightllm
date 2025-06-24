python3 -m lightllm.server.api_server \
    --model_dir /mtc/wufeiyang/Qwen2.5-72B-Instruct \
    --run_mode "pd_master" \
    --host 10.120.178.74 \
    --port 60011 \
    --pd_chunk_size 4096