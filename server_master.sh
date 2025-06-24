python3 -m lightllm.server.api_server \
    --model_dir /mnt/youwei-data/zhuohang/model/Qwen/Qwen2.5-14B \
    --run_mode "pd_master" \
    --host 127.0.1.1 \
    --port 60011 \
    --pd_chunk_size 4096