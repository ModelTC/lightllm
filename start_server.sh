export DISABLE_CHECK_MAX_LEN_INFER=1

python -m lightllm.server.api_server --model_dir /mtc/chenjunyi/models/qwen25_7b  \
                                     --host 0.0.0.0                 \
                                     --port 8888                   \
                                     --tp 1                        \
                                     --nccl_port 65535                \
				                     --data_type bf16   \
				                     --trust_remote_code  \
                                     --tool_call_parser qwen25 \
			                         --disable_cudagraph