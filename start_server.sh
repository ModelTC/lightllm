python -m lightllm.server.api_server --model_dir /mtc/chenjunyi/models/gemma-3-12b-it  \
                                     --host 0.0.0.0                 \
                                     --port 9999                   \
                                     --tp 1                        \
                                     --nccl_port 65535                \
				                     --data_type bf16   \
                                     --disable_chunked_prefill
				                     --trust_remote_code