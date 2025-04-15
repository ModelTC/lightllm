DISABLE_CHECK_MAX_LEN_INFER=1 CUDA_VISIBLE_DEVICES=2 python -m lightllm.server.api_server --model_dir /mtc/chenjunyi/models/gemma-3-12b-it  \
                                     --host 0.0.0.0                 \
                                     --port 9999                   \
                                     --tp 1                        \
                                     --nccl_port 65535                \
				                     --data_type bf16   \
                                     --disable_chunked_prefill \
                                     --enable_multimodal \
                                     --disable_cudagraph \
				                     --trust_remote_code