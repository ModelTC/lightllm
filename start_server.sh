export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:~/.local/lib/
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
FLUX_SRC_DIR=${SCRIPT_DIR}

# add flux python package to PYTHONPATH
export NVSHMEM_BOOTSTRAP_MPI_PLUGIN=nvshmem_bootstrap_torch.so
export NVSHMEM_DISABLE_CUDA_VMM=1 # moving from cpp to shell
export CUDA_DEVICE_MAX_CONNECTIONS=1

# set default communication env vars
export BYTED_TORCH_BYTECCL=O0
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:=23}

nproc_per_node=4
nnodes=1
node_rank=0
master_addr="127.0.0.1"
master_port="23456"
additional_args="--rdzv_endpoint=${master_addr}:${master_port}"
IB_HCA=mlx5

export CUDA_VISIBLE_DEVICES=0,1,2,5,6,7
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:=3}
export NVSHMEM_IB_GID_INDEX=3
export LIGHTLLM_ENABLE_FLUX=0
export DISABLE_CHECK_MAX_LEN_INFER=1
export TORCHELASTIC_RUN_ID=$(date "+%Y%m%d%H%M%S642466")
export LIGHTLLM_USE_TPSP_MIX=1
export PATH=${PATH}:/usr/local/bin

CMD="/mnt/nvme0/chenjunyi/miniconda3/envs/lightllm-flux/bin/python -m lightllm.server.api_server --model_dir /mnt/nvme0/models/Meta-Llama-3.1-8B-Instruct  \
                                     --host 0.0.0.0                 \
                                     --port 8899                   \
                                     --tp ${nproc_per_node}         \
                                     --nccl_port 65535                \
				                     --data_type bf16   \
				                     --trust_remote_code  \
			                         --graph_max_batch_size 32 \
                                     --disable_cudagraph"
echo ${CMD}
/usr/local/cuda/nsight-systems-2023.4.4/bin/nsys profile --stats=true -o no-flux-report ${CMD}