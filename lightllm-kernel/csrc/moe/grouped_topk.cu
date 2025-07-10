#include <cub/cub.cuh>
#include <torch/extension.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "../cuda_compat.h"

#ifndef USE_ROCM
    #include <cub/util_type.cuh>
    #include <cub/cub.cuh>
#else
    #include <hipcub/util_type.hpp>
    #include <hipcub/hipcub.hpp>
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace lightllm {
namespace ops{

template <int TPB>
__launch_bounds__(TPB) 
__global__ void moeGroupedTopK(
    const float* input, 
    const bool* finished, 
    float* inputs_after_softmax, 
    const int num_cols, 
    const float* correction_bias, 
    float* group_scores, 
    float* output, // topk_weights
    int* indices, // topk_indices
    int* group_indices, // token_expert_indices
    const int num_experts, 
    const int num_expert_group, 
    const int topk_group,
    const int k,
    const bool renormalize,
    const bool softmax_or_sigmoid, 
    const int start_expert, 
    const int end_expert)
{

    const int thread_row_offset = blockIdx.x * num_cols;

    if(softmax_or_sigmoid)
    {
        //softmax
        using BlockReduce_topk = cub::BlockReduce<float, TPB>;
        __shared__ typename BlockReduce_topk::TempStorage tmpStorage;

        __shared__ float normalizing_factor;
        __shared__ float float_max;

        cub::Sum sum;
        float threadData(-FLT_MAX);

        // Don't touch finished rows.
        if ((finished != nullptr) && finished[blockIdx.x])
        {
            return;
        }

        for (int i = threadIdx.x; i < num_cols; i += TPB)
        {
            const int idx = thread_row_offset + i;
            threadData = max(static_cast<float>(input[idx]), threadData);
        }

        const float maxElem = BlockReduce_topk(tmpStorage).Reduce(threadData, cub::Max());
        if (threadIdx.x == 0)
        {
            float_max = maxElem;
        }
        __syncthreads();

        threadData = 0;

        for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
        {
            const int idx = thread_row_offset + ii;
            threadData += exp((static_cast<float>(input[idx]) - float_max));
        }

        const auto Z = BlockReduce_topk(tmpStorage).Reduce(threadData, sum);

        if (threadIdx.x == 0)
        {
            normalizing_factor = 1.f / Z;
        }
        __syncthreads();

        for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
        {
            const int idx = thread_row_offset + ii;
            const float val = exp((static_cast<float>(input[idx]) - float_max)) * normalizing_factor;
            inputs_after_softmax[idx] = val + (correction_bias ? correction_bias[idx] : 0.f);
        }
    } else {
        // sigmoid
        for (int i = threadIdx.x; i < num_cols; i += TPB)
        {
            const int idx = thread_row_offset + i;
            float val = 1.f / (1.f + expf(-input[idx])); 
            inputs_after_softmax[idx] = val + (correction_bias ? correction_bias[idx] : 0.f);
        }
    }
    __syncthreads();

    using cub_kvp = cub::KeyValuePair<int, float>;
    using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage_kvp;

    int block_row = blockIdx.x;  // (0 - tokens-1) 即0-199
    int thread_read_offset = block_row * num_experts;

    int group_size = num_experts / num_expert_group;

    for(int group_id = threadIdx.x; group_id < num_expert_group; group_id += TPB)
    {
        float local_max = -FLT_MAX;
        const int start = group_id * group_size;
        const int end   = (group_id + 1) * group_size;

        // find max in this group
        for(int e = start; e < end; e++)
        {
            float val = inputs_after_softmax[thread_read_offset + e];
            local_max = fmaxf(local_max, val);
        }

        // store max in group_scores
        group_scores[block_row * num_expert_group + group_id] = local_max;
    }
    __syncthreads();

    cub_kvp thread_kvp;
    cub::ArgMax arg_max;

    const bool row_is_active = finished ? !finished[block_row] : true;
    thread_read_offset = blockIdx.x * num_expert_group;

    for (int k_idx = 0; k_idx < topk_group; ++k_idx)
    {
        thread_kvp.key = 0;
        thread_kvp.value = -1.f; // This is OK because inputs are probabilities

        // every thread finds the max expert in a different expert group
        cub_kvp inp_kvp;
        for (int expert = threadIdx.x; expert < num_expert_group; expert += TPB)
        {
            const int idx = thread_read_offset + expert;
            inp_kvp.key = expert;
            inp_kvp.value = group_scores[idx];

            for (int prior_k = 0; prior_k < k_idx; ++prior_k)
            {
                const int prior_winning_expert = group_indices[topk_group * block_row + prior_k]; 

                if (prior_winning_expert == expert)
                {
                    inp_kvp = thread_kvp;
                }
            }

            thread_kvp = arg_max(inp_kvp, thread_kvp);
        }

        const cub_kvp result_kvp = BlockReduce(tmpStorage_kvp).Reduce(thread_kvp, arg_max);
        if (threadIdx.x == 0)
        {
            // Ignore experts the node isn't responsible for with expert parallelism
            const int expert = result_kvp.key;
            const bool node_uses_expert = expert >= start_expert && expert < end_expert;
            const bool should_process_row = row_is_active && node_uses_expert;

            const int idx = topk_group * block_row + k_idx;
            group_indices[idx] = should_process_row ? (expert - start_expert) : num_expert_group;
            assert(group_indices[idx] >= 0);
        }
        __syncthreads();
    }

    int score_offset = block_row * num_experts; 
    for (int e = threadIdx.x; e < num_experts; e += TPB)
    {
        int grp = e / group_size;
        bool selected = false;
        // selected = True if e in group_indices[block_row, :]
        for (int i = 0; i < topk_group; i++) {
            int sel_grp = group_indices[block_row * topk_group + i];
            if (sel_grp == grp) {
                selected = true;
                break;
            }
        }
        if (!selected) {
            inputs_after_softmax[score_offset + e] = 0.0f;
        }
    }
    __syncthreads();

    for (int tk = 0; tk < k; tk++) {
        thread_kvp.key = -1;
        thread_kvp.value = -FLT_MAX;
        for (int e = threadIdx.x; e < num_experts; e += TPB) {
            bool already_selected = false;
            for (int prev = 0; prev < tk; prev++) {
                if (indices[block_row * k + prev] == e) {
                    already_selected = true;
                    break;
                }
            }
            float val = already_selected ? -FLT_MAX : inputs_after_softmax[score_offset + e];
            cub_kvp inp;
            inp.key = e;
            inp.value = val;
            thread_kvp = arg_max(inp, thread_kvp);
        }
        cub_kvp result = BlockReduce(tmpStorage_kvp).Reduce(thread_kvp, arg_max);
        if (threadIdx.x == 0) {
            output[block_row * k + tk] = result.value;
            indices[block_row * k + tk] = result.key;
        }
        __syncthreads();
    }

    // renormalize
    if (threadIdx.x == 0 && renormalize) {
        float sum = 0.0f;
        int out_offset = block_row * k;
        for (int j = 0; j < k; j++) {
            sum += output[out_offset + j];
        }
        // avoid division by zero
        if (sum > 0.0f) {
            for (int j = 0; j < k; j++) {
                output[out_offset + j] /= sum;
            }
        }
    }
    __syncthreads();

}

void GroupedTopKKernelLauncher(
    const float* gating_output,
    const float* correction_bias,
    float* topk_weights,
    int* topk_indicies,
    int* group_indices,
    float* softmax_workspace,
    float* group_scores,
    const int num_tokens,
    const int num_experts,
    const int num_expert_group,
    const int topk_group,
    const int topk,
    const bool renormalize,
    const bool softmax_or_sigmoid,
    cudaStream_t stream) {

    static constexpr int TPB = 256;
    moeGroupedTopK<TPB><<<num_tokens, TPB, 0, stream>>>(
        gating_output, nullptr, softmax_workspace, num_experts, correction_bias,
        group_scores, topk_weights, topk_indicies, group_indices,
        num_experts, num_expert_group, topk_group, topk, renormalize, softmax_or_sigmoid, 0, num_experts);
}

void grouped_topk_cuda(
    torch::Tensor& topk_weights,                // [num_tokens, topk]
    torch::Tensor& correction_bias,             // [num_tokens, num_experts]
    torch::Tensor& topk_indices,                // [num_tokens, topk]
    torch::Tensor& group_indices,               // [num_tokens, topk_group]
    torch::Tensor& gating_output,               // [num_tokens, num_experts]
    const int num_expert_group,
    const int topk_group,
    const int topk,
    const bool renormalize,
    std::string scoring_func,
    torch::Tensor group_scores = torch::Tensor() // [num_tokens, num_expert_group]
    )
{
    const int num_experts = gating_output.size(-1);
    const int num_tokens = gating_output.numel() / num_experts;

    const int64_t workspace_size = num_tokens * num_experts;

    const bool softmax_or_sigmoid = (scoring_func == "softmax") ? true : false;

    float* d_group_scores = nullptr;
    if (group_scores.defined() && group_scores.numel() > 0) {
        d_group_scores = group_scores.data_ptr<float>();
    } else {
        cudaMalloc(&d_group_scores, num_tokens * num_expert_group * sizeof(float));
        cudaMemset(d_group_scores, 0, num_tokens * num_expert_group * sizeof(float));
    }

    const at::cuda::OptionalCUDAGuard device_guard(device_of(gating_output));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor softmax_workspace = torch::empty({workspace_size}, gating_output.options());
    GroupedTopKKernelLauncher(
        gating_output.data_ptr<float>(),
        correction_bias.defined() ? correction_bias.data_ptr<float>() : nullptr,
        topk_weights.data_ptr<float>(),
        topk_indices.data_ptr<int>(),
        group_indices.data_ptr<int>(),
        softmax_workspace.data_ptr<float>(),
        d_group_scores,
        num_tokens,
        num_experts,
        num_expert_group,
        topk_group,
        topk,
        renormalize,
        softmax_or_sigmoid,
        stream);
}

torch::Tensor grouped_topk(
        torch::Tensor topk_weights,
        torch::Tensor correction_bias,
        torch::Tensor topk_indices,
        torch::Tensor group_indices,
        torch::Tensor gating_output,
        int64_t  num_expert_group,
        int64_t  topk_group,
        int64_t  topk,
        bool     renormalize,
        std::string scoring_func,
        torch::Tensor group_scores) {

    grouped_topk_cuda(topk_weights, correction_bias, topk_indices, group_indices,
                      gating_output,
                      static_cast<int>(num_expert_group),
                      static_cast<int>(topk_group),
                      static_cast<int>(topk),
                      renormalize, scoring_func, group_scores);

    return topk_weights;
}

} // namespace ops
} // namespace lightllm