#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <cmath>

#include "../tester/utils.h"

template <typename T>
__device__ __forceinline__ float to_float(T val) { return (float)val; }

template <>
__device__ __forceinline__ float to_float<half>(half val) { return __half2float(val); }

template <typename T>
__device__ __forceinline__ T from_float(float val) { return (T)val; }

template <>
__device__ __forceinline__ half from_float<half>(float val) { return __float2half(val); }

// --------------------------------------------------------------------------
// 1. Trace
// --------------------------------------------------------------------------

template <typename T>
struct TraceAcc{
    using Type = double;
};
template <>
struct TraceAcc<int>{
    using Type = int;
};

template <typename T>
__global__ void trace_kernel_standard(const T *input, T *output, size_t cols, size_t n){
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    typename TraceAcc<T>::Type local_sum = 0;

    for (size_t i = tid; i < n; i += stride){
        local_sum += (typename TraceAcc<T>::Type)input[i * cols + i];
    }

    // Shared Memory 归约
    // volatile 确保跨warp可见性，兼容天数平台
    __shared__ volatile typename TraceAcc<T>::Type s_data[256];
    s_data[tid] = local_sum;
    __syncthreads();

    // 树状归约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads(); 
    }
    if (tid == 0){
        *output = (T)s_data[0];
    }
}

template <typename T>
T trace(const std::vector<T> &h_input, size_t rows, size_t cols){
    size_t n = std::min(rows, cols);
    if (n == 0)
        return T(0);

    T *d_input = nullptr;
    T *d_result = nullptr;
    T h_result = 0;

    cudaMalloc(&d_input, h_input.size() * sizeof(T));
    cudaMalloc(&d_result, sizeof(T));

    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(T), cudaMemcpyHostToDevice);

    trace_kernel_standard<<<1, 256>>>(d_input, d_result, cols, n);

    cudaMemcpy(&h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_result);

    return h_result;
}

// --------------------------------------------------------------------------
// 2. Flash Attention
// --------------------------------------------------------------------------

#define CHUNK_SIZE 32

template <typename T>
__global__ void flash_attn_one_pass_hybrid_kernel(
    const T *__restrict__ Q,
    const T *__restrict__ K,
    const T *__restrict__ V,
    T *__restrict__ O,
    int src_seq_len, int head_dim, int kv_heads, int group_size,
    bool is_causal, float scale){
    int tid = threadIdx.x;
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_row = blockIdx.x;
    // GQA
    int kv_h = h / group_size;

    long long q_idx_base = ((long long)b * gridDim.x * gridDim.y + q_row * gridDim.y + h) * head_dim;
    long long kv_base_offset = ((long long)b * src_seq_len * kv_heads + kv_h) * head_dim;
    long long kv_stride = kv_heads * head_dim;

    // Shared Memory
    extern __shared__ float s_mem[];
    float *s_Q = s_mem;
    float *s_K = s_Q + head_dim;
    float *s_V = s_K + CHUNK_SIZE * head_dim;
    float *s_Scores = s_V + CHUNK_SIZE * head_dim;

    // 1. 加载Q到Shared Memory
    if (tid < head_dim){
        s_Q[tid] = to_float(Q[q_idx_base + tid]);
    }
    __syncthreads();

    // 统计量初始化
    float m_prev = -1e38f;
    float l_prev = 0.0f;
    float acc_o = 0.0f;

    for (int k_base = 0; k_base < src_seq_len; k_base += CHUNK_SIZE){
        int items = min(CHUNK_SIZE, src_seq_len - k_base);

        // A. 加载K V
        int num_elements = CHUNK_SIZE * head_dim;
        for (int i = tid; i < num_elements; i += blockDim.x){
            int r = i / head_dim;
            int c = i % head_dim;
            if (r < items){
                long long off = kv_base_offset + (k_base + r) * kv_stride + c;
                s_K[i] = to_float(K[off]);
                s_V[i] = to_float(V[off]);
            }
        }
        __syncthreads();

        // B. 计算分数 S = Q*K^T
        // 优化策略：前 32 个线程并行处理 32 个 Key，线程内部串行处理head_dim维度的点积
        // 保证点积运算的顺序与 CPU 完全一致
        if (tid < items){
            int k_curr = k_base + tid;

            // Causal Masking
            // 如果开启is_causal且当前Key位置 > Query位置，则mask掉
            if (is_causal && k_curr > q_row){
                s_Scores[tid] = -1e38f;
            }
            else{
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d){
                    dot += s_Q[d] * s_K[tid * head_dim + d];
                }
                s_Scores[tid] = dot * scale;
            }
        }
        __syncthreads();

        // C. Softmax 更新
        // 计算当前块的 max 和 sum
        // 为了稳健选择串行
        float m_block = -1e38f;

        if (tid == 0){
            for (int j = 0; j < items; ++j){
                m_block = fmaxf(m_block, s_Scores[j]);
            }
        }

        // 广播 m_block
        __shared__ float s_m_block;
        if (tid == 0)
            s_m_block = m_block;
        __syncthreads();
        m_block = s_m_block;

        // 如果整个块都被 mask，跳过更新
        if (m_block > -1e37f){
            float m_new = fmaxf(m_prev, m_block);
            float alpha = expf(m_prev - m_new);
            float block_scale = expf(m_block - m_new);

            // 计算当前块的Attention Prob, 并存回Shared Memory
            // P[j] = exp(S[j] - m_block) * block_scale = exp(S[j] - m_new)
            if (tid < items){
                s_Scores[tid] = expf(s_Scores[tid] - m_new);
            }
            __syncthreads();

            // Thread 0 计算当前块的 sum P
            float l_block = 0.0f;
            if (tid == 0){
                for (int j = 0; j < items; ++j){
                    l_block += s_Scores[j];
                }
            }
            __shared__ float s_l_block;
            if (tid == 0)
                s_l_block = l_block;
            __syncthreads();
            l_block = s_l_block;

            // D. 累加 Output
            // 每个线程负责 head_dim 中的一个维度
            // 线程内部串行遍历当前块的 32 个 Key，累加 P[j] * V[j]
            // O_new = O_old * alpha + sum(P_j * V_j)
            acc_o *= alpha;

            if (tid < head_dim){
                for (int j = 0; j < items; ++j){
                    acc_o += s_Scores[j] * s_V[j * head_dim + tid];
                }
            }

            // 更新运行总和
            l_prev = l_prev * alpha + l_block;
            m_prev = m_new;
        }
        __syncthreads();
    }

    // 最终归一化并写回
    if (tid < head_dim){
        if (l_prev > 0.0f){
            O[q_idx_base + tid] = from_float<T>(acc_o / l_prev);
        }
        else{
            O[q_idx_base + tid] = from_float<T>(0.0f);
        }
    }
}

template <typename T>
void flashAttention(const std::vector<T> &h_q, const std::vector<T> &h_k,
                    const std::vector<T> &h_v, std::vector<T> &h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal){

    size_t q_size = h_q.size() * sizeof(T);
    size_t k_size = h_k.size() * sizeof(T);
    size_t v_size = h_v.size() * sizeof(T);
    size_t o_size = h_o.size() * sizeof(T);

    if (h_o.size() != h_q.size())
        h_o.resize(h_q.size());

    T *d_q, *d_k, *d_v, *d_o;
    cudaMalloc(&d_q, q_size);
    cudaMalloc(&d_k, k_size);
    cudaMalloc(&d_v, v_size);
    cudaMalloc(&d_o, o_size);

    cudaMemcpy(d_q, h_q.data(), q_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), k_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), v_size, cudaMemcpyHostToDevice);

    float scale = 1.0f / sqrtf((float)head_dim);
    int group_size = query_heads / kv_heads;

    dim3 grid(target_seq_len, query_heads, batch_size);

    int blockSize = 128;
    while (blockSize < head_dim)
        blockSize += 32;
    if (blockSize < CHUNK_SIZE)
        blockSize = CHUNK_SIZE;

    // SMEM 大小 = Q + K + V + Scores
    size_t smem_size = (head_dim + 2 * CHUNK_SIZE * head_dim + CHUNK_SIZE) * sizeof(float);

    flash_attn_one_pass_hybrid_kernel<<<grid, blockSize, smem_size>>>(
        d_q, d_k, d_v, d_o,
        src_seq_len, head_dim, kv_heads, group_size,
        is_causal, scale);

    cudaMemcpy(h_o.data(), d_o, o_size, cudaMemcpyDeviceToHost);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
}

// ==========================================
// Explicit Template Instantiations
// ==========================================
template int trace<int>(const std::vector<int> &, size_t, size_t);
template float trace<float>(const std::vector<float> &, size_t, size_t);
template void flashAttention<float>(const std::vector<float> &, const std::vector<float> &,
                                    const std::vector<float> &, std::vector<float> &,
                                    int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half> &, const std::vector<half> &,
                                   const std::vector<half> &, std::vector<half> &,
                                   int, int, int, int, int, int, bool);