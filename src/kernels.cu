#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <cmath>

#include "../tester/utils.h"

// --------------------------------------------------------------------------
// Helper Functions
// --------------------------------------------------------------------------

template<typename T>
__device__ __forceinline__ float to_float(T val) { return (float)val; }

template<>
__device__ __forceinline__ float to_float<half>(half val) { return __half2float(val); }

template<typename T>
__device__ __forceinline__ T from_float(float val) { return (T)val; }

template<>
__device__ __forceinline__ half from_float<half>(float val) { return __float2half(val); }

// --------------------------------------------------------------------------
// 1. Trace Implementation (Robust Single-Block Version)
// --------------------------------------------------------------------------

template <typename T>
__global__ void trace_kernel_single_block(const T* input, T* output, size_t cols, size_t n) {
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    // 内部使用 double 累加保证精度
    double local_sum = 0.0;
    
    // Grid-Stride Loop
    for (size_t i = tid; i < n; i += stride) {
        local_sum += (double)input[i * cols + i];
    }

    // [关键修改] 使用 volatile 确保 Shared Memory 在 Warp 内外的可见性
    // 防止非 NV 编译器过度优化导致归约逻辑错误
    __shared__ volatile double s_data[256];
    s_data[tid] = local_sum;
    __syncthreads();

    // 树状归约 (带有 volatile 保证)
    if (blockDim.x >= 256) { if (tid < 128) { s_data[tid] += s_data[tid + 128]; } __syncthreads(); }
    if (blockDim.x >= 128) { if (tid < 64)  { s_data[tid] += s_data[tid + 64]; } __syncthreads(); }
    
    // Warp 内归约通常不需要 __syncthreads，但在异构兼容时加上更稳妥
    if (tid < 32) {
        if (blockDim.x >= 64) s_data[tid] += s_data[tid + 32];
        if (blockDim.x >= 32) s_data[tid] += s_data[tid + 16];
        if (blockDim.x >= 16) s_data[tid] += s_data[tid + 8];
        if (blockDim.x >= 8)  s_data[tid] += s_data[tid + 4];
        if (blockDim.x >= 4)  s_data[tid] += s_data[tid + 2];
        if (blockDim.x >= 2)  s_data[tid] += s_data[tid + 1];
    }

    // 单线程写回
    if (tid == 0) {
        *output = (T)s_data[0];
    }
}

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
    size_t n = std::min(rows, cols);
    if (n == 0) return T(0);

    T* d_input = nullptr;
    T* d_result = nullptr;
    T h_result = 0;

    cudaMalloc(&d_input, h_input.size() * sizeof(T));
    cudaMalloc(&d_result, sizeof(T));
    
    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(T), cudaMemcpyHostToDevice);

    // 仅启动 1 个 Block，256 个线程
    trace_kernel_single_block<<<1, 256>>>(d_input, d_result, cols, n);

    cudaMemcpy(&h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_result);

    return h_result;
}

// --------------------------------------------------------------------------
// 2. Flash Attention Implementation (Safe Hybrid Parallel)
// --------------------------------------------------------------------------

// 保持 CHUNK_SIZE = 32，这对所有平台最安全
#define CHUNK_SIZE 32

template <typename T>
__global__ void flash_attn_hybrid_kernel(
    const T* __restrict__ Q, 
    const T* __restrict__ K, 
    const T* __restrict__ V, 
    T* __restrict__ O,
    int src_seq_len, int head_dim, int kv_heads, int group_size, 
    bool is_causal, float scale
) {
    int tid = threadIdx.x;
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_row = blockIdx.x;
    int kv_h = h / group_size;

    long long q_idx_base = ((long long)b * gridDim.x * gridDim.y + q_row * gridDim.y + h) * head_dim;
    long long kv_base_offset = ((long long)b * src_seq_len * kv_heads + kv_h) * head_dim;
    long long kv_stride = kv_heads * head_dim;

    // Shared Memory Layout
    extern __shared__ float s_mem[];
    float* s_Q = s_mem;                        
    float* s_K = s_Q + head_dim;               
    float* s_Scores = s_K + CHUNK_SIZE * head_dim; 
    
    // Pass 2 Reuse
    float* s_V = s_K + CHUNK_SIZE * head_dim; 
    float* s_Scores_Safe = s_V + CHUNK_SIZE * head_dim;

    // Load Q
    if (tid < head_dim) {
        s_Q[tid] = to_float(Q[q_idx_base + tid]);
    }
    __syncthreads();

    // =============================================================
    // PASS 1: Statistics (Max, Sum)
    // =============================================================
    
    float m_global = -1e38f;
    float l_global = 0.0f;

    for (int k_base = 0; k_base < src_seq_len; k_base += CHUNK_SIZE) {
        int items = min(CHUNK_SIZE, src_seq_len - k_base);

        // Load K Chunk
        int num_elements = CHUNK_SIZE * head_dim;
        for (int i = tid; i < num_elements; i += blockDim.x) {
            int r = i / head_dim;
            int c = i % head_dim;
            if (r < items) {
                s_K[i] = to_float(K[kv_base_offset + (k_base + r) * kv_stride + c]);
            }
        }
        __syncthreads();

        // Compute Scores
        if (tid < items) {
            int k_curr = k_base + tid;
            if (is_causal && k_curr > q_row) {
                s_Scores_Safe[tid] = -1e38f; 
            } else {
                float dot = 0.0f;
                // [关键] 移除 pragma unroll 以避免特定编译器的寄存器溢出风险
                for (int d = 0; d < head_dim; ++d) {
                    dot += s_Q[d] * s_K[tid * head_dim + d];
                }
                s_Scores_Safe[tid] = dot * scale;
            }
        }
        __syncthreads();

        // Update Stats (Serial)
        if (tid == 0) {
            for (int j = 0; j < items; ++j) {
                float val = s_Scores_Safe[j];
                if (val > -1e37f) {
                    // [关键修改] 使用 fmaxf 替代 max，确保浮点行为一致
                    float new_m = fmaxf(m_global, val);
                    float exp_delta = expf(m_global - new_m);
                    float exp_val = expf(val - new_m);
                    l_global = l_global * exp_delta + exp_val;
                    m_global = new_m;
                }
            }
        }
        __syncthreads();
    }

    __shared__ float final_m;
    __shared__ float final_l;
    if (tid == 0) {
        final_m = m_global;
        final_l = l_global;
    }
    __syncthreads();

    if (final_l <= 0.0f) {
        if (tid < head_dim) O[q_idx_base + tid] = from_float<T>(0.0f);
        return;
    }

    // =============================================================
    // PASS 2: Output Aggregation
    // =============================================================
    
    float acc_o = 0.0f; 

    for (int k_base = 0; k_base < src_seq_len; k_base += CHUNK_SIZE) {
        int items = min(CHUNK_SIZE, src_seq_len - k_base);

        // Load K and V
        int num_elements = CHUNK_SIZE * head_dim;
        for (int i = tid; i < num_elements; i += blockDim.x) {
            int r = i / head_dim;
            int c = i % head_dim;
            if (r < items) {
                long long off = kv_base_offset + (k_base + r) * kv_stride + c;
                s_K[i] = to_float(K[off]);
                s_V[i] = to_float(V[off]); 
            }
        }
        __syncthreads();

        // Re-Compute Scores
        if (tid < items) {
            int k_curr = k_base + tid;
            if (is_causal && k_curr > q_row) {
                s_Scores_Safe[tid] = -1e38f; 
            } else {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += s_Q[d] * s_K[tid * head_dim + d];
                }
                s_Scores_Safe[tid] = dot * scale;
            }
        }
        __syncthreads();

        // Accumulate Output
        if (tid < head_dim) {
            for (int j = 0; j < items; ++j) {
                float val = s_Scores_Safe[j];
                if (val > -1e37f) {
                    float weight = expf(val - final_m);
                    acc_o += weight * s_V[j * head_dim + tid];
                }
            }
        }
        __syncthreads();
    }

    if (tid < head_dim) {
        O[q_idx_base + tid] = from_float<T>(acc_o / final_l);
    }
}


template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
    
    size_t q_size = h_q.size() * sizeof(T);
    size_t k_size = h_k.size() * sizeof(T);
    size_t v_size = h_v.size() * sizeof(T);
    size_t o_size = h_o.size() * sizeof(T); 
    
    if (h_o.size() != h_q.size()) h_o.resize(h_q.size());

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
    
    // Block Size 设为 128，安全配置
    int blockSize = 128;
    while (blockSize < head_dim) blockSize += 32;
    if (blockSize < CHUNK_SIZE) blockSize = CHUNK_SIZE;
    
    size_t smem_size = (head_dim + 2 * CHUNK_SIZE * head_dim + CHUNK_SIZE) * sizeof(float);

    flash_attn_hybrid_kernel<<<grid, blockSize, smem_size>>>(
        d_q, d_k, d_v, d_o, 
        src_seq_len, head_dim, kv_heads, group_size, 
        is_causal, scale
    );

    cudaMemcpy(h_o.data(), d_o, o_size, cudaMemcpyDeviceToHost);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
}

// ==========================================
// Explicit Template Instantiations
// ==========================================
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);