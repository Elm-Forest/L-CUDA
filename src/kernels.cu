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
// 1. Trace Implementation (Ultra-Safe Serial Version)
// --------------------------------------------------------------------------
// 针对 BI100 优化：放弃多 Block 并行，改用单线程串行。
// 彻底消除了 atomicAdd，避免了原子锁死锁（Hang）和编译报错问题。
// 由于 N 通常较小，这种实现在高端卡上依然瞬间完成，且数值精度最高。

template <typename T>
__global__ void trace_kernel_serial(const T* input, T* output, size_t cols, size_t n) {
    // 仅由第 0 个 Block 的第 0 个线程执行
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double sum = 0.0;
        // 串行累加，与 CPU 行为完全一致，保证精度
        for (size_t i = 0; i < n; ++i) {
            sum += (double)input[i * cols + i];
        }
        *output = (T)sum;
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

    // 启动 1 个 Block，1 个线程。绝对稳定。
    trace_kernel_serial<<<1, 1>>>(d_input, d_result, cols, n);

    cudaMemcpy(&h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_result);

    return h_result;
}

// --------------------------------------------------------------------------
// 2. Flash Attention (Double-Precision Tiled One-Pass)
// --------------------------------------------------------------------------
// 策略：
// 1. 使用标准的 One-Pass 分块算法，逻辑简单，避免死锁。
// 2. 关键：所有中间累加器（Max, Sum, Output）均使用 double。
//    这能有效解决 float 测试中的精度误差 (Test 6/13/14)，同时不需要 Two-Pass 的复杂同步。

#define CHUNK_SIZE 32 // 保持为 Warp 大小，兼顾效率和 Shared Memory 限制

template <typename T>
__global__ void flash_attn_double_precision_kernel(
    const T* __restrict__ Q, 
    const T* __restrict__ K, 
    const T* __restrict__ V, 
    T* __restrict__ O,
    int src_seq_len, int head_dim, int kv_heads, int group_size, 
    bool is_causal, float scale
) {
    // Grid: (Tgt_Len, Heads, Batch)
    // Block: 128 threads (覆盖 head_dim)
    
    int tid = threadIdx.x;
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_row = blockIdx.x;
    int kv_h = h / group_size;

    // Offsets
    long long q_idx_base = ((long long)b * gridDim.x * gridDim.y + q_row * gridDim.y + h) * head_dim;
    long long kv_base_offset = ((long long)b * src_seq_len * kv_heads + kv_h) * head_dim;
    long long kv_stride = kv_heads * head_dim;

    // Shared Memory: 缓存 K 和 V 的块
    // Layout: K_chunk [CHUNK][Dim] | V_chunk [CHUNK][Dim]
    extern __shared__ float s_mem[];
    float* s_K = s_mem;
    float* s_V = s_mem + CHUNK_SIZE * head_dim;

    // 寄存器缓存 Q (每个线程持有一个 Q 的维度分量)
    // 注意：这里我们让每个线程处理 Head_Dim 的一个元素，并在循环中处理 K 
    float q_val = 0.0f;
    if (tid < head_dim) {
        q_val = to_float(Q[q_idx_base + tid]);
    }

    // 累加器使用 double 以保证极高精度
    double m_i = -1e100; // Global Max
    double l_i = 0.0;    // Global Sum
    double acc_o = 0.0;  // Global Output accumulator for this dimension

    // Loop over K/V Chunks
    for (int k_base = 0; k_base < src_seq_len; k_base += CHUNK_SIZE) {
        int items = min(CHUNK_SIZE, src_seq_len - k_base);

        // 1. Load K & V Chunks to Shared Memory
        int num_elements = CHUNK_SIZE * head_dim;
        for (int i = tid; i < num_elements; i += blockDim.x) {
            int r = i / head_dim;
            int c = i % head_dim;
            if (r < items) {
                long long off = kv_base_offset + (k_base + r) * kv_stride + c;
                s_K[i] = to_float(K[off]);
                s_V[i] = to_float(V[off]);
            } else {
                // Pad with zeros to avoid NaN issues
                s_K[i] = 0.0f;
                s_V[i] = 0.0f;
            }
        }
        __syncthreads();

        // 2. Compute Attention for this Chunk
        for (int j = 0; j < items; ++j) {
            int k_curr = k_base + j;
            if (is_causal && k_curr > q_row) continue;

            // Compute Dot Product: Q . K[j]
            // 每个线程计算一部分？不，我们需要完整的点积。
            // 这里使用更简单的方式：每个线程计算一个 d 分量的贡献，然后归约？太慢。
            // 我们让当前线程计算 Output[tid]，这需要遍历所有 j。
            // 为了算出 Softmax，我们需要先算出 Score[j]。
            // 这需要 Q 和 K[j] 的完整点积。
            
            // 为了避免复杂的 Block 归约，我们这里重复计算点积。
            // 虽然有冗余，但在 High-End GPU 上通常带宽是瓶颈，计算不是。
            // 且这种方式绝对无死锁。
            
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                // 从 Shared Mem 读取 K，Q 在寄存器或 Global
                // Q 我们只有 q_val (对应 tid)。我们需要整个 Q。
                // 这意味着上面的 q_val 策略不够。
                // 修正：我们需要 Q 的所有值。
                // 鉴于 Shared Mem 有限，且 Head Dim 不大（<=128），
                // 我们在内层循环直接读取 s_K 即可，但 Q 最好也在 Shared 或重读。
                // 考虑到 head_dim 很小，我们可以在循环里直接通过 Global Memory 读取 Q[d] ? 
                // 或者，我们假设 tid < head_dim，我们无法拥有所有 Q。
                
                // 为了简单且正确，我们需要 Score。
                // 让我们使用 Shared Memory 存储 Q。
                // 重新规划 Shared Memory: Q | K | V
            }
        }
    }
}
// 上面的 Kernel 逻辑写到一半发现 One-Pass 在并行 Output 维度时，计算 Score 比较麻烦。
// 让我们换回最稳的 "Naive Block-Parallel" (每个 Block 处理一行 Q)
// 并且使用 Shared Mem 缓存 Q, K, V。