#ifndef CPUINFER_OPERATOR_DENSE_QWEN3_CPU_OPS_H
#define CPUINFER_OPERATOR_DENSE_QWEN3_CPU_OPS_H

#include "../../cpu_backend/worker_pool.h"
#include "llama.cpp/ggml.h"

namespace dense {

/**
 * @brief 对连续 FP16 行执行 Qwen3 使用的 RMSNorm。
 *
 * 输入和输出布局均为 [rows, hidden_dim]，支持 input 与 output 指向同一缓冲区。
 * 每一行由 WorkerPool 中的一个任务独立处理，平方和使用 FP32 累加。
 */
void rms_norm_fp16(const ggml_fp16_t* input, const ggml_fp16_t* weight,
                   ggml_fp16_t* output, int rows, int hidden_dim, float eps,
                   WorkerPool* backend);

/**
 * @brief 对 Qwen3 的 FP16 Query 和 Key 原地应用 split-half RoPE。
 *
 * q 布局为 [batch_size, q_len, q_head_num, head_dim]，k 布局为
 * [batch_size, q_len, kv_head_num, head_dim]。position_ids 布局为
 * [batch_size, q_len]，cos/sin 为连续的 [max_position, head_dim] 表。
 * 调用完成后，q 和 k 可以直接交给 Dense KVCache；写入 cache 的 Key 为 post-RoPE Key。
 */
void qwen3_rope_fp16(ggml_fp16_t* q, ggml_fp16_t* k,
                     const int* position_ids, const ggml_fp16_t* cos,
                     const ggml_fp16_t* sin, int batch_size, int q_len,
                     int q_head_num, int kv_head_num, int head_dim,
                     WorkerPool* backend);

}  // namespace dense

#endif
