#include "dense_kvcache.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>

#include "ggml-impl.h"
#include "llamafile/sgemm.h"

namespace dense {

/**
 * @brief 按 KV head 独立计算分块 Attention。
 *
 * 每个工作窃取任务处理一个 batch、一个 KV head 和一个 cache block。函数为共享该 KV head 的全部
 * GQA Query heads 计算精确 Attention，对未填满的尾块施加 mask，并通过 log-sum-exp 缩放合并各 block，
 * 使结果等价于在 block_table 中的全部有效 block 逻辑拼接后执行一次 softmax。
 * 这个函数来自原 attention_kvhead_()，删除了稀疏检索表和量化分支，但完整保留 block 级并行及归并策略。
 *
 * @param q_in_data FP16 Query 数据，布局为 [batch_size, q_head_num, head_dim]。
 * @param output FP16 Attention 输出，逻辑布局与 q_in_data 相同。
 * @param attn_lse FP32 log-sum-exp 输出，每个 batch、每个 Query head 对应一个值。
 * @param batch_size 要处理的 Query 行数；调用方已经将 q_len 折叠到该维度中。
 * @param backend 用于并行执行 block Attention 和线程结果归并的工作线程池。
 */
void KVCache::attention_kvhead_(const uint16_t* q_in_data, ggml_fp16_t* output,
                                float* attn_lse, int batch_size,
                                WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();
  seq_len_ = config_.block_len;
  backend->do_work_stealing_job(
      batch_size * config_.kv_head_num * block_num_per_seq_,
      // Initial function
      [&](int thread_id) {
        thread_cur_head_idx_[thread_id].first = -1;
        thread_cur_head_idx_[thread_id].second = -1;
      },

      // Compute function
      [&](int task_id) {
        // 目前在哪个batch
        int batch_id = task_id / (config_.kv_head_num * block_num_per_seq_);

        // 先取%，得到一个task在batch内的相对位置，再除以block_num_per_seq_，得到当前task对应的head_id
        int head_id =
            (task_id % (config_.kv_head_num * block_num_per_seq_)) /
            block_num_per_seq_;
        
        // 当前处理的block_id，
        int block_id = task_id % block_num_per_seq_;
        int thread_id = WorkerPool::thread_local_id;

        // 如果当前block_id超出了该batch的有效block数量，则直接返回
        if (cache_seqlens_[batch_id] / config_.block_len < block_id) return;

        // 根据 block_id 查找实际在 block_table 中的索引
        int block_idx = block_table_[batch_id * block_num_per_seq_ + block_id];

        // 处理最后一个没有被填满的block
        if (cache_seqlens_[batch_id] / config_.block_len == block_id) {
          int seq_len = cache_seqlens_[batch_id] % config_.block_len;
          if (seq_len == 0) return;
          int full_blocks = seq_len / 8;
          int remaining_bits = seq_len % 8;
          for (int i = 0; i < full_blocks; ++i) thread_local_attn_mask_[thread_id][i] = 0xFF;
          if (remaining_bits > 0 && full_blocks < seq_len_ / 8) {
            thread_local_attn_mask_[thread_id][full_blocks] = (1 << remaining_bits) - 1;
          } else {
            thread_local_attn_mask_[thread_id][full_blocks] = 0;
          }
          for (int i = full_blocks + 1; i < seq_len_ / 8; ++i) {
            thread_local_attn_mask_[thread_id][i] = 0;
          }
          attn_with_kvcache_one_block_(
              config_.head_dim, config_.q_head_num / config_.kv_head_num,
              GGML_TYPE_F16,
              (void*)&q_in_data[batch_id * config_.kv_head_num * n_gqa_ * config_.head_dim +
                                head_id * n_gqa_ * config_.head_dim],
              seq_len_, 0, false, thread_local_attn_mask_[thread_id].data(),
              GGML_TYPE_F16,
              k_cache_fp16_[layer_id_][head_id][block_idx].data(),
              GGML_TYPE_F16,
              v_cache_fp16_[layer_id_][head_id][block_idx].data(),
              thread_local_attn_score_[thread_id].data(),
              thread_local_output_fp32_[thread_id].data(),
              thread_local_attn_lse_[thread_id].data(),
              thread_local_draft_[thread_id].data());
        } 
        
        // 处理第0个block，到倒数第二个满block, 这里的倒数第二个不是计算机科学家计数法
        else {
          attn_with_kvcache_one_block_(
              config_.head_dim, config_.q_head_num / config_.kv_head_num,
              GGML_TYPE_F16,
              (void*)&q_in_data[batch_id * config_.kv_head_num * n_gqa_ * config_.head_dim +
                                head_id * n_gqa_ * config_.head_dim],
              seq_len_, 0, true, nullptr, GGML_TYPE_F16,
              k_cache_fp16_[layer_id_][head_id][block_idx].data(),
              GGML_TYPE_F16,
              v_cache_fp16_[layer_id_][head_id][block_idx].data(),
              thread_local_attn_score_[thread_id].data(),
              thread_local_output_fp32_[thread_id].data(),
              thread_local_attn_lse_[thread_id].data(),
              thread_local_draft_[thread_id].data());
        }

        // 一个线程在完成一个任务的局部 Attention 计算后，先进行局部reduce
        int cur_batch_idx = thread_cur_head_idx_[thread_id].first;
        int cur_head_id = thread_cur_head_idx_[thread_id].second;
        if (batch_id == cur_batch_idx && head_id == cur_head_id) {
          for (int i = 0; i < n_gqa_; i++) {
            float new_attn_lse = thread_local_cur_attn_lse_[thread_id][i] +
                                 std::log(1.0 + std::exp(thread_local_attn_lse_[thread_id][i] -
                                                         thread_local_cur_attn_lse_[thread_id][i]));
            ggml_vec_scale_f32(
                config_.head_dim,
                thread_local_cur_output_fp32_[thread_id].data() + i * config_.head_dim,
                std::exp(thread_local_cur_attn_lse_[thread_id][i] - new_attn_lse));
            ggml_vec_scale_f32(
                config_.head_dim,
                thread_local_output_fp32_[thread_id].data() + i * config_.head_dim,
                std::exp(thread_local_attn_lse_[thread_id][i] - new_attn_lse));
            for (int j = 0; j < config_.head_dim; j++) {
              thread_local_cur_output_fp32_[thread_id][i * config_.head_dim + j] +=
                  thread_local_output_fp32_[thread_id][i * config_.head_dim + j];
            }
            thread_local_cur_attn_lse_[thread_id][i] = new_attn_lse;
          }
        } 
        
        // 当线程切换到不同的 head/ batch 时，则需要将之前的局部结果归约到全局
        else {
          if (cur_batch_idx != -1) {
            mutex_[cur_batch_idx][cur_head_id]->lock();
            for (int i = 0; i < n_gqa_; i++) {
              if (std::abs(attn_lse_[cur_batch_idx][cur_head_id][i]) < 1e-6) {
                attn_lse_[cur_batch_idx][cur_head_id][i] =
                    thread_local_cur_attn_lse_[thread_id][i];
                for (int j = 0; j < config_.head_dim; j++) {
                  output_fp32_[cur_batch_idx][cur_head_id][i * config_.head_dim + j] =
                      thread_local_cur_output_fp32_[thread_id][i * config_.head_dim + j];
                }
                continue;
              }
              float new_attn_lse = attn_lse_[cur_batch_idx][cur_head_id][i] +
                                   std::log(1.0 + std::exp(
                                                      thread_local_cur_attn_lse_[thread_id][i] -
                                                      attn_lse_[cur_batch_idx][cur_head_id][i]));
              ggml_vec_scale_f32(
                  config_.head_dim,
                  output_fp32_[cur_batch_idx][cur_head_id].data() + i * config_.head_dim,
                  std::exp(attn_lse_[cur_batch_idx][cur_head_id][i] - new_attn_lse));
              ggml_vec_scale_f32(
                  config_.head_dim,
                  thread_local_cur_output_fp32_[thread_id].data() + i * config_.head_dim,
                  std::exp(thread_local_cur_attn_lse_[thread_id][i] - new_attn_lse));
              for (int j = 0; j < config_.head_dim; j++) {
                output_fp32_[cur_batch_idx][cur_head_id][i * config_.head_dim + j] +=
                    thread_local_cur_output_fp32_[thread_id][i * config_.head_dim + j];
              }
              attn_lse_[cur_batch_idx][cur_head_id][i] = new_attn_lse;
            }
            mutex_[cur_batch_idx][cur_head_id]->unlock();
          }
          thread_cur_head_idx_[thread_id].first = batch_id;
          thread_cur_head_idx_[thread_id].second = head_id;
          for (int i = 0; i < n_gqa_; i++) {
            thread_local_cur_attn_lse_[thread_id][i] =
                thread_local_attn_lse_[thread_id][i];
            for (int j = 0; j < config_.head_dim; j++) {
              thread_local_cur_output_fp32_[thread_id][i * config_.head_dim + j] =
                  thread_local_output_fp32_[thread_id][i * config_.head_dim + j];
            }
          }
        }
      },

      // Final function，处理各个线程的reduce工作
      [&](int thread_id) {
        int cur_batch_idx = thread_cur_head_idx_[thread_id].first;
        int cur_head_id = thread_cur_head_idx_[thread_id].second;
        if (cur_head_id != -1) {
          mutex_[cur_batch_idx][cur_head_id]->lock();
          for (int i = 0; i < n_gqa_; i++) {
            float new_attn_lse;
            if (std::abs(attn_lse_[cur_batch_idx][cur_head_id][i]) < 1e-6) {
              attn_lse_[cur_batch_idx][cur_head_id][i] =
                  thread_local_cur_attn_lse_[thread_id][i];
              for (int j = 0; j < config_.head_dim; j++) {
                output_fp32_[cur_batch_idx][cur_head_id][i * config_.head_dim + j] =
                    thread_local_cur_output_fp32_[thread_id][i * config_.head_dim + j];
              }
              continue;
            }
            new_attn_lse = attn_lse_[cur_batch_idx][cur_head_id][i] +
                           std::log(1.0 + std::exp(
                                              thread_local_cur_attn_lse_[thread_id][i] -
                                              attn_lse_[cur_batch_idx][cur_head_id][i]));
            ggml_vec_scale_f32(
                config_.head_dim,
                output_fp32_[cur_batch_idx][cur_head_id].data() + i * config_.head_dim,
                std::exp(attn_lse_[cur_batch_idx][cur_head_id][i] - new_attn_lse));
            ggml_vec_scale_f32(
                config_.head_dim,
                thread_local_cur_output_fp32_[thread_id].data() + i * config_.head_dim,
                std::exp(thread_local_cur_attn_lse_[thread_id][i] - new_attn_lse));
            for (int j = 0; j < config_.head_dim; j++) {
              output_fp32_[cur_batch_idx][cur_head_id][i * config_.head_dim + j] +=
                  thread_local_cur_output_fp32_[thread_id][i * config_.head_dim + j];
            }
            attn_lse_[cur_batch_idx][cur_head_id][i] = new_attn_lse;
          }
          mutex_[cur_batch_idx][cur_head_id]->unlock();
        }
      });

  
  
  // 将 FP32 输出和 LSE 转换为 FP16，并按 batch、KV head、GQA、head_dim 展开到输出缓冲区
  uint16_t* output_data = reinterpret_cast<uint16_t*>(output);
  float* attn_lse_data = attn_lse;
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    for (int i = 0; i < config_.kv_head_num; i++) {
      for (int j = 0; j < n_gqa_ * config_.head_dim; j++) {
        output_data[batch_idx * config_.kv_head_num * n_gqa_ * config_.head_dim +
                    i * n_gqa_ * config_.head_dim + j] =
            GGML_FP32_TO_FP16(output_fp32_[batch_idx][i][j]);
      }
      for (int j = 0; j < n_gqa_; j++) {
        attn_lse_data[batch_idx * config_.kv_head_num * n_gqa_ + i * n_gqa_ + j] =
            attn_lse_[batch_idx][i][j];
      }
    }
  }

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
}

/**
 * @brief 重置 Attention 累加状态，并导入完整的 KV-head block 表，初始化 Attention 输出
 * 
 * 这里的输出形状: (batch_size, head_num, head_dim)
 * 计算结束后，把后两个维度展开, 并进行向下矩阵乘, 得到: (batch_size, hidden_size)
 *
 * 函数清零每个 batch、KV head 的 FP32 输出和 LSE，保存当前层、完整 block_table 以及有效序列长度。
 * 
 * 来自原 attn_initialize_kvhead_()，删除了检索堆、相似度表和按 KV head 扩展稀疏表的代码。
 *
 * @param batch_size 要初始化的 Query 行数。
 * @param layer_idx 当前模型层编号。
 * @param block_table 行优先 [batch_size, max_block_num] 逻辑 block 到物理 block 的完整映射。
 * @param max_block_num block_table 每行的宽度；Dense 路径会完整遍历该宽度内的有效 block。
 * @param cache_seqlens 每个 batch 当前有效的 KV cache token 数。
 */
void KVCache::attn_initialize_kvhead_(int batch_size, int layer_idx, int* block_table,
                                      int& max_block_num, int* cache_seqlens) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();
  layer_id_ = layer_idx;
  block_table_ = block_table;
  block_num_per_seq_ = max_block_num;
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    for (int i = 0; i < config_.kv_head_num; i++) {
      for (int j = 0; j < n_gqa_ * config_.head_dim; j++) {
        output_fp32_[batch_idx][i][j] = 0;
      }
      for (int j = 0; j < n_gqa_; j++) attn_lse_[batch_idx][i][j] = 0;
    }
    cache_seqlens_[batch_idx] = cache_seqlens[batch_idx];
  }
  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
}

/**
 * @brief 使用已有 paged KV cache 计算 Decode Attention。
 *
 * 函数直接读取 FP16 Query、初始化 KV-head 累加状态，然后对 block_table 中的全部有效 block执行分块
 * Attention。这个函数来自原 attn()，删除了 Query 量化、retrieval_type 分派、Top-K block 检索及其稀疏参数。
 *
 * @param q_in 已完成 Qwen3 Q Norm 和 RoPE 的 FP16 Query，形状为
 * [batch_size, q_len, q_head_num, head_dim]。
 * @param output FP16 输出，形状与 q_in 相同。
 * @param attn_lse FP32 log-sum-exp 输出，形状为 [batch_size, q_len, q_head_num]。
 * @param layer_idx 要读取 KV cache 的模型层编号，从 0 开始。
 * @param generate_token_idx 保留原 Decode token 编号参数；Dense 路径不再用它控制检索复用。
 * @param q_len 每个 batch 中的 Query token 数；当前 Decode 调用约定为 1。
 * @param batch_size batch 数量，内部会将 q_len 折叠到任务维度。
 * @param max_block_num 每个 batch 的 block_table 表项数。
 * @param block_table 行优先 [batch_size, max_block_num] 逻辑 block 到物理 block 的完整映射。
 * @param cache_seqlens 每个 batch 当前有效的 KV cache token 数。
 * @param backend 用于 block kernel 和结果归并的工作线程池。
 */
void KVCache::attn(const ggml_fp16_t* q_in, ggml_fp16_t* output, float* attn_lse,
                   int layer_idx, int generate_token_idx, int q_len, int batch_size,
                   int max_block_num, int* block_table, int* cache_seqlens,
                   WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();
  layer_id_ = layer_idx;
  batch_size = batch_size * q_len;
  const uint16_t* q_in_data = const_cast<const uint16_t*>(q_in);
  attn_initialize_kvhead_(batch_size, layer_idx, block_table, max_block_num,
                          cache_seqlens);
  

  // 实际工作的函数
  attention_kvhead_(q_in_data, output, attn_lse, batch_size, backend);

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
}

/**
 * @brief 将当前 Decode token 追加到 KV cache，并基于更新后的完整 cache 计算 Attention。
 *
 * 函数先把新 K/V 写入 paged 物理 block，原地增加各序列长度，再调用 attn() 全量读取有效 KV。
 * 这个函数来自原 attn_with_kvcache()，删除了 topk、local、init block 计算和稀疏检索调用。
 *
 * @param q_in 已完成 Qwen3 Q Norm 和 RoPE 的当前 token FP16 Query，形状为
 * [batch_size, 1, q_head_num, head_dim]。
 * @param k_in 已完成 Qwen3 K Norm 和 RoPE 的当前 token FP16 Key，形状为
 * [batch_size, 1, kv_head_num, head_dim]。
 * @param v_in 当前 token 的 FP16 Value，形状与 k_in 相同。
 * @param output FP16 输出，形状与 q_in 相同。
 * @param attn_lse FP32 log-sum-exp 输出，形状为 [batch_size, 1, q_head_num]。
 * @param layer_idx 要更新和计算的模型层编号，从 0 开始。
 * @param generate_token_idx 保留原 Decode token 编号参数；Dense 路径不再用它控制检索复用。
 * @param q_len Query 长度；该 Decode 接口要求值为 1。
 * @param batch_size 相互独立的序列数量。
 * @param max_block_num 每个序列的 block_table 表项数。
 * @param block_table 行优先的逻辑 block 到物理 block 映射。
 * @param cache_seqlens 输入/输出有效 token 数；执行 Attention 前会增加 q_len。
 * @param backend 用于更新 cache 和计算 Attention 的工作线程池。
 */
void KVCache::attn_with_kvcache(
    const ggml_fp16_t* q_in, const ggml_fp16_t* k_in,
    const ggml_fp16_t* v_in, ggml_fp16_t* output, float* attn_lse,
    int layer_idx, int generate_token_idx, int q_len, int batch_size,
    int max_block_num, int* block_table, int* cache_seqlens,
    WorkerPool* backend) {
  assert(q_len == 1);
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();
  layer_id_ = layer_idx;
  update_kvcache_fp16(k_in, v_in, layer_idx, block_table, batch_size,
                      max_block_num, cache_seqlens, q_len, backend);
  for (int i = 0; i < batch_size; i++) cache_seqlens[i] += q_len;
  attn(q_in, output, attn_lse, layer_idx, generate_token_idx, q_len, batch_size,
       max_block_num, block_table, cache_seqlens, backend);

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
}

/**
 * @brief 使用 KV cache 计算单个物理 block 的 FP16 Attention。
 *
 * 函数依次执行 QK、缩放、可选 mask、softmax 和 PV，并输出当前 block 的 FP32 结果与 log-sum-exp。
 * 这个函数来自原 attn_with_kvcache_one_block_()，删除了 anchor 参数、量化布局参数、Q4_0/Q8_0
 * 执行分支和量化临时区。q_type、k_type、v_type 作为未来增加 INT8 kernel dispatch 的类型入口继续保留，
 * 当前实现会显式要求三者均为 GGML_TYPE_F16。
 *
 * @param head_dim 每个 Attention head 的特征维度。
 * @param bsz 共享同一个 KV head 的 GQA Query head 数。
 * @param q_type Query 的 GGML 类型；当前必须为 GGML_TYPE_F16。
 * @param q Query，布局为 [bsz, head_dim]。
 * @param past_kv_len 当前物理 block 的 token 容量。
 * @param past_kv_offset 当前物理 block 在逻辑序列中的偏移；保留原参数位置。
 * @param is_full_attn 是否使用全 1 mask。
 * @param attn_mask is_full_attn 为 false 时使用的位矩阵，逻辑布局为 [bsz, past_kv_len]。
 * @param k_type Key cache 的 GGML 类型；当前必须为 GGML_TYPE_F16。
 * @param k_cache Key cache，布局为 [past_kv_len, head_dim]。
 * @param v_type Value cache 的 GGML 类型；当前必须为 GGML_TYPE_F16。
 * @param v_cache Value cache，布局为 [head_dim, past_kv_len]。
 * @param attn_score 调用方预分配的 FP32 score 缓冲区，布局为 [bsz, past_kv_len]。
 * @param output 当前 block 的 FP32 Attention 输出，布局为 [bsz, head_dim]。
 * @param lse 当前 block 每个 Query head 的 FP32 log-sum-exp 输出。
 * @param draft 调用方预分配的 FP16 路径临时区：开头存放 FP32 PV 结果，随后存放 FP16
 * Attention probability；不再包含任何 Q4_0/Q8_0 或 RoPE 数据。
 * 调用约定：Query 和 Key cache 均已在 Attention 之前完成所需的 Norm 与 RoPE；本函数只负责
 * QK、缩放、mask、softmax 和 PV。
 */
void KVCache::attn_with_kvcache_one_block_(
    int head_dim, int bsz, ggml_type q_type, const void* q, int past_kv_len,
    int past_kv_offset, bool is_full_attn, const uint8_t* attn_mask,
    ggml_type k_type, const void* k_cache, ggml_type v_type,
    const void* v_cache, float* attn_score,
    void* output, float* lse, void* draft) {
  assert(head_dim % 32 == 0);
  assert(q_type == GGML_TYPE_F16);
  assert(k_type == GGML_TYPE_F16);
  assert(v_type == GGML_TYPE_F16);

  char* draft_bytes = reinterpret_cast<char*>(draft);
  float* sum = reinterpret_cast<float*>(draft_bytes);
  ggml_fp16_t* fp16_workspace = reinterpret_cast<ggml_fp16_t*>(
      draft_bytes + sizeof(float) * bsz * head_dim);

  bool ok = llamafile_sgemm(past_kv_len, bsz, head_dim, (ggml_fp16_t*)k_cache,
                            head_dim, (ggml_fp16_t*)q, head_dim, attn_score,
                            past_kv_len, 0, 1, GGML_TASK_TYPE_COMPUTE, k_type,
                            GGML_TYPE_F16, GGML_TYPE_F32, GGML_PREC_DEFAULT);
  if (!ok) {
    printf("llamafile_sgemm failed\n");
  }

  // attn = attn * scale
  float scale_factor = 1.0 / std::sqrt(float(head_dim));
  ggml_vec_scale_f32(bsz * past_kv_len, attn_score, scale_factor);

  // attn = attn & mask
  if (!is_full_attn) {
    for (int i = 0; i < bsz; i++) {
      for (int j = 0; j < past_kv_len; j++) {
        int index = i * past_kv_len + j;
        if (!(attn_mask[j / 8] & (1 << (j % 8)))) {
          attn_score[index] = std::numeric_limits<float>::lowest();
        }
      }
    }
  }

  // attn = softmax(attn)
  for (int i = 0; i < bsz; i++) {
    float sum_exp = 0;
    for (int j = 0; j < past_kv_len; j++) {
      attn_score[i * past_kv_len + j] = std::exp(attn_score[i * past_kv_len + j]);
      sum_exp += attn_score[i * past_kv_len + j];
    }
    for (int j = 0; j < past_kv_len; j++) {
      attn_score[i * past_kv_len + j] /= sum_exp;
    }
    if (lse != nullptr) {
      lse[i] = std::log(sum_exp);
    }
  }

  // output = attn * v
  ggml_fp16_t* attn_score_fp16 = fp16_workspace;
  for (int i = 0; i < bsz * past_kv_len; i++) {
    attn_score_fp16[i] = GGML_FP32_TO_FP16(attn_score[i]);
  }

  bool ok = llamafile_sgemm(head_dim, bsz, past_kv_len,
                            (ggml_fp16_t*)v_cache, past_kv_len,
                            (ggml_fp16_t*)attn_score_fp16, past_kv_len, sum,
                            head_dim, 0, 1, GGML_TASK_TYPE_COMPUTE, v_type,
                            GGML_TYPE_F16, GGML_TYPE_F32, GGML_PREC_DEFAULT);
  if (!ok) {
    printf("llamafile_sgemm failed\n");
  }

  // copy to output
  for (int i = 0; i < bsz; i++) {
    for (int j = 0; j < head_dim; j++) {
      ((float*)output)[i * head_dim + j] = sum[i * head_dim + j];
    }
  }
}

}  // namespace dense
