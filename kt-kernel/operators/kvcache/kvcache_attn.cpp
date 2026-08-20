/**
 * @Description  :
 * @Author       : Jianwei Dong
 * @Date         : 2024-08-26 22:47:06
 * @Version      : 1.0.0
 * @LastEditors  : Jianwei Dong
 * @LastEditTime : 2024-08-26 22:47:06
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/

#include <chrono>
#include <cmath>

#include "ggml-impl.h"
#include "kvcache.h"
#include "llamafile/sgemm.h"

/**
 * @brief 按 KV head 独立的检索结果计算分块 Attention。
 *
 * 每个工作窃取任务处理一个 batch、一个 KV head 和一个已选 cache block。函数为共享该 KV head 的全部
 * GQA query heads 计算精确 Attention，对未填满的尾块施加 mask，并通过 log-sum-exp 缩放合并各 block，
 * 使结果等价于在所有已选 block 逻辑拼接后执行一次 softmax。
 *
 * @param q_in_data FP16 Query 数据，布局为 [batch_size, q_head_num, head_dim]。
 * @param output FP16 Attention 输出，逻辑布局与 q_in_data 相同。
 * @param attn_lse FP32 log-sum-exp 输出，每个 batch、每个 query head 对应一个值。
 * @param batch_size 要处理的 Query 行数；调用方已经将 q_len 折叠到该维度中。
 * @param backend 用于并行执行 block Attention 和线程结果归并的工作线程池。
 */
void KVCache::attention_kvhead_(const uint16_t* q_in_data, ggml_fp16_t* output, float* attn_lse, int batch_size,
                                WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();
  seq_len_ = config_.block_len;

  backend->do_work_stealing_job(
      batch_size * config_.kv_head_num * max_block_num_after_retrieval_,
      [&](int thread_id) {
        thread_cur_head_idx_[thread_id].first = -1;
        thread_cur_head_idx_[thread_id].second = -1;
      },
      [&](int task_id) {
        int batch_id = task_id / (config_.kv_head_num * max_block_num_after_retrieval_);
        int head_id =
            (task_id % (config_.kv_head_num * max_block_num_after_retrieval_)) / max_block_num_after_retrieval_;
        int block_id = task_id % max_block_num_after_retrieval_;
        int thread_id = WorkerPool::thread_local_id;

        // If the block is out of the sequence length, skip it.
        if (cache_seqlens_[batch_id] / config_.block_len < block_id) {
          return;
        }
        int block_idx = block_table_after_retrieval_kvhead_[batch_id][block_id][head_id];
        if (cache_seqlens_[batch_id] / config_.block_len == block_id) {
          int seq_len = cache_seqlens_[batch_id] % config_.block_len;
          if (seq_len == 0) return;

          // Prepare the attention mask for the last block.
          int full_blocks = seq_len / 8;
          int remaining_bits = seq_len % 8;
          // Fill full blocks with 1s
          for (int i = 0; i < full_blocks; ++i) {
            thread_local_attn_mask_[thread_id][i] = 0xFF;
          }
          // Fill the remaining bits in the next block
          if (remaining_bits > 0 && full_blocks < seq_len_ / 8) {
            thread_local_attn_mask_[thread_id][full_blocks] = (1 << remaining_bits) - 1;
          } else {
            thread_local_attn_mask_[thread_id][full_blocks] = 0;
          }

          for (int i = full_blocks + 1; i < seq_len_ / 8; ++i) {
            thread_local_attn_mask_[thread_id][i] = 0;
          }
          if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
            attn_with_kvcache_one_block_(config_.head_dim, config_.q_head_num / config_.kv_head_num, GGML_TYPE_F16,
                                         (void*)&q_in_data[batch_id * config_.kv_head_num * n_gqa_ * config_.head_dim +
                                                           head_id * n_gqa_ * config_.head_dim],
                                         seq_len_, 0, false, thread_local_attn_mask_[thread_id].data(), GGML_TYPE_F16,
                                         0, k_cache_fp16_[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                                         GGML_TYPE_F16, 1, v_cache_fp16_[layer_id_][head_id][block_idx].data(), 0,
                                         nullptr, nullptr, thread_local_attn_score_[thread_id].data(),
                                         thread_local_output_fp32_[thread_id].data(),
                                         thread_local_attn_lse_[thread_id].data(),
                                         thread_local_draft_[thread_id].data(), nullptr, cos_.data(), sin_.data());
          } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
            attn_with_kvcache_one_block_(
                config_.head_dim, config_.q_head_num / config_.kv_head_num, GGML_TYPE_Q8_0,
                q_q8_0_[batch_id][head_id].data(), seq_len_, 0, false, thread_local_attn_mask_[thread_id].data(),
                GGML_TYPE_Q4_0, 0, k_cache_q4[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                GGML_TYPE_Q4_0, 1, v_cache_q4[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                thread_local_attn_score_[thread_id].data(), thread_local_output_q8_0_[thread_id].data(),
                thread_local_attn_lse_[thread_id].data(), thread_local_draft_[thread_id].data(), nullptr, cos_.data(),
                sin_.data());
            dequantize_row_q8_0(thread_local_output_q8_0_[thread_id].data(),
                                thread_local_output_fp32_[thread_id].data(), n_gqa_ * config_.head_dim);
          } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
            attn_with_kvcache_one_block_(
                config_.head_dim, config_.q_head_num / config_.kv_head_num, GGML_TYPE_Q8_0,
                q_q8_0_[batch_id][head_id].data(), seq_len_, 0, false, thread_local_attn_mask_[thread_id].data(),
                GGML_TYPE_Q8_0, 0, k_cache_q8[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                GGML_TYPE_Q8_0, 1, v_cache_q8[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                thread_local_attn_score_[thread_id].data(), thread_local_output_q8_0_[thread_id].data(),
                thread_local_attn_lse_[thread_id].data(), thread_local_draft_[thread_id].data(), nullptr, cos_.data(),
                sin_.data());
            dequantize_row_q8_0(thread_local_output_q8_0_[thread_id].data(),
                                thread_local_output_fp32_[thread_id].data(), n_gqa_ * config_.head_dim);
          }
        } else {
          if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
            attn_with_kvcache_one_block_(
                config_.head_dim, config_.q_head_num / config_.kv_head_num, GGML_TYPE_F16,
                (void*)&q_in_data[batch_id * config_.kv_head_num * n_gqa_ * config_.head_dim +
                                  head_id * n_gqa_ * config_.head_dim],
                seq_len_, 0, true, nullptr, GGML_TYPE_F16, 0, k_cache_fp16_[layer_id_][head_id][block_idx].data(), 0,
                nullptr, nullptr, GGML_TYPE_F16, 1, v_cache_fp16_[layer_id_][head_id][block_idx].data(), 0, nullptr,
                nullptr, thread_local_attn_score_[thread_id].data(), thread_local_output_fp32_[thread_id].data(),
                thread_local_attn_lse_[thread_id].data(), thread_local_draft_[thread_id].data(), nullptr, cos_.data(),
                sin_.data());

          } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
            attn_with_kvcache_one_block_(config_.head_dim, config_.q_head_num / config_.kv_head_num, GGML_TYPE_Q8_0,
                                         q_q8_0_[batch_id][head_id].data(), seq_len_, 0, true, nullptr, GGML_TYPE_Q4_0,
                                         0, k_cache_q4[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                                         GGML_TYPE_Q4_0, 1, v_cache_q4[layer_id_][head_id][block_idx].data(), 0,
                                         nullptr, nullptr, thread_local_attn_score_[thread_id].data(),
                                         thread_local_output_q8_0_[thread_id].data(),
                                         thread_local_attn_lse_[thread_id].data(),
                                         thread_local_draft_[thread_id].data(), nullptr, cos_.data(), sin_.data());
            dequantize_row_q8_0(thread_local_output_q8_0_[thread_id].data(),
                                thread_local_output_fp32_[thread_id].data(), n_gqa_ * config_.head_dim);
          } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
            attn_with_kvcache_one_block_(config_.head_dim, config_.q_head_num / config_.kv_head_num, GGML_TYPE_Q8_0,
                                         q_q8_0_[batch_id][head_id].data(), seq_len_, 0, true, nullptr, GGML_TYPE_Q8_0,
                                         0, k_cache_q8[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                                         GGML_TYPE_Q8_0, 1, v_cache_q8[layer_id_][head_id][block_idx].data(), 0,
                                         nullptr, nullptr, thread_local_attn_score_[thread_id].data(),
                                         thread_local_output_q8_0_[thread_id].data(),
                                         thread_local_attn_lse_[thread_id].data(),
                                         thread_local_draft_[thread_id].data(), nullptr, cos_.data(), sin_.data());
            dequantize_row_q8_0(thread_local_output_q8_0_[thread_id].data(),
                                thread_local_output_fp32_[thread_id].data(), n_gqa_ * config_.head_dim);
          }
        }
        int cur_batch_idx = thread_cur_head_idx_[thread_id].first;
        int cur_head_id = thread_cur_head_idx_[thread_id].second;
        if (batch_id == cur_batch_idx && head_id == cur_head_id) {
          for (int i = 0; i < n_gqa_; i++) {
            float new_attn_lse = thread_local_cur_attn_lse_[thread_id][i] +
                                 std::log(1.0 + std::exp(thread_local_attn_lse_[thread_id][i] -
                                                         thread_local_cur_attn_lse_[thread_id][i]));
            ggml_vec_scale_f32(config_.head_dim, thread_local_cur_output_fp32_[thread_id].data() + i * config_.head_dim,
                               std::exp(thread_local_cur_attn_lse_[thread_id][i] - new_attn_lse));
            ggml_vec_scale_f32(config_.head_dim, thread_local_output_fp32_[thread_id].data() + i * config_.head_dim,
                               std::exp(thread_local_attn_lse_[thread_id][i] - new_attn_lse));
            for (int j = 0; j < config_.head_dim; j++) {
              thread_local_cur_output_fp32_[thread_id][i * config_.head_dim + j] +=
                  thread_local_output_fp32_[thread_id][i * config_.head_dim + j];
            }
            thread_local_cur_attn_lse_[thread_id][i] = new_attn_lse;
          }
        } else {
          if (cur_batch_idx != -1) {
            mutex_[cur_batch_idx][cur_head_id]->lock();
            for (int i = 0; i < n_gqa_; i++) {
              if (std::abs(attn_lse_[cur_batch_idx][cur_head_id][i]) < 1e-6) {
                attn_lse_[cur_batch_idx][cur_head_id][i] = thread_local_cur_attn_lse_[thread_id][i];
                for (int j = 0; j < config_.head_dim; j++) {
                  output_fp32_[cur_batch_idx][cur_head_id][i * config_.head_dim + j] =
                      thread_local_cur_output_fp32_[thread_id][i * config_.head_dim + j];
                }
                continue;
              }
              float new_attn_lse = attn_lse_[cur_batch_idx][cur_head_id][i] +
                                   std::log(1.0 + std::exp(thread_local_cur_attn_lse_[thread_id][i] -
                                                           attn_lse_[cur_batch_idx][cur_head_id][i]));
              ggml_vec_scale_f32(config_.head_dim,
                                 output_fp32_[cur_batch_idx][cur_head_id].data() + i * config_.head_dim,
                                 std::exp(attn_lse_[cur_batch_idx][cur_head_id][i] - new_attn_lse));
              ggml_vec_scale_f32(config_.head_dim,
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
            thread_local_cur_attn_lse_[thread_id][i] = thread_local_attn_lse_[thread_id][i];
            for (int j = 0; j < config_.head_dim; j++) {
              thread_local_cur_output_fp32_[thread_id][i * config_.head_dim + j] =
                  thread_local_output_fp32_[thread_id][i * config_.head_dim + j];
            }
          }
        }
      },
      // Merge the results of the remaining blocks.
      [&](int thread_id) {
        int cur_batch_idx = thread_cur_head_idx_[thread_id].first;
        int cur_head_id = thread_cur_head_idx_[thread_id].second;
        if (cur_head_id != -1) {
          mutex_[cur_batch_idx][cur_head_id]->lock();
          for (int i = 0; i < n_gqa_; i++) {
            float new_attn_lse;
            if (std::abs(attn_lse_[cur_batch_idx][cur_head_id][i]) < 1e-6) {
              attn_lse_[cur_batch_idx][cur_head_id][i] = thread_local_cur_attn_lse_[thread_id][i];
              for (int j = 0; j < config_.head_dim; j++) {
                output_fp32_[cur_batch_idx][cur_head_id][i * config_.head_dim + j] =
                    thread_local_cur_output_fp32_[thread_id][i * config_.head_dim + j];
              }
              continue;
            }
            new_attn_lse = attn_lse_[cur_batch_idx][cur_head_id][i] +
                           std::log(1.0 + std::exp(thread_local_cur_attn_lse_[thread_id][i] -
                                                   attn_lse_[cur_batch_idx][cur_head_id][i]));
            ggml_vec_scale_f32(config_.head_dim, output_fp32_[cur_batch_idx][cur_head_id].data() + i * config_.head_dim,
                               std::exp(attn_lse_[cur_batch_idx][cur_head_id][i] - new_attn_lse));
            ggml_vec_scale_f32(config_.head_dim, thread_local_cur_output_fp32_[thread_id].data() + i * config_.head_dim,
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
  // move the results to output and attn_lse
  uint16_t* output_data = reinterpret_cast<uint16_t*>(output);
  float* attn_lse_data = attn_lse;
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    for (int i = 0; i < config_.kv_head_num; i++) {
      for (int j = 0; j < n_gqa_ * config_.head_dim; j++) {
        output_data[batch_idx * config_.kv_head_num * n_gqa_ * config_.head_dim + i * n_gqa_ * config_.head_dim + j] =
            GGML_FP32_TO_FP16(output_fp32_[batch_idx][i][j]);
      }
      for (int j = 0; j < n_gqa_; j++) {
        attn_lse_data[batch_idx * config_.kv_head_num * n_gqa_ + i * n_gqa_ + j] = attn_lse_[batch_idx][i][j];
      }
    }
  }

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  // printf("layer %d time of computing attention: %f s\n", layer_idx,
  //        diff.count());
}

/**
 * @brief 使用该层所有 KV head 共享的 block 表计算分块 Attention。
 *
 * 各个已选 block 被独立并行处理。每个 block 为某个 KV head 对应的 GQA Query head 组计算精确的 QK、
 * softmax 和 PV，再通过 log-sum-exp 归一化合并 block 结果。传入完整 block 表时，该流程无需构造完整
 * Attention 分数矩阵即可实现 dense paged Attention。
 *
 * @param q_in_data FP16 Query 数据，布局为 [batch_size, q_head_num, head_dim]。
 * @param output FP16 Attention 输出，布局为 [batch_size, q_head_num, head_dim]。
 * @param attn_lse FP32 log-sum-exp 输出，每个 query head 对应一个值。
 * @param batch_size 要处理的 Query 行数；q_len 已经折叠到该维度中。
 * @param backend 负责调度 (batch, KV head, block) 任务及其归并操作的工作线程池。
 */
void KVCache::attention_layer_(const uint16_t* q_in_data, ggml_fp16_t* output, float* attn_lse, int batch_size,
                               WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();
  seq_len_ = config_.block_len;
  backend->do_work_stealing_job(
      batch_size * config_.kv_head_num * max_block_num_after_retrieval_,
      [&](int thread_id) {
        thread_cur_head_idx_[thread_id].first = -1;
        thread_cur_head_idx_[thread_id].second = -1;
      },
      [&](int task_id) {
        int batch_id = task_id / (config_.kv_head_num * max_block_num_after_retrieval_);
        int head_id =
            (task_id % (config_.kv_head_num * max_block_num_after_retrieval_)) / max_block_num_after_retrieval_;
        int block_id = task_id % max_block_num_after_retrieval_;
        int thread_id = WorkerPool::thread_local_id;
        // If the block is out of the sequence length, skip it.
        if (cache_seqlens_[batch_id] / config_.block_len < block_id) {
          return;
        }
        int block_idx = block_table_after_retrieval_[batch_id][block_id];
        if (cache_seqlens_[batch_id] / config_.block_len == block_id) {
          int seq_len = cache_seqlens_[batch_id] % config_.block_len;
          if (seq_len == 0) return;

          // Prepare the attention mask for the last block.
          int full_blocks = seq_len / 8;
          int remaining_bits = seq_len % 8;

          // Fill full blocks with 1s
          for (int i = 0; i < full_blocks; ++i) {
            thread_local_attn_mask_[thread_id][i] = 0xFF;
          }
          // Fill the remaining bits in the next block
          if (remaining_bits > 0 && full_blocks < seq_len_ / 8) {
            thread_local_attn_mask_[thread_id][full_blocks] = (1 << remaining_bits) - 1;
          } else {
            thread_local_attn_mask_[thread_id][full_blocks] = 0;
          }

          for (int i = full_blocks + 1; i < seq_len_ / 8; ++i) {
            thread_local_attn_mask_[thread_id][i] = 0;
          }
          if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
            attn_with_kvcache_one_block_(config_.head_dim, config_.q_head_num / config_.kv_head_num, GGML_TYPE_F16,
                                         (void*)&q_in_data[batch_id * config_.kv_head_num * n_gqa_ * config_.head_dim +
                                                           head_id * n_gqa_ * config_.head_dim],
                                         seq_len_, 0, false, thread_local_attn_mask_[thread_id].data(), GGML_TYPE_F16,
                                         0, k_cache_fp16_[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                                         GGML_TYPE_F16, 1, v_cache_fp16_[layer_id_][head_id][block_idx].data(), 0,
                                         nullptr, nullptr, thread_local_attn_score_[thread_id].data(),
                                         thread_local_output_fp32_[thread_id].data(),
                                         thread_local_attn_lse_[thread_id].data(),
                                         thread_local_draft_[thread_id].data(), nullptr, cos_.data(), sin_.data());
          } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
            attn_with_kvcache_one_block_(
                config_.head_dim, config_.q_head_num / config_.kv_head_num, GGML_TYPE_Q8_0,
                q_q8_0_[batch_id][head_id].data(), seq_len_, 0, false, thread_local_attn_mask_[thread_id].data(),
                GGML_TYPE_Q4_0, 0, k_cache_q4[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                GGML_TYPE_Q4_0, 1, v_cache_q4[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                thread_local_attn_score_[thread_id].data(), thread_local_output_q8_0_[thread_id].data(),
                thread_local_attn_lse_[thread_id].data(), thread_local_draft_[thread_id].data(), nullptr, cos_.data(),
                sin_.data());
            dequantize_row_q8_0(thread_local_output_q8_0_[thread_id].data(),
                                thread_local_output_fp32_[thread_id].data(), n_gqa_ * config_.head_dim);
          } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
            attn_with_kvcache_one_block_(
                config_.head_dim, config_.q_head_num / config_.kv_head_num, GGML_TYPE_Q8_0,
                q_q8_0_[batch_id][head_id].data(), seq_len_, 0, false, thread_local_attn_mask_[thread_id].data(),
                GGML_TYPE_Q8_0, 0, k_cache_q8[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                GGML_TYPE_Q8_0, 1, v_cache_q8[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                thread_local_attn_score_[thread_id].data(), thread_local_output_q8_0_[thread_id].data(),
                thread_local_attn_lse_[thread_id].data(), thread_local_draft_[thread_id].data(), nullptr, cos_.data(),
                sin_.data());
            dequantize_row_q8_0(thread_local_output_q8_0_[thread_id].data(),
                                thread_local_output_fp32_[thread_id].data(), n_gqa_ * config_.head_dim);
          }
        } else {
          if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
            attn_with_kvcache_one_block_(
                config_.head_dim, config_.q_head_num / config_.kv_head_num, GGML_TYPE_F16,
                (void*)&q_in_data[batch_id * config_.kv_head_num * n_gqa_ * config_.head_dim +
                                  head_id * n_gqa_ * config_.head_dim],
                seq_len_, 0, true, nullptr, GGML_TYPE_F16, 0, k_cache_fp16_[layer_id_][head_id][block_idx].data(), 0,
                nullptr, nullptr, GGML_TYPE_F16, 1, v_cache_fp16_[layer_id_][head_id][block_idx].data(), 0, nullptr,
                nullptr, thread_local_attn_score_[thread_id].data(), thread_local_output_fp32_[thread_id].data(),
                thread_local_attn_lse_[thread_id].data(), thread_local_draft_[thread_id].data(), nullptr, cos_.data(),
                sin_.data());

          } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
            attn_with_kvcache_one_block_(config_.head_dim, config_.q_head_num / config_.kv_head_num, GGML_TYPE_Q8_0,
                                         q_q8_0_[batch_id][head_id].data(), seq_len_, 0, true, nullptr, GGML_TYPE_Q4_0,
                                         0, k_cache_q4[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                                         GGML_TYPE_Q4_0, 1, v_cache_q4[layer_id_][head_id][block_idx].data(), 0,
                                         nullptr, nullptr, thread_local_attn_score_[thread_id].data(),
                                         thread_local_output_q8_0_[thread_id].data(),
                                         thread_local_attn_lse_[thread_id].data(),
                                         thread_local_draft_[thread_id].data(), nullptr, cos_.data(), sin_.data());
            dequantize_row_q8_0(thread_local_output_q8_0_[thread_id].data(),
                                thread_local_output_fp32_[thread_id].data(), n_gqa_ * config_.head_dim);
          } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
            attn_with_kvcache_one_block_(config_.head_dim, config_.q_head_num / config_.kv_head_num, GGML_TYPE_Q8_0,
                                         q_q8_0_[batch_id][head_id].data(), seq_len_, 0, true, nullptr, GGML_TYPE_Q8_0,
                                         0, k_cache_q8[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                                         GGML_TYPE_Q8_0, 1, v_cache_q8[layer_id_][head_id][block_idx].data(), 0,
                                         nullptr, nullptr, thread_local_attn_score_[thread_id].data(),
                                         thread_local_output_q8_0_[thread_id].data(),
                                         thread_local_attn_lse_[thread_id].data(),
                                         thread_local_draft_[thread_id].data(), nullptr, cos_.data(), sin_.data());
            dequantize_row_q8_0(thread_local_output_q8_0_[thread_id].data(),
                                thread_local_output_fp32_[thread_id].data(), n_gqa_ * config_.head_dim);
          }
        }
        int cur_batch_idx = thread_cur_head_idx_[thread_id].first;
        int cur_head_id = thread_cur_head_idx_[thread_id].second;
        if (batch_id == cur_batch_idx && head_id == cur_head_id) {
          for (int i = 0; i < n_gqa_; i++) {
            float new_attn_lse = thread_local_cur_attn_lse_[thread_id][i] +
                                 std::log(1.0 + std::exp(thread_local_attn_lse_[thread_id][i] -
                                                         thread_local_cur_attn_lse_[thread_id][i]));
            ggml_vec_scale_f32(config_.head_dim, thread_local_cur_output_fp32_[thread_id].data() + i * config_.head_dim,
                               std::exp(thread_local_cur_attn_lse_[thread_id][i] - new_attn_lse));
            ggml_vec_scale_f32(config_.head_dim, thread_local_output_fp32_[thread_id].data() + i * config_.head_dim,
                               std::exp(thread_local_attn_lse_[thread_id][i] - new_attn_lse));
            for (int j = 0; j < config_.head_dim; j++) {
              thread_local_cur_output_fp32_[thread_id][i * config_.head_dim + j] +=
                  thread_local_output_fp32_[thread_id][i * config_.head_dim + j];
            }
            thread_local_cur_attn_lse_[thread_id][i] = new_attn_lse;
          }
        } else {
          if (cur_batch_idx != -1) {
            mutex_[cur_batch_idx][cur_head_id]->lock();
            for (int i = 0; i < n_gqa_; i++) {
              if (std::abs(attn_lse_[cur_batch_idx][cur_head_id][i]) < 1e-6) {
                attn_lse_[cur_batch_idx][cur_head_id][i] = thread_local_cur_attn_lse_[thread_id][i];
                for (int j = 0; j < config_.head_dim; j++) {
                  output_fp32_[cur_batch_idx][cur_head_id][i * config_.head_dim + j] =
                      thread_local_cur_output_fp32_[thread_id][i * config_.head_dim + j];
                }
                continue;
              }
              float new_attn_lse = attn_lse_[cur_batch_idx][cur_head_id][i] +
                                   std::log(1.0 + std::exp(thread_local_cur_attn_lse_[thread_id][i] -
                                                           attn_lse_[cur_batch_idx][cur_head_id][i]));
              ggml_vec_scale_f32(config_.head_dim,
                                 output_fp32_[cur_batch_idx][cur_head_id].data() + i * config_.head_dim,
                                 std::exp(attn_lse_[cur_batch_idx][cur_head_id][i] - new_attn_lse));
              ggml_vec_scale_f32(config_.head_dim,
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
            thread_local_cur_attn_lse_[thread_id][i] = thread_local_attn_lse_[thread_id][i];
            for (int j = 0; j < config_.head_dim; j++) {
              thread_local_cur_output_fp32_[thread_id][i * config_.head_dim + j] =
                  thread_local_output_fp32_[thread_id][i * config_.head_dim + j];
            }
          }
        }
      },
      // Merge the results of the remaining blocks.
      [&](int thread_id) {
        int cur_batch_idx = thread_cur_head_idx_[thread_id].first;
        int cur_head_id = thread_cur_head_idx_[thread_id].second;
        if (cur_head_id != -1) {
          mutex_[cur_batch_idx][cur_head_id]->lock();
          for (int i = 0; i < n_gqa_; i++) {
            float new_attn_lse;
            if (std::abs(attn_lse_[cur_batch_idx][cur_head_id][i]) < 1e-6) {
              attn_lse_[cur_batch_idx][cur_head_id][i] = thread_local_cur_attn_lse_[thread_id][i];
              for (int j = 0; j < config_.head_dim; j++) {
                output_fp32_[cur_batch_idx][cur_head_id][i * config_.head_dim + j] =
                    thread_local_cur_output_fp32_[thread_id][i * config_.head_dim + j];
              }
              continue;
            }
            new_attn_lse = attn_lse_[cur_batch_idx][cur_head_id][i] +
                           std::log(1.0 + std::exp(thread_local_cur_attn_lse_[thread_id][i] -
                                                   attn_lse_[cur_batch_idx][cur_head_id][i]));
            ggml_vec_scale_f32(config_.head_dim, output_fp32_[cur_batch_idx][cur_head_id].data() + i * config_.head_dim,
                               std::exp(attn_lse_[cur_batch_idx][cur_head_id][i] - new_attn_lse));
            ggml_vec_scale_f32(config_.head_dim, thread_local_cur_output_fp32_[thread_id].data() + i * config_.head_dim,
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

  // move the results to output and attn_lse
  uint16_t* output_data = reinterpret_cast<uint16_t*>(output);
  float* attn_lse_data = attn_lse;
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    for (int i = 0; i < config_.kv_head_num; i++) {
      for (int j = 0; j < n_gqa_ * config_.head_dim; j++) {
        output_data[batch_idx * config_.kv_head_num * n_gqa_ * config_.head_dim + i * n_gqa_ * config_.head_dim + j] =
            GGML_FP32_TO_FP16(output_fp32_[batch_idx][i][j]);
      }
      for (int j = 0; j < n_gqa_; j++) {
        attn_lse_data[batch_idx * config_.kv_head_num * n_gqa_ + i * n_gqa_ + j] = attn_lse_[batch_idx][i][j];
      }
    }
  }
  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  //     printf("layer %d time of computing attention: %f s\n", layer_id_,
  //     diff.count());
}

/**
 * @brief 为一个模型层执行支持稀疏检索的 paged GQA Attention。
 *
 * 该总控入口准备或量化 Q，导入调用方提供的逻辑 block 到物理 block 映射，按需检索稀疏 block 子集，
 * 最后调度精确的 block Attention。pick_block_num 为 -1 时跳过检索并计算全部有效 block，即 CPU dense
 * Attention。
 *
 * @param q_in FP16 Query，形状为 [batch_size, q_len, q_head_num, head_dim]。
 * @param output FP16 输出，形状与 q_in 相同。
 * @param attn_lse FP32 log-sum-exp 输出，形状为 [batch_size, q_len, q_head_num]。
 * @param layer_idx 要读取 KV cache 的模型层编号，从 0 开始。
 * @param generate_token_idx Decode token 编号，用于判断能否复用历史检索结果。
 * @param q_len 每个 batch 中的 Query token 数。
 * @param batch_size batch 数量，内部尚未将 q_len 折叠到任务维度。
 * @param max_block_num 每个 batch 可用的 block 表项数。
 * @param block_table 可选的行优先 [batch_size, max_block_num] 逻辑 block 到物理 block 映射。
 * @param cache_seqlens 每个 batch 当前有效的 KV cache token 数。
 * @param pick_block_num 要检索的相似中间 block 数；-1 表示使用全部 block。
 * @param init_block_num 始终保留的开头 block 数。
 * @param local_block_num 始终保留的最近完整 block 数。
 * @param backend 用于检索、block kernel 和结果归并的工作线程池。
 */
void KVCache::attn(const ggml_fp16_t* q_in, ggml_fp16_t* output, float* attn_lse, int layer_idx, int generate_token_idx,
                   int q_len, int batch_size, int max_block_num, int* block_table, int* cache_seqlens,
                   int pick_block_num, int init_block_num, int local_block_num, WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();
  layer_id_ = layer_idx;
  batch_size = batch_size * q_len;

  const uint16_t* q_in_data = const_cast<const uint16_t*>(q_in);

  quantize_q_(q_in_data, batch_size);
  if (config_.retrieval_type == RetrievalType::LAYER) {
    attn_initialize_layer_(batch_size, layer_idx, block_table, max_block_num, cache_seqlens);
    // 生成或复用本次 Attention 要遍历的物理 block 表，供紧随其后的 attention_layer_() 使用。
    retrieval_kvcache_layer_(q_in_data, init_block_num, local_block_num, pick_block_num, q_len, generate_token_idx,
                             batch_size, layer_idx, cache_seqlens, max_block_num, backend);
    attention_layer_(q_in_data, output, attn_lse, batch_size, backend);
  } else if (config_.retrieval_type == RetrievalType::KVHEAD) {
    // 初始化每个 KV head 的完整物理 block 映射、序列长度和相似度缓冲区，为按 KV head 独立检索做准备。
    attn_initialize_kvhead_(batch_size, layer_idx, block_table, max_block_num, cache_seqlens);

    // 为每个 KV head 独立选择或复用相关的 Top-K block，生成后续 Attention 实际遍历的稀疏 block 表。

    // 这段函数不用看，产出是：block_table_after_retrieval_kvhead_
    retrieval_kvcache_kvhead_(q_in_data, init_block_num, local_block_num, pick_block_num, q_len, generate_token_idx,
                              batch_size, layer_idx, cache_seqlens, max_block_num, backend);
    

    // 在每个 KV head 选出的 block 上执行 QK、Softmax 和 PV，并将各 block 的结果合并为最终 Attention 输出。                          
    attention_kvhead_(q_in_data, output, attn_lse, batch_size, backend);
  }

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  // printf("layer %d time of computing attention: %f s\n", layer_idx,
  //        diff.count());
}

/**
 * @brief 将当前 decode token 追加到 KV cache，并基于更新后的 cache 计算 Attention。
 *
 * 这是供 Python 封装调用的 decode 入口。函数先把新 K/V 写入 paged 物理 block，增加各序列长度，计算
 * 始终保留的开头 block 数，再调用 attn()。当前实现要求 q_len == 1。
 *
 * @param q_in 当前 token 的 FP16 Query，形状为 [batch_size, 1, q_head_num, head_dim]。
 * @param k_in 当前 token 的 FP16 Key，形状为 [batch_size, 1, kv_head_num, head_dim]。
 * @param v_in 当前 token 的 FP16 Value，形状为 [batch_size, 1, kv_head_num, head_dim]。
 * @param output FP16 输出，形状为 [batch_size, 1, q_head_num, head_dim]。
 * @param attn_lse FP32 log-sum-exp 输出，形状为 [batch_size, 1, q_head_num]。
 * @param layer_idx 要更新和计算的模型层编号，从 0 开始。
 * @param generate_token_idx 用于复用检索结果的 decode token 编号。
 * @param q_len Query 长度；该 decode 接口要求值为 1。
 * @param batch_size 相互独立的序列数量。
 * @param max_block_num 每个序列的 block 表项数。
 * @param block_table 行优先的逻辑 block 到物理 block 映射。
 * @param cache_seqlens 输入/输出有效 token 数；执行 Attention 前会增加 q_len。
 * @param topk 要检索的相似历史 block 数；-1 表示 dense Attention。
 * @param local 始终保留的最近完整 block 数。
 * @param backend 用于更新 cache 和计算 Attention 的工作线程池。
 */
void KVCache::attn_with_kvcache(const ggml_fp16_t* q_in, const ggml_fp16_t* k_in, const ggml_fp16_t* v_in,
                                ggml_fp16_t* output, float* attn_lse, int layer_idx, int generate_token_idx, int q_len,
                                int batch_size, int max_block_num, int* block_table, int* cache_seqlens, int topk,
                                int local, WorkerPool* backend) {
  //    printf("attn_with_kvcache start\n");
  assert(q_len == 1);
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();

  layer_id_ = layer_idx;

  update_kvcache_fp16(k_in, v_in, layer_idx, block_table, batch_size, max_block_num, cache_seqlens, q_len, backend);
  //    printf("update finished.\n");

  // cache_seqlens memory is modified.
  for (int i = 0; i < batch_size; i++) {
    cache_seqlens[i] += q_len;
  }
  int init_block_num = 1;
  if (config_.block_len <= 32) {
    init_block_num = 64 / config_.block_len;
  }

  attn(q_in, output, attn_lse, layer_idx, generate_token_idx, q_len, batch_size, max_block_num, block_table,
       cache_seqlens, topk, init_block_num, local, backend);

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  //     printf("layer %d time of computing attention with kvcache: %f s\n",
  //     layer_idx, diff.count());
}

/**
 * @brief 将 Query 转换为当前 KV-cache kernel 所需的表示形式。
 *
 * FP16 cache 模式把 Q 转换到每个 batch 的 FP32 临时缓冲区；量化 cache 模式则把每组 GQA Query heads
 * 量化为 Q8_0，以便与 Q4_0 或 Q8_0 Key 相乘。
 *
 * @param q_in_data FP16 Query 位数据，布局为 [batch_size, q_head_num, head_dim]。
 * @param batch_size 要转换的 Query 行数。
 */
void KVCache::quantize_q_(const uint16_t* q_in_data, int batch_size) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
      // quantize q
      for (int i = 0; i < config_.kv_head_num; i++) {
        for (int j = 0; j < n_gqa_ * config_.head_dim; j++) {
          q_fp32_[batch_idx][i][j] =
              GGML_FP16_TO_FP32(q_in_data[batch_idx * config_.kv_head_num * n_gqa_ * config_.head_dim +
                                          i * n_gqa_ * config_.head_dim + j]);
        }
      }
    } else {
      // quantize q
      for (int i = 0; i < config_.kv_head_num; i++) {
        for (int j = 0; j < n_gqa_ * config_.head_dim; j++) {
          q_fp32[j] = GGML_FP16_TO_FP32(q_in_data[batch_idx * config_.kv_head_num * n_gqa_ * config_.head_dim +
                                                  i * n_gqa_ * config_.head_dim + j]);
        }
        quantize_row_q8_0(q_fp32.data(), q_q8_0_[batch_idx][i].data(), n_gqa_ * config_.head_dim);
      }
    }
  }
  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  // printf("time of quantizing q: %f s\n",
  //        std::chrono::duration<double>(end - start).count());
}
/**
 * @brief 重置 Attention 累加状态，并导入该层所有 KV head 共享的逻辑 block 表。
 *
 * 函数清空输出和 LSE 临时缓冲区、清空检索堆、复制序列长度，并暂存调用方 block 映射。block_table
 * 为空时，函数根据内部已存 block 构造单位映射，并同步更新 max_block_num。
 *
 * @param batch_size 要初始化的逻辑 Query 行数。
 * @param layer_idx 当前模型层；block_table 为空时用于读取该层内部 block 数。
 * @param block_table 可选的行优先 [batch_size, max_block_num] 逻辑到物理 block 映射。
 * @param max_block_num 输入/输出 block 表宽度；构造单位映射时改为内部已存 block 数。
 * @param cache_seqlens 可选的每行有效 token 数；提供 block_table 时必须有效。
 */
void KVCache::attn_initialize_layer_(int batch_size, int layer_idx, int* block_table, int& max_block_num,
                                     int* cache_seqlens) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    // initialize output_fp32_ and attn_lse_
    for (int i = 0; i < config_.kv_head_num; i++) {
      for (int j = 0; j < n_gqa_ * config_.head_dim; j++) {
        output_fp32_[batch_idx][i][j] = 0;
      }
      for (int j = 0; j < n_gqa_; j++) {
        attn_lse_[batch_idx][i][j] = 0;
      }
    }
    // clear top_similar_block_

    while (!top_similar_block_[batch_idx].empty()) top_similar_block_[batch_idx].pop();
  }

  // get block_table_before_retrieval_ and cache_seqlens_
  if (block_table == nullptr) {
    max_block_num = past_block_num_[layer_idx];
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
      if (cache_total_len_ != 0)
        cache_seqlens_[batch_idx] = cache_total_len_;
      else
        cache_seqlens_[batch_idx] = max_block_num * config_.block_len;
      for (int i = 0; i < max_block_num; i++) {
        block_table_before_retrieval_[batch_idx][i] = i;
        block_similar_[batch_idx][i] = 0;
      }
    }
  } else {
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
      cache_seqlens_[batch_idx] = cache_seqlens[batch_idx];
      for (int i = 0; i < max_block_num; i++) {
        block_table_before_retrieval_[batch_idx][i] = block_table[batch_idx * max_block_num + i];
        block_similar_[batch_idx][i] = 0;
      }
    }
  }
  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  // printf("layer %d time of initializing attention: %f s\n", layer_idx,
  //        std::chrono::duration<double>(end - start).count());
}

/**
 * @brief 检索阶段第一步：使用 Query 和 Anchor 为每个候选 KV block 计算近似相关度。
 *
 * 此函数只负责“打分”，不选择 block，也不读取完整 K/V 做 Attention。输入的
 * block_table_before_retrieval_[batch][logical_block] 保存逻辑 block 到物理 block 的映射；函数根据该映射
 * 找到 anchor_[layer][physical_block][anchor][query_head][dim]，并把结果写入
 * block_similar_[batch][logical_block]。
 *
 * prefix 区间 [0, init_block_num) 和最近的 local_block_num 个完整 block 会被无条件保留，因此不参与打分；
 * 未填满的尾 block 也不参与打分。对于其余中间 block，先对 q_len 个 Query token 求平均，再近似计算：
 *
 *   score(block) = sum_{query_head, dim}
 *                    max_{anchor_id}(mean_q[query_head, dim] * anchor[block, anchor_id, query_head, dim])
 *
 * batch_size == 1、anchor_num == 1 且候选物理 block 连续时，将所有 Anchor 视为矩阵并调用一次
 * llamafile_sgemm()；否则通过 WorkerPool 按 (batch, logical_block) 并行逐个打分。函数结束后完整 block 表
 * 保持不变，后续 select_block_layer_() 才会根据 block_similar_ 构造稀疏 block 表。
 *
 * @param q_in_data FP16 Query，布局为 [batch_size, q_len, q_head_num, head_dim]。
 * @param batch_size 要计算检索分数的序列数量。
 * @param layer_idx Anchor 所属的模型层编号。
 * @param q_len 每个序列包含的 Query token 数；大于 1 时先沿该维度求平均。
 * @param max_block_num block_table_before_retrieval_ 每个 batch 的表宽，也是并行遍历上限。
 * @param cache_seqlens 调用方传入的序列长度；当前实现保留此参数以统一接口，候选边界实际读取
 *                      cache_seqlens_ 内部副本。
 * @param init_block_num 无条件保留、不参与打分的最早完整 block 数。
 * @param local_block_num 无条件保留、不参与打分的最近完整 block 数。
 * @param pick_block_num 中间候选区最终希望保留的 Top-K 数；本函数不直接使用它做选择。
 * @param backend 工作线程池，用于并行执行 llamafile SGEMM 或逐 block 打分任务。
 */
void KVCache::calculate_block_similarity_layer_(const uint16_t* q_in_data, int batch_size, int layer_idx, int q_len,
                                                int max_block_num, int* cache_seqlens, int init_block_num,
                                                int local_block_num, int pick_block_num, WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();

  if (batch_size == 1 && config_.anchor_num == 1) {  // TODO: improve batch_size > 1
    for (int batch_id = 0; batch_id < batch_size; batch_id++) {
      if (q_len == 1) {
        for (int j = 0; j < config_.head_dim * config_.q_head_num; j++) {
          avg_q[batch_id][j] =
              GGML_FP16_TO_FP32(q_in_data[batch_id * q_len * config_.q_head_num * config_.head_dim + j]);
          avg_q_fp16[batch_id][j] = q_in_data[batch_id * q_len * config_.q_head_num * config_.head_dim + j];
        }
      } else {
        for (int j = 0; j < config_.head_dim * config_.q_head_num; j++) {
          avg_q[batch_id][j] = 0;
        }
        for (int i = 0; i < q_len; i++) {
          for (int j = 0; j < config_.head_dim; j++) {
            avg_q[batch_id][j] += GGML_FP16_TO_FP32(q_in_data[batch_id * q_len * config_.q_head_num * config_.head_dim +
                                                              i * config_.q_head_num * config_.head_dim + j]);
          }
        }
        for (int j = 0; j < config_.head_dim * config_.q_head_num; j++) {
          avg_q[batch_id][j] /= q_len;
          avg_q_fp16[batch_id][j] = GGML_FP32_TO_FP16(avg_q[batch_id][j]);
        }
      }
      int seq_len = cache_seqlens_[batch_id];
      int block_num = (seq_len / config_.block_len) - local_block_num - init_block_num;
      if (block_num <= 0) {
        continue;
      }
      bool is_seq = true;
      for (int i = init_block_num + 1; i < (seq_len / config_.block_len) - local_block_num; i++) {
        if (block_table_before_retrieval_[batch_id][i] != block_table_before_retrieval_[batch_id][i - 1] + 1) {
          is_seq = false;
          break;
        }
      }
      if (is_seq) {
        int nth = backend->get_thread_num();
        backend->do_work_stealing_job(
            nth, nullptr,
            [&](int task_id) {
              int ith = task_id;
              bool ok = llamafile_sgemm(
                  block_num, 1, config_.q_head_num * config_.head_dim,
                  anchor_.data() +
                      (layer_idx * config_.max_block_num + block_table_before_retrieval_[batch_id][init_block_num]) *
                          config_.anchor_num * config_.q_head_num * config_.head_dim,
                  config_.q_head_num * config_.head_dim, avg_q_fp16[batch_id].data(),
                  config_.q_head_num * config_.head_dim, block_similar_[batch_id].data() + init_block_num, block_num,
                  ith, nth, GGML_TASK_TYPE_COMPUTE, GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_PREC_DEFAULT);
              if (!ok) {
                printf("llamafile_sgemm failed\n");
              }
            },
            nullptr);
      } else {
        backend->do_work_stealing_job(
            block_num, nullptr,
            [&](int task_id) {
              int block_id = task_id + init_block_num;
              int block_idx = block_table_before_retrieval_[batch_id][block_id];
              bool ok = llamafile_sgemm(
                  1, 1, config_.q_head_num * config_.head_dim,
                  anchor_.data() +
                      (layer_idx * config_.max_block_num + block_table_before_retrieval_[batch_id][block_idx]) *
                          config_.anchor_num * config_.q_head_num * config_.head_dim,
                  config_.q_head_num * config_.head_dim, avg_q_fp16[batch_id].data(),
                  config_.q_head_num * config_.head_dim, block_similar_[batch_id].data() + block_id, 1, 0, 1,
                  GGML_TASK_TYPE_COMPUTE, GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_PREC_DEFAULT);
              if (!ok) {
                printf("llamafile_sgemm failed\n");
              }
            },
            nullptr);
      }
    }
  } else {
    backend->do_work_stealing_job(
        batch_size * max_block_num, nullptr,
        [&](int task_id) {
          int batch_id = task_id / max_block_num;
          int block_id = task_id % max_block_num;
          int seq_len = cache_seqlens_[batch_id];

          if (block_id < init_block_num || block_id >= (seq_len / config_.block_len) - local_block_num) {
            return;
          }

          int block_idx = block_table_before_retrieval_[batch_id][block_id];
          float sim = 0;

          for (int head_id = 0; head_id < config_.q_head_num; head_id++) {
            for (int i = 0; i < config_.head_dim; i++) {
              float q_i = 0, qa_i = std::numeric_limits<float>::lowest();
              for (int q_id = 0; q_id < q_len; q_id++) {
                q_i += GGML_FP16_TO_FP32(
                    q_in_data[batch_id * q_len * config_.q_head_num * config_.head_dim +
                              q_id * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + i]);
              }
              q_i /= q_len;
              for (int anchor_id = 0; anchor_id < config_.anchor_num; anchor_id++) {
                qa_i = std::max(
                    qa_i,
                    GGML_FP16_TO_FP32(
                        anchor_[(long long)layer_idx * config_.max_block_num * config_.anchor_num * config_.q_head_num *
                                    config_.head_dim +
                                block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                                anchor_id * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + i]) *
                        q_i);
              }
              sim += qa_i;
            }
          }
          block_similar_[batch_id][block_id] = sim;
        },
        nullptr);
  }
  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  // printf("layer %d time of calculating similarity: %f s\n", layer_idx,
  //        diff.count());
}

/**
 * @brief 检索阶段第二步：根据 block_similar_ 选出 Top-K，并构造供 Attention 使用的稀疏物理 block 表。
 *
 * 此函数读取 calculate_block_similarity_layer_() 写入的分数。对每个 batch，它使用容量为
 * pick_block_num 的最小堆扫描中间候选区：新候选入堆后若容量超限，就弹出当前最低分，因此最终留下分数
 * 最高的 K 个物理 block。随后按以下顺序写入 block_table_after_retrieval_[batch][selected_slot]：
 *
 *   [无条件保留的 prefix] + [Top-K 中间 block] + [最近的 local block] + [未填满的尾 block]
 *
 * before/after 表中保存的都是 physical_block 编号；选择过程只复制整数索引，不移动或复制实际 K/V 数据。
 * 如果完整 block 数没有超过保留预算，则直接使用原始完整 block 表，不产生稀疏。选择完成后，函数把
 * cache_seqlens_ 改写为“选中完整 block 的总长度 + 尾 block 的有效 token 数”，使 attention_layer_() 可以
 * 把 after 表当作一个逻辑紧凑的序列遍历，并把结果保存到 selected_blocks_history_ 供后续层或 token 复用。
 *
 * @param batch_size 要分别构造稀疏 block 表的序列数量。
 * @param layer_idx 当前模型层编号，用于定位该检索周期对应的 selected_blocks_history_。
 * @param max_block_num block_table_before_retrieval_ 的原始表宽。
 * @param init_block_num 无条件放在输出表开头的最早完整 block 数。
 * @param local_block_num 无条件放在 Top-K 后面的最近完整 block 数。
 * @param pick_block_num 从中间候选区最多选出的高分 block 数。
 */
void KVCache::select_block_layer_(int batch_size, int layer_idx, int max_block_num, int init_block_num,
                                  int local_block_num, int pick_block_num) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();

  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    if (cache_seqlens_[batch_idx] / config_.block_len <= init_block_num + pick_block_num + local_block_num) {
      block_table_after_retrieval_[batch_idx].swap(block_table_before_retrieval_[batch_idx]);
      selected_blocks_num_history_[(layer_idx - config_.layer_offset) / config_.layer_step] = 0;
      continue;
    }

    // 扫描中间候选区，用固定容量最小堆保留分数最高的 pick_block_num 个 block。
    for (int block_id = init_block_num; block_id < (cache_seqlens_[batch_idx] / config_.block_len) - local_block_num;
         block_id++) {
      top_similar_block_[batch_idx].push(
          std::make_pair(block_similar_[batch_idx][block_id], block_table_before_retrieval_[batch_idx][block_id]));
      if (top_similar_block_[batch_idx].size() > pick_block_num) {
        top_similar_block_[batch_idx].pop();
      }
    }

    // after 表的 selected_slot 从 0 开始，依次写入 prefix、Top-K、local 和可选尾 block。
    int i = 0;
    for (; i < init_block_num; i++) {
      block_table_after_retrieval_[batch_idx][i] = block_table_before_retrieval_[batch_idx][i];
    }
    while (!top_similar_block_[batch_idx].empty()) {
      block_table_after_retrieval_[batch_idx][i] = top_similar_block_[batch_idx].top().second;
      top_similar_block_[batch_idx].pop();
      i++;
    }
    for (; i < init_block_num + pick_block_num + local_block_num; i++) {
      block_table_after_retrieval_[batch_idx][i] =
          block_table_before_retrieval_[batch_idx][(cache_seqlens_[batch_idx] / config_.block_len) - local_block_num +
                                                   i - init_block_num - pick_block_num];
    }
    if (cache_seqlens_[batch_idx] % config_.block_len != 0) {
      block_table_after_retrieval_[batch_idx][i] =
          block_table_before_retrieval_[batch_idx][(cache_seqlens_[batch_idx] / config_.block_len)];
      cache_seqlens_[batch_idx] = (cache_seqlens_[batch_idx] % config_.block_len) + i * config_.block_len;
      i++;
    } else {
      cache_seqlens_[batch_idx] = (cache_seqlens_[batch_idx] % config_.block_len) + i * config_.block_len;
    }
    for (int j = 0; j < i; j++) {
      selected_blocks_history_[(layer_idx - config_.layer_offset) / config_.layer_step][batch_idx][j] =
          block_table_after_retrieval_[batch_idx][j];
    }
    selected_blocks_num_history_[(layer_idx - config_.layer_offset) / config_.layer_step] = i;
  }

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  // printf("layer %d time of selecting blocks: %f s\n", layer_idx,
  //        diff.count());
}

/**
 * @brief 稀疏检索总控：决定复用历史、重新执行 Top-K，还是关闭稀疏并使用全部 block。
 *
 * attn() 完成 Query 准备和 before 表初始化后调用此函数。函数最终必须设置两个结果：
 * max_block_num_after_retrieval_ 表示 Attention 要遍历多少个 selected slot，
 * block_table_after_retrieval_ 保存每个 selected slot 对应的 physical_block 编号。
 *
 * 函数有三个互斥分支：
 *
 * 1. pick_block_num != -1，但当前 token 或 layer 不在刷新周期：从 selected_blocks_history_ 复用上次结果；
 *    如果 Decode 刚进入一个新的尾 block，则把这个尾 block 追加到历史和 after 表中。若尚无历史结果，
 *    本次退化为完整 block 表以保证可计算。
 * 2. pick_block_num != -1，且 generate_token_idx % token_step == 0、
 *    layer_idx % layer_step == layer_offset：先调用 calculate_block_similarity_layer_() 给候选打分，再调用
 *    select_block_layer_() 生成新的 Top-K 稀疏表并保存历史。
 * 3. pick_block_num == -1：关闭稀疏检索，after 表直接接管完整 before 表，后续执行 dense Attention。
 *
 * 此函数本身不执行 QK/Softmax/PV；它只决定 attention_layer_() 随后能够看到哪些物理 KV block。
 *
 * @param q_in_data 刷新 Top-K 时用于计算 Anchor 相似度的 FP16 Query。
 * @param init_block_num 无条件保留的最早完整 block 数。
 * @param local_block_num 无条件保留的最近完整 block 数。
 * @param pick_block_num 中间候选区的 Top-K 预算；-1 表示不稀疏、使用完整 block 表。
 * @param q_len 每个序列参与相似度计算的 Query token 数。
 * @param generate_token_idx 当前 Decode token 编号，用于判断是否满足 token_step 刷新周期。
 * @param batch_size 要处理的序列数量。
 * @param layer_idx 当前模型层编号，用于判断 layer_step/layer_offset 周期并索引检索历史。
 * @param cache_seqlens 调用方的有效 token 数；复用历史时用于识别是否刚创建新的尾 block。
 * @param max_block_num 输入时是完整 before 表宽度；函数通过成员 max_block_num_after_retrieval_ 输出选择后的宽度。
 * @param backend 刷新 Top-K 时供相似度计算使用的工作线程池；历史复用和 dense 分支不调用它计算分数。
 */
void KVCache::retrieval_kvcache_layer_(const uint16_t* q_in_data, int init_block_num, int local_block_num,
                                       int pick_block_num, int q_len, int generate_token_idx, int batch_size,
                                       int layer_idx, int* cache_seqlens, int& max_block_num, WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();
  max_block_num_after_retrieval_ = 0;
  // 非刷新步：优先复用历史选择，避免每个 Decode token、每个 Layer 都重新计算 Anchor 分数。
  if (pick_block_num != -1 &&
      (generate_token_idx % config_.token_step != 0 || (layer_idx % config_.layer_step != config_.layer_offset))) {
    if (selected_blocks_num_history_[(layer_idx - config_.layer_offset) / config_.layer_step] == 0) {
      max_block_num_after_retrieval_ = max_block_num;
      block_table_after_retrieval_.swap(block_table_before_retrieval_);
    } else {
      max_block_num_after_retrieval_ =
          selected_blocks_num_history_[(layer_idx - config_.layer_offset) / config_.layer_step];
      for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        for (int i = 0; i < max_block_num_after_retrieval_; i++) {
          block_table_after_retrieval_[batch_idx][i] =
              selected_blocks_history_[(layer_idx - config_.layer_offset) / config_.layer_step][batch_idx][i];
        }

        if (cache_seqlens[batch_idx] % config_.block_len == 1) {
          selected_blocks_num_history_[(layer_idx - config_.layer_offset) / config_.layer_step] += 1;
          int x = selected_blocks_num_history_[(layer_idx - config_.layer_offset) / config_.layer_step];
          int last_block_idx = block_table_before_retrieval_[batch_idx][cache_seqlens[batch_idx] / config_.block_len];
          selected_blocks_history_[(layer_idx - config_.layer_offset) / config_.layer_step][batch_idx][x - 1] =
              last_block_idx;
          block_table_after_retrieval_[batch_idx][x - 1] = last_block_idx;
        }
        cache_seqlens_[batch_idx] =
            (cache_seqlens_[batch_idx] % config_.block_len) +
            selected_blocks_num_history_[(layer_idx - config_.layer_offset) / config_.layer_step] * config_.block_len -
            config_.block_len;
      }
    }
  } else if (pick_block_num != -1) {
    // 刷新步：重新计算候选分数，并据此构造新的 prefix + Top-K + local + tail 表。
    max_block_num_after_retrieval_ = std::min(max_block_num, init_block_num + pick_block_num + local_block_num + 1);
    // 为每个中间候选 block 计算 Query-Anchor 近似相关度，作为下一步 Top-K 选择的排序依据。
    calculate_block_similarity_layer_(q_in_data, batch_size, layer_idx, q_len, max_block_num, cache_seqlens,
                                      init_block_num, local_block_num, pick_block_num, backend);
    // 根据候选分数选出 Top-K，并与 prefix、local、尾 block 拼成 Attention 使用的稀疏 block 表。
    select_block_layer_(batch_size, layer_idx, max_block_num, init_block_num, local_block_num, pick_block_num);
  } else {
    // Dense 模式：不筛选 block，Attention 直接遍历完整输入表。
    selected_blocks_num_history_[(layer_idx - config_.layer_offset) / config_.layer_step] = 0;
    max_block_num_after_retrieval_ = max_block_num;
    block_table_after_retrieval_.swap(block_table_before_retrieval_);
  }

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  //     printf("layer %d time of retrieval kvcache: %f s\n", layer_idx,
  //     std::chrono::duration<double>(end - start).count());
}
/**
 * @brief 测量 layer 共享的已选 block 对每个 Query head 保留的 Attention 质量。
 *
 * 函数分别在原始 block 表和检索后 block 表上计算 Attention 统计量，再报告已选集合保留了多少完整
 * Attention 分布。该函数仅用于分析，不生成模型激活值。
 *
 * @param q_in_data 用于计算 Attention 分数的 FP16 Query。
 * @param attn_sparsity FP32 输出数组，每个 Query head 写入一个保留质量或稀疏度值。
 * @param batch_size 序列数量。
 * @param max_block_num 已选 block 表宽度。
 * @param block_table_origin 原始完整的逻辑到物理 block 映射。
 * @param cache_seqlens_origin 完整 cache 的原始有效 token 数。
 * @param max_block_num_origin block_table_origin 的宽度。
 * @param backend 用于 block Attention 和结果归并的工作线程池。
 */
void KVCache::calculate_sparsity_layer_(const uint16_t* q_in_data, float* attn_sparsity, int batch_size,
                                        int max_block_num, int* block_table, int* cache_seqlens, WorkerPool* backend

) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();
  seq_len_ = config_.block_len;
  backend->do_work_stealing_job(
      batch_size * config_.kv_head_num * max_block_num,
      [&](int thread_id) {
        thread_cur_head_idx_[thread_id].first = -1;
        thread_cur_head_idx_[thread_id].second = -1;
      },
      [&](int task_id) {
        int batch_id = task_id / (config_.kv_head_num * max_block_num);
        int head_id = (task_id % (config_.kv_head_num * max_block_num)) / max_block_num;
        int block_id = task_id % max_block_num;
        int thread_id = WorkerPool::thread_local_id;
        // If the block is out of the sequence length, skip it.
        if (cache_seqlens[batch_id] / config_.block_len < block_id) {
          return;
        }
        int block_idx = block_table[batch_id * max_block_num + block_id];
        if (cache_seqlens_[batch_id] / config_.block_len == block_id) {
          int seq_len = cache_seqlens_[batch_id] % config_.block_len;
          if (seq_len == 0) return;

          // Prepare the attention mask for the last block.
          int full_blocks = seq_len / 8;
          int remaining_bits = seq_len % 8;
          // Fill full blocks with 1s
          for (int i = 0; i < full_blocks; ++i) {
            thread_local_attn_mask_[thread_id][i] = 0xFF;
          }
          // Fill the remaining bits in the next block
          if (remaining_bits > 0 && full_blocks < seq_len_ / 8) {
            thread_local_attn_mask_[thread_id][full_blocks] = (1 << remaining_bits) - 1;
          } else {
            thread_local_attn_mask_[thread_id][full_blocks] = 0;
          }

          for (int i = full_blocks + 1; i < seq_len_ / 8; ++i) {
            thread_local_attn_mask_[thread_id][i] = 0;
          }
          if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
            attn_with_kvcache_one_block_(config_.head_dim, config_.q_head_num / config_.kv_head_num, GGML_TYPE_F16,
                                         (void*)&q_in_data[batch_id * config_.kv_head_num * n_gqa_ * config_.head_dim +
                                                           head_id * n_gqa_ * config_.head_dim],
                                         seq_len_, 0, false, thread_local_attn_mask_[thread_id].data(), GGML_TYPE_F16,
                                         0, k_cache_fp16_[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                                         GGML_TYPE_F16, 1, v_cache_fp16_[layer_id_][head_id][block_idx].data(), 0,
                                         nullptr, nullptr, thread_local_attn_score_[thread_id].data(),
                                         thread_local_output_fp32_[thread_id].data(),
                                         thread_local_attn_lse_[thread_id].data(),
                                         thread_local_draft_[thread_id].data(), nullptr, cos_.data(), sin_.data());
          } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
            attn_with_kvcache_one_block_(
                config_.head_dim, config_.q_head_num / config_.kv_head_num, GGML_TYPE_Q8_0,
                q_q8_0_[batch_id][head_id].data(), seq_len_, 0, false, thread_local_attn_mask_[thread_id].data(),
                GGML_TYPE_Q4_0, 0, k_cache_q4[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                GGML_TYPE_Q4_0, 1, v_cache_q4[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                thread_local_attn_score_[thread_id].data(), thread_local_output_q8_0_[thread_id].data(),
                thread_local_attn_lse_[thread_id].data(), thread_local_draft_[thread_id].data(), nullptr, cos_.data(),
                sin_.data());
            dequantize_row_q8_0(thread_local_output_q8_0_[thread_id].data(),
                                thread_local_output_fp32_[thread_id].data(), n_gqa_ * config_.head_dim);
          } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
            attn_with_kvcache_one_block_(
                config_.head_dim, config_.q_head_num / config_.kv_head_num, GGML_TYPE_Q8_0,
                q_q8_0_[batch_id][head_id].data(), seq_len_, 0, false, thread_local_attn_mask_[thread_id].data(),
                GGML_TYPE_Q8_0, 0, k_cache_q8[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                GGML_TYPE_Q8_0, 1, v_cache_q8[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                thread_local_attn_score_[thread_id].data(), thread_local_output_q8_0_[thread_id].data(),
                thread_local_attn_lse_[thread_id].data(), thread_local_draft_[thread_id].data(), nullptr, cos_.data(),
                sin_.data());
            dequantize_row_q8_0(thread_local_output_q8_0_[thread_id].data(),
                                thread_local_output_fp32_[thread_id].data(), n_gqa_ * config_.head_dim);
          }
        } else {
          if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
            attn_with_kvcache_one_block_(
                config_.head_dim, config_.q_head_num / config_.kv_head_num, GGML_TYPE_F16,
                (void*)&q_in_data[batch_id * config_.kv_head_num * n_gqa_ * config_.head_dim +
                                  head_id * n_gqa_ * config_.head_dim],
                seq_len_, 0, true, nullptr, GGML_TYPE_F16, 0, k_cache_fp16_[layer_id_][head_id][block_idx].data(), 0,
                nullptr, nullptr, GGML_TYPE_F16, 1, v_cache_fp16_[layer_id_][head_id][block_idx].data(), 0, nullptr,
                nullptr, thread_local_attn_score_[thread_id].data(), thread_local_output_fp32_[thread_id].data(),
                thread_local_attn_lse_[thread_id].data(), thread_local_draft_[thread_id].data(), nullptr, cos_.data(),
                sin_.data());

          } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
            attn_with_kvcache_one_block_(config_.head_dim, config_.q_head_num / config_.kv_head_num, GGML_TYPE_Q8_0,
                                         q_q8_0_[batch_id][head_id].data(), seq_len_, 0, true, nullptr, GGML_TYPE_Q4_0,
                                         0, k_cache_q4[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                                         GGML_TYPE_Q4_0, 1, v_cache_q4[layer_id_][head_id][block_idx].data(), 0,
                                         nullptr, nullptr, thread_local_attn_score_[thread_id].data(),
                                         thread_local_output_q8_0_[thread_id].data(),
                                         thread_local_attn_lse_[thread_id].data(),
                                         thread_local_draft_[thread_id].data(), nullptr, cos_.data(), sin_.data());
            dequantize_row_q8_0(thread_local_output_q8_0_[thread_id].data(),
                                thread_local_output_fp32_[thread_id].data(), n_gqa_ * config_.head_dim);
          } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
            attn_with_kvcache_one_block_(config_.head_dim, config_.q_head_num / config_.kv_head_num, GGML_TYPE_Q8_0,
                                         q_q8_0_[batch_id][head_id].data(), seq_len_, 0, true, nullptr, GGML_TYPE_Q8_0,
                                         0, k_cache_q8[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                                         GGML_TYPE_Q8_0, 1, v_cache_q8[layer_id_][head_id][block_idx].data(), 0,
                                         nullptr, nullptr, thread_local_attn_score_[thread_id].data(),
                                         thread_local_output_q8_0_[thread_id].data(),
                                         thread_local_attn_lse_[thread_id].data(),
                                         thread_local_draft_[thread_id].data(), nullptr, cos_.data(), sin_.data());
            dequantize_row_q8_0(thread_local_output_q8_0_[thread_id].data(),
                                thread_local_output_fp32_[thread_id].data(), n_gqa_ * config_.head_dim);
          }
        }
        for (int i = 0; i < n_gqa_; i++) {
          block_lse_[batch_id][block_idx][head_id * n_gqa_ + i] = thread_local_attn_lse_[thread_id][i];
        }
        int cur_batch_idx = thread_cur_head_idx_[thread_id].first;
        int cur_head_id = thread_cur_head_idx_[thread_id].second;
        if (batch_id == cur_batch_idx && head_id == cur_head_id) {
          for (int i = 0; i < n_gqa_; i++) {
            float new_attn_lse = thread_local_cur_attn_lse_[thread_id][i] +
                                 std::log(1.0 + std::exp(thread_local_attn_lse_[thread_id][i] -
                                                         thread_local_cur_attn_lse_[thread_id][i]));
            ggml_vec_scale_f32(config_.head_dim, thread_local_cur_output_fp32_[thread_id].data() + i * config_.head_dim,
                               std::exp(thread_local_cur_attn_lse_[thread_id][i] - new_attn_lse));
            ggml_vec_scale_f32(config_.head_dim, thread_local_output_fp32_[thread_id].data() + i * config_.head_dim,
                               std::exp(thread_local_attn_lse_[thread_id][i] - new_attn_lse));
            for (int j = 0; j < config_.head_dim; j++) {
              thread_local_cur_output_fp32_[thread_id][i * config_.head_dim + j] +=
                  thread_local_output_fp32_[thread_id][i * config_.head_dim + j];
            }
            thread_local_cur_attn_lse_[thread_id][i] = new_attn_lse;
          }
        } else {
          if (cur_batch_idx != -1) {
            mutex_[cur_batch_idx][cur_head_id]->lock();
            for (int i = 0; i < n_gqa_; i++) {
              if (std::abs(attn_lse_[cur_batch_idx][cur_head_id][i]) < 1e-6) {
                attn_lse_[cur_batch_idx][cur_head_id][i] = thread_local_cur_attn_lse_[thread_id][i];
                for (int j = 0; j < config_.head_dim; j++) {
                  output_fp32_[cur_batch_idx][cur_head_id][i * config_.head_dim + j] =
                      thread_local_cur_output_fp32_[thread_id][i * config_.head_dim + j];
                }
                continue;
              }
              float new_attn_lse = attn_lse_[cur_batch_idx][cur_head_id][i] +
                                   std::log(1.0 + std::exp(thread_local_cur_attn_lse_[thread_id][i] -
                                                           attn_lse_[cur_batch_idx][cur_head_id][i]));
              ggml_vec_scale_f32(config_.head_dim,
                                 output_fp32_[cur_batch_idx][cur_head_id].data() + i * config_.head_dim,
                                 std::exp(attn_lse_[cur_batch_idx][cur_head_id][i] - new_attn_lse));
              ggml_vec_scale_f32(config_.head_dim,
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
            thread_local_cur_attn_lse_[thread_id][i] = thread_local_attn_lse_[thread_id][i];
            for (int j = 0; j < config_.head_dim; j++) {
              thread_local_cur_output_fp32_[thread_id][i * config_.head_dim + j] =
                  thread_local_output_fp32_[thread_id][i * config_.head_dim + j];
            }
          }
        }
      },
      // Merge the results of the remaining blocks.
      [&](int thread_id) {
        int cur_batch_idx = thread_cur_head_idx_[thread_id].first;
        int cur_head_id = thread_cur_head_idx_[thread_id].second;
        if (cur_head_id != -1) {
          mutex_[cur_batch_idx][cur_head_id]->lock();
          for (int i = 0; i < n_gqa_; i++) {
            float new_attn_lse;
            if (std::abs(attn_lse_[cur_batch_idx][cur_head_id][i]) < 1e-6) {
              attn_lse_[cur_batch_idx][cur_head_id][i] = thread_local_cur_attn_lse_[thread_id][i];
              for (int j = 0; j < config_.head_dim; j++) {
                output_fp32_[cur_batch_idx][cur_head_id][i * config_.head_dim + j] =
                    thread_local_cur_output_fp32_[thread_id][i * config_.head_dim + j];
              }
              continue;
            }
            new_attn_lse = attn_lse_[cur_batch_idx][cur_head_id][i] +
                           std::log(1.0 + std::exp(thread_local_cur_attn_lse_[thread_id][i] -
                                                   attn_lse_[cur_batch_idx][cur_head_id][i]));
            ggml_vec_scale_f32(config_.head_dim, output_fp32_[cur_batch_idx][cur_head_id].data() + i * config_.head_dim,
                               std::exp(attn_lse_[cur_batch_idx][cur_head_id][i] - new_attn_lse));
            ggml_vec_scale_f32(config_.head_dim, thread_local_cur_output_fp32_[thread_id].data() + i * config_.head_dim,
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

  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < max_block_num_after_retrieval_; j++) {
      int block_idx = block_table_after_retrieval_[i][j];
      for (int k = 0; k < config_.q_head_num; k++) {
        attn_sparsity[i * config_.q_head_num + k] +=
            std::exp(block_lse_[i][block_idx][k] - attn_lse_[i][k / n_gqa_][k % n_gqa_]);
      }
    }
  }

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  // printf("layer %d time of calculating sparsity: %f s\n", layer_id_,
  //        diff.count());
}

/**
 * @brief 重置累加状态，并把输入 block 表扩展为按 KV head 独立检索的形式。
 *
 * 调用方为每个序列提供一份逻辑 block 映射。函数把该映射复制到每个 KV-head 通道，清空相似度和输出
 * 状态，并记录有效序列长度，供后续打分和 Attention 使用。
 *
 * @param batch_size 序列数量。
 * @param layer_idx 当前层编号；保留该参数以与 layer 共享初始化接口一致。
 * @param block_table 行优先 [batch_size, max_block_num] 逻辑到物理 block 映射。
 * @param max_block_num 输入/输出可供检索使用的 block 表宽度。
 * @param cache_seqlens 每个序列的有效 token 数。
 */
void KVCache::attn_initialize_kvhead_(int batch_size, int layer_idx, int* block_table, int& max_block_num,
                                      int* cache_seqlens) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();

  // 对于每个batch，把output和attn_lse_初始化为0
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    // initialize output_fp32_ and attn_lse_
    for (int i = 0; i < config_.kv_head_num; i++) {
      for (int j = 0; j < n_gqa_ * config_.head_dim; j++) {
        output_fp32_[batch_idx][i][j] = 0;
      }
      for (int j = 0; j < n_gqa_; j++) {
        attn_lse_[batch_idx][i][j] = 0;
      }
    }

    // clear top_similar_block_
    while (!top_similar_block_[batch_idx].empty()) top_similar_block_[batch_idx].pop();
  }

  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    cache_seqlens_[batch_idx] = cache_seqlens[batch_idx];
    for (int i = 0; i < max_block_num; i++) {
      for (int j = 0; j < config_.kv_head_num; j++) {
        block_table_before_retrieval_kvhead_[batch_idx][i][j] = block_table[batch_idx * max_block_num + i];
        block_similar_kv_head_[batch_idx][i][j] = 0;
      }
    }
  }

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  // printf("layer %d time of initializing attn: %f s\n", layer_idx,
  //        std::chrono::duration<double>(end - start).count());
}
/**
 * @brief 为每个 KV head 选择或复用一份独立的检索后 block 表。
 *
 * 刷新步骤按 KV head 独立打分和选择 block，其余步骤恢复已保存的选择。pick_block_num 为 -1 时把所有
 * 输入 block 复制到每个 KV-head 表，从而启用 dense GQA Attention。
 *
 * @param q_in_data 用于检索打分的 FP16 Query。
 * @param init_block_num 每个 KV head 始终保留的开头 block 数。
 * @param local_block_num 每个 KV head 始终保留的最近完整 block 数。
 * @param pick_block_num 每个 KV head 的相似中间 block 预算；-1 表示使用全部 block。
 * @param q_len 计算平均 Q 时使用的 Query token 数。
 * @param generate_token_idx 控制选择结果复用的 decode token 编号。
 * @param batch_size 序列数量。
 * @param layer_idx 控制选择复用并索引历史的当前层编号。
 * @param cache_seqlens 调用方有效 token 数。
 * @param max_block_num 输入/输出可用 block 表宽度。
 * @param backend 用于打分的工作线程池。
 */
void KVCache::retrieval_kvcache_kvhead_(const uint16_t* q_in_data, int init_block_num, int local_block_num,
                                        int pick_block_num, int q_len, int generate_token_idx, int batch_size,
                                        int layer_idx, int* cache_seqlens, int& max_block_num, WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();
  max_block_num_after_retrieval_ = 0;
  if (pick_block_num != -1 &&
      (generate_token_idx % config_.token_step != 0 || (layer_idx % config_.layer_step != config_.layer_offset))) {
    if (selected_blocks_num_history_[(layer_idx - config_.layer_offset) / config_.layer_step] == 0) {
      max_block_num_after_retrieval_ = max_block_num;
      for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        for (int i = 0; i < max_block_num; i++) {
          for (int j = 0; j < config_.kv_head_num; j++) {
            block_table_after_retrieval_kvhead_[batch_idx][i][j] =
                block_table_before_retrieval_kvhead_[batch_idx][i][j];
          }
        }
      }
    } else {
      max_block_num_after_retrieval_ =
          selected_blocks_num_history_[(layer_idx - config_.layer_offset) / config_.layer_step];

      for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        for (int i = 0; i < max_block_num_after_retrieval_; i++) {
          for (int j = 0; j < config_.kv_head_num; j++) {
            block_table_after_retrieval_kvhead_[batch_idx][i][j] =
                selected_blocks_history_kvhead_[(layer_idx - config_.layer_offset) / config_.layer_step][batch_idx][i]
                                               [j];
          }
        }

        if (cache_seqlens[batch_idx] % config_.block_len == 1) {
          selected_blocks_num_history_[(layer_idx - config_.layer_offset) / config_.layer_step] += 1;
          int x = selected_blocks_num_history_[(layer_idx - config_.layer_offset) / config_.layer_step];
          for (int i = 0; i < config_.kv_head_num; i++) {
            int last_block_idx =
                block_table_before_retrieval_kvhead_[batch_idx][cache_seqlens[batch_idx] / config_.block_len][i];
            selected_blocks_history_kvhead_[(layer_idx - config_.layer_offset) / config_.layer_step][batch_idx][x - 1]
                                           [i] = last_block_idx;
            block_table_after_retrieval_kvhead_[batch_idx][x - 1][i] = last_block_idx;
          }
        }
        cache_seqlens_[batch_idx] = std::min(
            cache_seqlens_[batch_idx], (cache_seqlens_[batch_idx] % config_.block_len) +
                                           (init_block_num + pick_block_num + local_block_num) * config_.block_len);
      }
    }
  } else if (pick_block_num != -1) {
    max_block_num_after_retrieval_ = std::min(max_block_num, init_block_num + pick_block_num + local_block_num + 1);
    calculate_block_similarity_kvhead_(q_in_data, batch_size, layer_idx, q_len, max_block_num, cache_seqlens,
                                       init_block_num, local_block_num, pick_block_num, backend);
    select_block_kvhead_(batch_size, layer_idx, max_block_num, init_block_num, local_block_num, pick_block_num);
  } else {
    selected_blocks_num_history_[(layer_idx - config_.layer_offset) / config_.layer_step] = 0;
    max_block_num_after_retrieval_ = max_block_num;
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
      for (int i = 0; i < max_block_num; i++) {
        for (int j = 0; j < config_.kv_head_num; j++) {
          block_table_after_retrieval_kvhead_[batch_idx][i][j] = block_table_before_retrieval_kvhead_[batch_idx][i][j];
        }
      }
    }
  }

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  // printf("layer %d time of retrieval kvcache: %f s\n", layer_idx,
  //        std::chrono::duration<double>(end - start).count());
}
/**
 * @brief 测量每个 KV head 使用独立 block 表时保留的 Attention 质量。
 *
 * 函数在已选 cache 和原始 cache 上计算精确的 block Attention 统计量，并归并为每个 Query head 一个
 * 诊断值。该路径用于分析稀疏度，不生成推理输出。
 *
 * @param q_in_data 用于计算 Attention 分数的 FP16 Query。
 * @param attn_sparsity FP32 诊断值输出，每个 Query head 对应一个值。
 * @param batch_size 序列数量。
 * @param max_block_num 已选 per-KV-head block 表宽度。
 * @param block_table_origin 原始完整的逻辑到物理 block 映射。
 * @param cache_seqlens_origin 原始有效 token 数。
 * @param max_block_num_origin block_table_origin 的宽度。
 * @param backend 用于 Attention 任务和结果归并的工作线程池。
 */
void KVCache::calculate_sparsity_kvhead_(const uint16_t* q_in_data, float* attn_sparsity, int batch_size,
                                         int max_block_num, int* block_table, int* cache_seqlens, WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();
  seq_len_ = config_.block_len;
  backend->do_work_stealing_job(
      batch_size * config_.kv_head_num * max_block_num,
      [&](int thread_id) {
        thread_cur_head_idx_[thread_id].first = -1;
        thread_cur_head_idx_[thread_id].second = -1;
      },
      [&](int task_id) {
        int batch_id = task_id / (config_.kv_head_num * max_block_num);
        int head_id = (task_id % (config_.kv_head_num * max_block_num)) / max_block_num;
        int block_id = task_id % max_block_num;
        int thread_id = WorkerPool::thread_local_id;
        // If the block is out of the sequence length, skip it.
        if (cache_seqlens[batch_id] / config_.block_len < block_id) {
          return;
        }
        int block_idx = block_table[batch_id * max_block_num + block_id];
        if (cache_seqlens_[batch_id] / config_.block_len == block_id) {
          int seq_len = cache_seqlens_[batch_id] % config_.block_len;
          if (seq_len == 0) return;

          // Prepare the attention mask for the last block.
          int full_blocks = seq_len / 8;
          int remaining_bits = seq_len % 8;

          // Fill full blocks with 1s
          for (int i = 0; i < full_blocks; ++i) {
            thread_local_attn_mask_[thread_id][i] = 0xFF;
          }
          // Fill the remaining bits in the next block
          if (remaining_bits > 0 && full_blocks < seq_len_ / 8) {
            thread_local_attn_mask_[thread_id][full_blocks] = (1 << remaining_bits) - 1;
          } else {
            thread_local_attn_mask_[thread_id][full_blocks] = 0;
          }

          for (int i = full_blocks + 1; i < seq_len_ / 8; ++i) {
            thread_local_attn_mask_[thread_id][i] = 0;
          }
          if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
            attn_with_kvcache_one_block_(config_.head_dim, config_.q_head_num / config_.kv_head_num, GGML_TYPE_F16,
                                         (void*)&q_in_data[batch_id * config_.kv_head_num * n_gqa_ * config_.head_dim +
                                                           head_id * n_gqa_ * config_.head_dim],
                                         seq_len_, 0, false, thread_local_attn_mask_[thread_id].data(), GGML_TYPE_F16,
                                         0, k_cache_fp16_[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                                         GGML_TYPE_F16, 1, v_cache_fp16_[layer_id_][head_id][block_idx].data(), 0,
                                         nullptr, nullptr, thread_local_attn_score_[thread_id].data(),
                                         thread_local_output_fp32_[thread_id].data(),
                                         thread_local_attn_lse_[thread_id].data(),
                                         thread_local_draft_[thread_id].data(), nullptr, cos_.data(), sin_.data());
          } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
            attn_with_kvcache_one_block_(
                config_.head_dim, config_.q_head_num / config_.kv_head_num, GGML_TYPE_Q8_0,
                q_q8_0_[batch_id][head_id].data(), seq_len_, 0, false, thread_local_attn_mask_[thread_id].data(),
                GGML_TYPE_Q4_0, 0, k_cache_q4[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                GGML_TYPE_Q4_0, 1, v_cache_q4[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                thread_local_attn_score_[thread_id].data(), thread_local_output_q8_0_[thread_id].data(),
                thread_local_attn_lse_[thread_id].data(), thread_local_draft_[thread_id].data(), nullptr, cos_.data(),
                sin_.data());
            dequantize_row_q8_0(thread_local_output_q8_0_[thread_id].data(),
                                thread_local_output_fp32_[thread_id].data(), n_gqa_ * config_.head_dim);
          } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
            attn_with_kvcache_one_block_(
                config_.head_dim, config_.q_head_num / config_.kv_head_num, GGML_TYPE_Q8_0,
                q_q8_0_[batch_id][head_id].data(), seq_len_, 0, false, thread_local_attn_mask_[thread_id].data(),
                GGML_TYPE_Q8_0, 0, k_cache_q8[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                GGML_TYPE_Q8_0, 1, v_cache_q8[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                thread_local_attn_score_[thread_id].data(), thread_local_output_q8_0_[thread_id].data(),
                thread_local_attn_lse_[thread_id].data(), thread_local_draft_[thread_id].data(), nullptr, cos_.data(),
                sin_.data());
            dequantize_row_q8_0(thread_local_output_q8_0_[thread_id].data(),
                                thread_local_output_fp32_[thread_id].data(), n_gqa_ * config_.head_dim);
          }
        } else {
          if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
            attn_with_kvcache_one_block_(
                config_.head_dim, config_.q_head_num / config_.kv_head_num, GGML_TYPE_F16,
                (void*)&q_in_data[batch_id * config_.kv_head_num * n_gqa_ * config_.head_dim +
                                  head_id * n_gqa_ * config_.head_dim],
                seq_len_, 0, true, nullptr, GGML_TYPE_F16, 0, k_cache_fp16_[layer_id_][head_id][block_idx].data(), 0,
                nullptr, nullptr, GGML_TYPE_F16, 1, v_cache_fp16_[layer_id_][head_id][block_idx].data(), 0, nullptr,
                nullptr, thread_local_attn_score_[thread_id].data(), thread_local_output_fp32_[thread_id].data(),
                thread_local_attn_lse_[thread_id].data(), thread_local_draft_[thread_id].data(), nullptr, cos_.data(),
                sin_.data());

          } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
            attn_with_kvcache_one_block_(config_.head_dim, config_.q_head_num / config_.kv_head_num, GGML_TYPE_Q8_0,
                                         q_q8_0_[batch_id][head_id].data(), seq_len_, 0, true, nullptr, GGML_TYPE_Q4_0,
                                         0, k_cache_q4[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                                         GGML_TYPE_Q4_0, 1, v_cache_q4[layer_id_][head_id][block_idx].data(), 0,
                                         nullptr, nullptr, thread_local_attn_score_[thread_id].data(),
                                         thread_local_output_q8_0_[thread_id].data(),
                                         thread_local_attn_lse_[thread_id].data(),
                                         thread_local_draft_[thread_id].data(), nullptr, cos_.data(), sin_.data());
            dequantize_row_q8_0(thread_local_output_q8_0_[thread_id].data(),
                                thread_local_output_fp32_[thread_id].data(), n_gqa_ * config_.head_dim);
          } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
            attn_with_kvcache_one_block_(config_.head_dim, config_.q_head_num / config_.kv_head_num, GGML_TYPE_Q8_0,
                                         q_q8_0_[batch_id][head_id].data(), seq_len_, 0, true, nullptr, GGML_TYPE_Q8_0,
                                         0, k_cache_q8[layer_id_][head_id][block_idx].data(), 0, nullptr, nullptr,
                                         GGML_TYPE_Q8_0, 1, v_cache_q8[layer_id_][head_id][block_idx].data(), 0,
                                         nullptr, nullptr, thread_local_attn_score_[thread_id].data(),
                                         thread_local_output_q8_0_[thread_id].data(),
                                         thread_local_attn_lse_[thread_id].data(),
                                         thread_local_draft_[thread_id].data(), nullptr, cos_.data(), sin_.data());
            dequantize_row_q8_0(thread_local_output_q8_0_[thread_id].data(),
                                thread_local_output_fp32_[thread_id].data(), n_gqa_ * config_.head_dim);
          }
        }
        for (int i = 0; i < n_gqa_; i++) {
          block_lse_[batch_id][block_idx][head_id * n_gqa_ + i] = thread_local_attn_lse_[thread_id][i];
        }
        int cur_batch_idx = thread_cur_head_idx_[thread_id].first;
        int cur_head_id = thread_cur_head_idx_[thread_id].second;
        if (batch_id == cur_batch_idx && head_id == cur_head_id) {
          for (int i = 0; i < n_gqa_; i++) {
            float new_attn_lse = thread_local_cur_attn_lse_[thread_id][i] +
                                 std::log(1.0 + std::exp(thread_local_attn_lse_[thread_id][i] -
                                                         thread_local_cur_attn_lse_[thread_id][i]));
            ggml_vec_scale_f32(config_.head_dim, thread_local_cur_output_fp32_[thread_id].data() + i * config_.head_dim,
                               std::exp(thread_local_cur_attn_lse_[thread_id][i] - new_attn_lse));
            ggml_vec_scale_f32(config_.head_dim, thread_local_output_fp32_[thread_id].data() + i * config_.head_dim,
                               std::exp(thread_local_attn_lse_[thread_id][i] - new_attn_lse));
            for (int j = 0; j < config_.head_dim; j++) {
              thread_local_cur_output_fp32_[thread_id][i * config_.head_dim + j] +=
                  thread_local_output_fp32_[thread_id][i * config_.head_dim + j];
            }
            thread_local_cur_attn_lse_[thread_id][i] = new_attn_lse;
          }
        } else {
          if (cur_batch_idx != -1) {
            mutex_[cur_batch_idx][cur_head_id]->lock();
            for (int i = 0; i < n_gqa_; i++) {
              if (std::abs(attn_lse_[cur_batch_idx][cur_head_id][i]) < 1e-6) {
                attn_lse_[cur_batch_idx][cur_head_id][i] = thread_local_cur_attn_lse_[thread_id][i];
                for (int j = 0; j < config_.head_dim; j++) {
                  output_fp32_[cur_batch_idx][cur_head_id][i * config_.head_dim + j] =
                      thread_local_cur_output_fp32_[thread_id][i * config_.head_dim + j];
                }
                continue;
              }
              float new_attn_lse = attn_lse_[cur_batch_idx][cur_head_id][i] +
                                   std::log(1.0 + std::exp(thread_local_cur_attn_lse_[thread_id][i] -
                                                           attn_lse_[cur_batch_idx][cur_head_id][i]));
              ggml_vec_scale_f32(config_.head_dim,
                                 output_fp32_[cur_batch_idx][cur_head_id].data() + i * config_.head_dim,
                                 std::exp(attn_lse_[cur_batch_idx][cur_head_id][i] - new_attn_lse));
              ggml_vec_scale_f32(config_.head_dim,
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
            thread_local_cur_attn_lse_[thread_id][i] = thread_local_attn_lse_[thread_id][i];
            for (int j = 0; j < config_.head_dim; j++) {
              thread_local_cur_output_fp32_[thread_id][i * config_.head_dim + j] =
                  thread_local_output_fp32_[thread_id][i * config_.head_dim + j];
            }
          }
        }
      },
      // Merge the results of the remaining blocks.
      [&](int thread_id) {
        int cur_batch_idx = thread_cur_head_idx_[thread_id].first;
        int cur_head_id = thread_cur_head_idx_[thread_id].second;
        if (cur_head_id != -1) {
          mutex_[cur_batch_idx][cur_head_id]->lock();
          for (int i = 0; i < n_gqa_; i++) {
            float new_attn_lse;
            if (std::abs(attn_lse_[cur_batch_idx][cur_head_id][i]) < 1e-6) {
              attn_lse_[cur_batch_idx][cur_head_id][i] = thread_local_cur_attn_lse_[thread_id][i];
              for (int j = 0; j < config_.head_dim; j++) {
                output_fp32_[cur_batch_idx][cur_head_id][i * config_.head_dim + j] =
                    thread_local_cur_output_fp32_[thread_id][i * config_.head_dim + j];
              }
              continue;
            }
            new_attn_lse = attn_lse_[cur_batch_idx][cur_head_id][i] +
                           std::log(1.0 + std::exp(thread_local_cur_attn_lse_[thread_id][i] -
                                                   attn_lse_[cur_batch_idx][cur_head_id][i]));
            ggml_vec_scale_f32(config_.head_dim, output_fp32_[cur_batch_idx][cur_head_id].data() + i * config_.head_dim,
                               std::exp(attn_lse_[cur_batch_idx][cur_head_id][i] - new_attn_lse));
            ggml_vec_scale_f32(config_.head_dim, thread_local_cur_output_fp32_[thread_id].data() + i * config_.head_dim,
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

  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < max_block_num_after_retrieval_; j++) {
      for (int k = 0; k < config_.q_head_num; k++) {
        int block_idx = block_table_after_retrieval_kvhead_[i][j][k / n_gqa_];
        attn_sparsity[i * config_.q_head_num + k] +=
            std::exp(block_lse_[i][block_idx][k] - attn_lse_[i][k / n_gqa_][k % n_gqa_]);
      }
    }
  }

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  // printf("layer %d time of calculating sparsity: %f s\n", layer_id_,
  //        diff.count());
}
/**
 * @brief 按 KV head 独立计算候选 block 的相似度分数。
 *
 * 对每个候选 block，Query heads 会映射到其共享的 KV head。分数在每个维度上取多个 anchor 的最大
 * Query-anchor 乘积，再对该 KV head 对应的 Query heads 和全部维度求和。prefix 和 local block 会被
 * 无条件保留，因此不参与打分。
 *
 * @param q_in_data FP16 Query，形状为 [batch_size, q_len, q_head_num, head_dim]。
 * @param batch_size 序列数量。
 * @param layer_idx 要读取 anchor 的模型层。
 * @param q_len 用于计算平均 Q 的 Query token 数。
 * @param max_block_num 输入 block 表宽度。
 * @param cache_seqlens 调用方有效 token 数；候选范围由内部副本确定。
 * @param init_block_num 不参与打分的开头 block 数。
 * @param local_block_num 不参与打分的最近完整 block 数。
 * @param pick_block_num 每个 KV head 的目标 Top-K 数。
 * @param backend 用于并行计算 block 分数的工作线程池。
 */
void KVCache::calculate_block_similarity_kvhead_(const uint16_t* q_in_data, int batch_size, int layer_idx, int q_len,
                                                 int max_block_num, int* cache_seqlens, int init_block_num,
                                                 int local_block_num, int pick_block_num, WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();
  backend->do_work_stealing_job(
      batch_size * max_block_num, nullptr,
      [&](int task_id) {
        int batch_id = task_id / max_block_num;
        int block_id = task_id % max_block_num;
        int seq_len = cache_seqlens_[batch_id];

        if (block_id < init_block_num || block_id >= (seq_len / config_.block_len) - local_block_num) {
          return;
        }
        int block_idx = block_table_before_retrieval_kvhead_[batch_id][block_id][0];

        for (int head_id = 0; head_id < config_.q_head_num; head_id++) {
          for (int i = 0; i < config_.head_dim; i++) {
            float q_i = 0, qa_i = std::numeric_limits<float>::lowest();
            for (int q_id = 0; q_id < q_len; q_id++) {
              q_i += GGML_FP16_TO_FP32(
                  q_in_data[batch_id * q_len * config_.q_head_num * config_.head_dim +
                            q_id * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + i]);
            }
            q_i /= q_len;
            for (int anchor_id = 0; anchor_id < config_.anchor_num; anchor_id++) {
              qa_i = std::max(
                  qa_i,
                  GGML_FP16_TO_FP32(
                      anchor_[layer_idx * config_.max_block_num * config_.anchor_num * config_.q_head_num *
                                  config_.head_dim +
                              block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                              anchor_id * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + i]) *
                      q_i);
            }
            block_similar_kv_head_[batch_id][block_id][head_id / n_gqa_] += qa_i;
          }
        }
      },
      nullptr);

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  // printf("layer %d time of calculating similarity: %f s\n", layer_idx,
  //        diff.count());
}
/**
 * @brief 为每个 KV head 构造一份检索后的 block 表。
 *
 * 对每个序列和 KV head，有界最小堆保留分数最高的中间 block。函数拼接 prefix、检索结果、local 和
 * 未填满的尾块，再保存选择结果，供非刷新层或后续 token 复用。
 *
 * @param batch_size 序列数量。
 * @param layer_idx 用于索引检索历史的模型层。
 * @param max_block_num 原始 block 表宽度。
 * @param init_block_num 始终保留的最早 block 数。
 * @param local_block_num 始终保留的最近完整 block 数。
 * @param pick_block_num 每个 KV head 最多保留的已打分中间 block 数。
 */
void KVCache::select_block_kvhead_(int batch_size, int layer_idx, int max_block_num, int init_block_num,
                                   int local_block_num, int pick_block_num) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();

  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    int cache_len_after_retrieval = 0;
    if (cache_seqlens_[batch_idx] / config_.block_len <= init_block_num + pick_block_num + local_block_num) {
      selected_blocks_num_history_[(layer_idx - config_.layer_offset) / config_.layer_step] = 0;
      for (int i = 0; i < max_block_num; i++) {
        for (int j = 0; j < config_.kv_head_num; j++) {
          block_table_after_retrieval_kvhead_[batch_idx][i][j] = block_table_before_retrieval_kvhead_[batch_idx][i][j];
        }
      }
      continue;
    }
    for (int head_id = 0; head_id < config_.kv_head_num; head_id++) {
      for (int block_id = init_block_num; block_id < (cache_seqlens_[batch_idx] / config_.block_len) - local_block_num;
           block_id++) {
        top_similar_block_[batch_idx].push(
            std::make_pair(block_similar_kv_head_[batch_idx][block_id][head_id],
                           block_table_before_retrieval_kvhead_[batch_idx][block_id][head_id]));
        if (top_similar_block_[batch_idx].size() > pick_block_num) {
          top_similar_block_[batch_idx].pop();
        }
      }

      int i = 0;
      for (; i < init_block_num; i++) {
        block_table_after_retrieval_kvhead_[batch_idx][i][head_id] =
            block_table_before_retrieval_kvhead_[batch_idx][i][head_id];
      }
      while (!top_similar_block_[batch_idx].empty()) {
        block_table_after_retrieval_kvhead_[batch_idx][i][head_id] = top_similar_block_[batch_idx].top().second;
        top_similar_block_[batch_idx].pop();
        i++;
      }
      for (; i < init_block_num + pick_block_num + local_block_num; i++) {
        block_table_after_retrieval_kvhead_[batch_idx][i][head_id] =
            block_table_before_retrieval_kvhead_[batch_idx][(cache_seqlens_[batch_idx] / config_.block_len) -
                                                            local_block_num + i - init_block_num - pick_block_num]
                                                [head_id];
      }
      if (cache_seqlens_[batch_idx] % config_.block_len != 0) {
        block_table_after_retrieval_kvhead_[batch_idx][i][head_id] =
            block_table_before_retrieval_kvhead_[batch_idx][(cache_seqlens_[batch_idx] / config_.block_len)][head_id];
        cache_len_after_retrieval = (cache_seqlens_[batch_idx] % config_.block_len) + i * config_.block_len;
        i++;
      } else {
        cache_len_after_retrieval = (cache_seqlens_[batch_idx] % config_.block_len) + i * config_.block_len;
      }
      for (int j = 0; j < i; j++) {
        selected_blocks_history_kvhead_[(layer_idx - config_.layer_offset) / config_.layer_step][batch_idx][j]
                                       [head_id] = block_table_after_retrieval_kvhead_[batch_idx][j][head_id];
      }
    }
    cache_seqlens_[batch_idx] = cache_len_after_retrieval;
    selected_blocks_num_history_[(layer_idx - config_.layer_offset) / config_.layer_step] =
        (cache_len_after_retrieval + config_.block_len - 1) / config_.block_len;
  }

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  // printf("layer %d time of selecting block: %f s\n", layer_idx,
  //        diff.count())
}

/**
 * @brief 对外诊断入口：执行 block 检索并测量其保留的 Attention 质量。
 *
 * 函数按正常 Attention 相同的方式准备 Q，执行配置的 layer 或 KV-head 检索策略，再由对应稀疏度分析
 * kernel 将已选 cache 与原始完整 cache 比较。
 *
 * @param q_in FP16 Query，形状为 [batch_size, q_len, q_head_num, head_dim]。
 * @param attn_sparsity FP32 稀疏度或保留质量输出，每个 Query head 对应一个值。
 * @param layer_idx 要分析的模型层。
 * @param generate_token_idx 控制检索刷新的 decode token 编号。
 * @param q_len Query token 数。
 * @param batch_size 序列数量。
 * @param max_block_num 候选 block 表宽度。
 * @param block_table 候选逻辑到物理 block 映射。
 * @param cache_seqlens 候选 cache 的有效 token 数。
 * @param block_table_origin 作为基准的完整 cache 逻辑到物理 block 映射。
 * @param cache_seqlens_origin 完整 cache 的有效 token 数。
 * @param max_block_num_origin block_table_origin 的宽度。
 * @param topk 相似中间 block 预算；-1 表示选择完整 cache。
 * @param local 始终保留的尾部完整 block 数。
 * @param backend 用于检索和分析的工作线程池。
 */
void KVCache::get_attn_sparsity(const ggml_fp16_t* q_in, float* attn_sparsity, int layer_idx, int generate_token_idx,
                                int q_len, int batch_size, int max_block_num, int* block_table, int* cache_seqlens,
                                int* block_table_origin, int* cache_seqlens_origin, int max_block_num_origin, int topk,
                                int local, WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();
  layer_id_ = layer_idx;
  int thread_num = backend->get_thread_num();
  batch_size = 1;

  const uint16_t* q_in_data = const_cast<const uint16_t*>(q_in);

  quantize_q_(q_in_data, batch_size);
  if (config_.retrieval_type == RetrievalType::LAYER) {
    attn_initialize_layer_(batch_size, layer_idx, block_table, max_block_num, cache_seqlens);
    // 先生成与真实 Layer Attention 相同的稀疏 block 集合，供后续 calculate_sparsity_layer_() 评估保留质量。
    retrieval_kvcache_layer_(q_in_data, 1, local, topk, q_len, generate_token_idx, batch_size, layer_idx, cache_seqlens,
                             max_block_num, backend);
    calculate_sparsity_layer_(q_in_data, attn_sparsity, batch_size, max_block_num_origin, block_table_origin,
                              cache_seqlens_origin, backend);
  } else if (config_.retrieval_type == RetrievalType::KVHEAD) {
    attn_initialize_kvhead_(batch_size, layer_idx, block_table, max_block_num, cache_seqlens);
    retrieval_kvcache_kvhead_(q_in_data, 1, local, topk, q_len, generate_token_idx, batch_size, layer_idx,
                              cache_seqlens, max_block_num, backend);
    calculate_sparsity_kvhead_(q_in_data, attn_sparsity, batch_size, max_block_num_origin, block_table_origin,
                               cache_seqlens_origin, backend);
  }
}

/**
 * @brief 为一个物理 KV-cache block 计算精确的 scaled dot-product Attention。
 *
 * kernel 依次计算 QK、缩放、可选 bit mask、block 内 softmax 和 PV。它支持 FP16 Q/K/V，也支持 Q8_0
 * Query 与 Q4_0 或 Q8_0 cache 配对。返回的 block 输出和 LSE 足以让调用方合并多个 block，得到对逻辑
 * cache 的精确 softmax 结果。
 *
 * @param head_dim 每个 Query、Key 和 Value head 的特征宽度；必须能被 32 整除。
 * @param bsz 同时计算的 Query head 数，通常等于 GQA 分组大小 n_gqa。
 * @param q_type q 的存储类型，只支持 FP16 或 Q8_0。
 * @param q Query 矩阵，形状为 [bsz, head_dim]。
 * @param past_kv_len 当前 block 缓冲区中的 token 位置数，通常为 config_.block_len。
 * @param past_kv_offset 当前 block 的预留逻辑偏移；核心计算目前未使用。
 * @param is_full_attn 为 true 表示全部 token 位置有效，无需 mask。
 * @param attn_mask 可选的压缩 bit mask，用于标识未填满 block 中 past_kv_len 个位置的有效性。
 * @param k_type k_cache 的存储类型：FP16、Q4_0 或 Q8_0。
 * @param k_quant_type Key 的量化轴选择；当前要求为按 token 量化，即 0。
 * @param k_cache Key block，逻辑形状为 [past_kv_len, head_dim]。
 * @param num_k_anchor 在线 Key anchor 数；当前要求为 0。
 * @param k_cache_anchors 可选 Key anchor 数据；num_k_anchor 为 0 时不使用。
 * @param k_cache_anchor_pos 可选 Key anchor 位置；num_k_anchor 为 0 时不使用。
 * @param v_type v_cache 的存储类型：FP16、Q4_0 或 Q8_0。
 * @param v_quant_type Value 的量化轴选择；当前要求为按通道量化，即 1。
 * @param v_cache Value block，逻辑存储形状为 [head_dim, past_kv_len]。
 * @param num_v_anchor 在线 Value anchor 数；当前要求为 0。
 * @param v_cache_anchors 可选 Value anchor 数据；num_v_anchor 为 0 时不使用。
 * @param v_cache_anchor_pos 可选 Value anchor 位置；num_v_anchor 为 0 时不使用。
 * @param attn_score 调用方提供的 FP32 临时缓冲区，形状为 [bsz, past_kv_len]。
 * @param output block 输出，形状为 [bsz, head_dim]，表示类型由 q_type 决定。
 * @param lse FP32 block log-sum-exp 输出，形状为 [bsz]。
 * @param draft 调用方提供的临时工作区，大小要求见 kvcache.h 中的函数声明。
 * @param rotary_angle 可选的逐 token 旋转位置编号，用于在线旋转 Key。
 * @param rotary_cos rotary_angle 非空时使用的可选 FP16 RoPE cos 表。
 * @param rotary_sin rotary_angle 非空时使用的可选 FP16 RoPE sin 表。
 */
void KVCache::attn_with_kvcache_one_block_(int head_dim, int bsz,
                                           ggml_type q_type,  // GGML data type of `Q`, only supports fp16 and q8_0
                                           // [bsz, head_dim]
                                           // Quantization is always on the head_dim dimension (per_token). If
                                           // head_dim % 32 != 0, an error will be raised. The size must be bsz *
                                           // head_dim/32 * qtype_size.
                                           const void* q,

                                           int past_kv_len, int past_kv_offset,
                                           bool is_full_attn,  // true indicates a full 1 mask
                                           // If is_full_attn = false, a bit matrix representing the mask is
                                           // passed. [bsz, past_kv_len]
                                           const uint8_t* attn_mask,

                                           ggml_type k_type,  // GGML data type of `K Cache`, only supports fp16,
                                                              // q4_0, q8_0
                                           int k_quant_type,  // 0 for per_token, 1 for per_channel, others raise an
                                                              // error
                                           // [seq_len, head_dim]
                                           // If quant_type == 0, head_dim % 32 must be 0.
                                           // If quant_type == 1, seq_len % 32 must be 0.
                                           const void* k_cache,

                                           // k_anchor_type must be fp16
                                           int num_k_anchor,  // num_k_anchor == 0 indicates no anchor
                                           // [num_k_anchor, head_dim]
                                           const void* k_cache_anchors,
                                           // Each token is associated with the nearest previous position's anchor,
                                           // with the same distance.
                                           const int* k_cache_anchor_pos,

                                           // v_cache similar to k_cache
                                           ggml_type v_type, int v_quant_type,
                                           // [head_dim, seq_len]
                                           const void* v_cache, int num_v_anchor, const void* v_cache_anchors,
                                           const int* v_cache_anchor_pos,

                                           // Pre-allocated buffer for intermediate calculations [bsz,
                                           // past_kv_len]. No malloc is performed inside this function.
                                           float* attn_score,

                                           // Output: [bsz, head_dim], with the same type as q_type
                                           void* output,
                                           // [bsz]
                                           float* lse,

                                           // Pre-allocated temporary buffer with sufficient size:
                                           // (2 * bsz * past_kv_len + 6 * bsz * head_dim + 2 * past_kv_len *
                                           // head_dim + past_kv_len * head_dim / 32) bytes.
                                           void* draft,

                                           // Apply rotary embedding online
                                           const int* rotary_angle, const void* rotary_cos, const void* rotary_sin
                                           // rotary_cos=None,
                                           // rotary_sin=None,
                                           // cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
                                           // cache_batch_idx: Optional[torch.Tensor] = None,
                                           // rotary_interleaved=True,

                                           // // Not supported for now
                                           // window_size=(-1, -1),  # -1 means infinite context window
                                           // alibi_slopes=None,
) {
  assert(head_dim % 32 == 0);
  assert(k_quant_type == 0);
  assert(v_quant_type == 1);
  assert(q_type == GGML_TYPE_F16 || q_type == GGML_TYPE_Q8_0);
  if (q_type == GGML_TYPE_F16) {
    assert(k_type == GGML_TYPE_F16);
    assert(v_type == GGML_TYPE_F16);

    // attn = q * k + q * k_anchor
    // TODO: anchor
    assert(num_k_anchor == 0);

    if (rotary_angle != nullptr) {
      ggml_fp16_t* k_cache_with_rope_fp16 =
          (reinterpret_cast<ggml_fp16_t*>(draft) + sizeof(block_q8_0) * bsz * past_kv_len / QK8_0 +
           sizeof(float) * bsz * head_dim);
      // dequant k_cache and apply rope
      // k_rope(i) = k(i) * cos(i) - k(i+l) * sin(i)
      // k_rope(i+l) = k(i+l) * cos(i+l) + k(i) * sin(i)

      // k(i)cos(i) -> k_rope(i)
      // k(i)sin(i+l) -> k_rope(i+l)

      // k(i)cos(i) -> k_rope(i)
      // -k(i)sin(i-l) -> k_rope(i-l)

      std::vector<float> block_fp32(32);
      for (int k = 0; k < past_kv_len; k++) {
        int angle = rotary_angle[k];
        for (int l = 0; l < head_dim / 32; l++) {
          for (int m = 0; m < 32; m++) {
            float x = GGML_FP16_TO_FP32(((ggml_fp16_t*)k_cache)[k * head_dim + l * 32 + m]);
            float sin_val = GGML_FP16_TO_FP32(((ggml_fp16_t*)rotary_sin)[angle * head_dim + l * 32 + m]);
            float cos_val = GGML_FP16_TO_FP32(((ggml_fp16_t*)rotary_cos)[angle * head_dim + l * 32 + m]);

            if (l * 32 + m < head_dim / 2) {
              k_cache_with_rope_fp16[k * head_dim + l * 32 + m] = GGML_FP32_TO_FP16(x * cos_val);
              k_cache_with_rope_fp16[k * head_dim + l * 32 + m + head_dim / 2] = GGML_FP32_TO_FP16(-x * sin_val);
            } else {
              k_cache_with_rope_fp16[k * head_dim + l * 32 + m] =
                  GGML_FP32_TO_FP16(GGML_FP16_TO_FP32(k_cache_with_rope_fp16[k * head_dim + l * 32 + m]) + x * sin_val);
              k_cache_with_rope_fp16[k * head_dim + l * 32 + m - head_dim / 2] = GGML_FP32_TO_FP16(
                  GGML_FP16_TO_FP32(k_cache_with_rope_fp16[k * head_dim + l * 32 + m - head_dim / 2]) - x * cos_val);
            }
          }
        }
      }

      llamafile_sgemm(past_kv_len, bsz, head_dim, (ggml_fp16_t*)k_cache_with_rope_fp16, head_dim, (ggml_fp16_t*)q,
                      head_dim, attn_score, past_kv_len, 0, 1, GGML_TASK_TYPE_COMPUTE, k_type, GGML_TYPE_F16,
                      GGML_TYPE_F32, GGML_PREC_DEFAULT);
    } else {
      bool ok = llamafile_sgemm(past_kv_len, bsz, head_dim, (ggml_fp16_t*)k_cache, head_dim, (ggml_fp16_t*)q, head_dim,
                                attn_score, past_kv_len, 0, 1, GGML_TASK_TYPE_COMPUTE, k_type, GGML_TYPE_F16,
                                GGML_TYPE_F32, GGML_PREC_DEFAULT);

      if (!ok) {
        printf("llamafile_sgemm failed\n");
      }
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

    // output = attn * v + attn * v_anchor
    // std::vector<float> sum(bsz * head_dim);
    float* sum =
        reinterpret_cast<float*>(reinterpret_cast<char*>(draft) + sizeof(block_q8_0) * bsz * past_kv_len / QK8_0);

    // float* attn_score_fp16(bsz, past_kv_len)
    ggml_fp16_t* attn_score_fp16 = (reinterpret_cast<ggml_fp16_t*>(reinterpret_cast<char*>(draft) +
                                                                   sizeof(block_q8_0) * bsz * past_kv_len / QK8_0 +
                                                                   sizeof(float) * bsz * head_dim));

    for (int i = 0; i < bsz * past_kv_len; i++) {
      attn_score_fp16[i] = GGML_FP32_TO_FP16(attn_score[i]);
    }

    // TODO: anchor
    assert(num_v_anchor == 0);
    bool ok = llamafile_sgemm(head_dim, bsz, past_kv_len, (ggml_fp16_t*)v_cache, past_kv_len,
                              (ggml_fp16_t*)attn_score_fp16, past_kv_len, sum, head_dim, 0, 1, GGML_TASK_TYPE_COMPUTE,
                              v_type, GGML_TYPE_F16, GGML_TYPE_F32, GGML_PREC_DEFAULT);
    if (!ok) {
      printf("llamafile_sgemm failed\n");
    }

    // copy to output
    for (int i = 0; i < bsz; i++) {
      for (int j = 0; j < head_dim; j++) {
        ((float*)output)[i * head_dim + j] = sum[i * head_dim + j];
      }
    }
  } else {
    assert(k_type == GGML_TYPE_Q4_0 || k_type == GGML_TYPE_Q8_0);
    assert(v_type == GGML_TYPE_Q4_0 || v_type == GGML_TYPE_Q8_0);

    // attn = q * k + q * k_anchor
    // TODO: anchor
    assert(num_k_anchor == 0);

    if (rotary_angle != nullptr) {
      ggml_fp16_t* k_cache_with_rope_fp16 =
          (reinterpret_cast<ggml_fp16_t*>(draft) + sizeof(block_q8_0) * bsz * past_kv_len / QK8_0 +
           sizeof(float) * bsz * head_dim);
      block_q4_0* k_cache_with_rope_q4 =
          (reinterpret_cast<block_q4_0*>(draft) + sizeof(block_q8_0) * bsz * past_kv_len / QK8_0 +
           sizeof(float) * bsz * head_dim) +
          sizeof(ggml_fp16_t) * bsz * head_dim;
      // dequant k_cache and apply rope
      // k_rope(i) = k(i) * cos(i) - k(i+l) * sin(i)
      // k_rope(i+l) = k(i+l) * cos(i+l) + k(i) * sin(i)

      // k(i)cos(i) -> k_rope(i)
      // k(i)sin(i+l) -> k_rope(i+l)

      // k(i)cos(i) -> k_rope(i)
      // -k(i)sin(i-l) -> k_rope(i-l)

      std::vector<float> block_fp32(32);
      for (int k = 0; k < past_kv_len; k++) {
        int angle = rotary_angle[k];
        for (int l = 0; l < head_dim / 32; l++) {
          block_q4_0 block = ((block_q4_0*)k_cache)[k * head_dim / 32 + l];
          dequantize_row_q4_0(&block, block_fp32.data(), 32);
          for (int m = 0; m < 32; m++) {
            float sin_val = GGML_FP16_TO_FP32(((ggml_fp16_t*)rotary_sin)[angle * head_dim + l * 32 + m]);
            float cos_val = GGML_FP16_TO_FP32(((ggml_fp16_t*)rotary_cos)[angle * head_dim + l * 32 + m]);

            if (l * 32 + m < head_dim / 2) {
              k_cache_with_rope_fp16[k * head_dim + l * 32 + m] = GGML_FP32_TO_FP16(block_fp32[m] * cos_val);
              k_cache_with_rope_fp16[k * head_dim + l * 32 + m + head_dim / 2] =
                  GGML_FP32_TO_FP16(-block_fp32[m] * sin_val);
            } else {
              k_cache_with_rope_fp16[k * head_dim + l * 32 + m] += GGML_FP32_TO_FP16(block_fp32[m] * sin_val);
              k_cache_with_rope_fp16[k * head_dim + l * 32 + m - head_dim / 2] -=
                  GGML_FP32_TO_FP16(block_fp32[m] * cos_val);
            }
          }
        }
      }
      // quantize k_cache_with_rope_fp16
      for (int k = 0; k < past_kv_len; k++) {
        for (int l = 0; l < head_dim / 32; l++) {
          for (int m = 0; m < 32; m++) {
            block_fp32[m] = GGML_FP16_TO_FP32(k_cache_with_rope_fp16[k * head_dim + l * 32 + m]);
          }
          quantize_row_q4_0(block_fp32.data(), &k_cache_with_rope_q4[k * head_dim / 32 + l], 32);
        }
      }

      llamafile_sgemm(past_kv_len, bsz, head_dim / 32, (block_q4_0*)k_cache_with_rope_q4, head_dim / 32, (block_q8_0*)q,
                      head_dim / 32, attn_score, past_kv_len, 0, 1, GGML_TASK_TYPE_COMPUTE, k_type, GGML_TYPE_Q8_0,
                      GGML_TYPE_F32, GGML_PREC_DEFAULT);
    } else {
      llamafile_sgemm(past_kv_len, bsz, head_dim / 32, (block_q4_0*)k_cache, head_dim / 32, (block_q8_0*)q,
                      head_dim / 32, attn_score, past_kv_len, 0, 1, GGML_TASK_TYPE_COMPUTE, k_type, GGML_TYPE_Q8_0,
                      GGML_TYPE_F32, GGML_PREC_DEFAULT);
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

    // output = attn * v + attn * v_anchor
    // std::vector<block_q8_0> attn_q8_0(bsz * past_kv_len / QK8_0);
    block_q8_0* attn_q8_0 = reinterpret_cast<block_q8_0*>(draft);
    quantize_row_q8_0(attn_score, attn_q8_0, bsz * past_kv_len);
    // std::vector<float> sum(bsz * head_dim);
    float* sum =
        reinterpret_cast<float*>(reinterpret_cast<char*>(draft) + sizeof(block_q8_0) * bsz * past_kv_len / QK8_0);
    // TODO: anchor
    assert(num_v_anchor == 0);
    llamafile_sgemm(head_dim, bsz, past_kv_len / 32, (block_q4_0*)v_cache, past_kv_len / 32, attn_q8_0,
                    past_kv_len / 32, sum, head_dim, 0, 1, GGML_TASK_TYPE_COMPUTE, v_type, GGML_TYPE_Q8_0,
                    GGML_TYPE_F32, GGML_PREC_DEFAULT);

    quantize_row_q8_0(sum, (block_q8_0*)output, bsz * head_dim);
  }
}
