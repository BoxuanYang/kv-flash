#include "dense_kvcache.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "ggml-impl.h"
#include "llamafile/sgemm.h"

namespace dense {

void KVCache::set_parallel_reduce(bool enabled) {
  parallel_reduce_ = enabled;
}

// 两种 reduce 路径均以同一 (batch, KV head) 的最多四个连续逻辑 block 为 task。
void KVCache::attention_kvhead_(const ggml_fp16_t* q_in, ggml_fp16_t* output,
                              float* attn_lse, int batch_size, WorkerPool* backend) {
  const int head_dim = config_.head_dim;
  const int block_len = config_.block_len;
  const int output_size = n_gqa_ * head_dim;
  std::fill(thread_local_failed_.begin(), thread_local_failed_.end(), 0);

  // 两个提交位置共用代码；锁的对象、粒度和提交时机保持不变。
  auto flush_thread_result = [&](int thread_id) {
    const auto [batch_id, head_id] = thread_cur_head_idx_[thread_id];
    if (batch_id == -1) return;
    auto& dst = output_fp32_[batch_id][head_id];
    auto& dst_lse = attn_lse_[batch_id][head_id];
    auto& src = thread_local_cur_output_fp32_[thread_id];
    auto& src_lse = thread_local_cur_attn_lse_[thread_id];
    std::lock_guard<std::mutex> lock(*mutex_[batch_id][head_id]);
    // LSE 可以合法地等于 0，使用独立有效标志判断首次提交。
    if (!output_valid_[batch_id][head_id]) {
      std::copy(src.begin(), src.end(), dst.begin());
      std::copy(src_lse.begin(), src_lse.end(), dst_lse.begin());
      output_valid_[batch_id][head_id] = 1;
    } else {
      for (int i = 0; i < n_gqa_; ++i) {
        const float hi = std::max(dst_lse[i], src_lse[i]);
        const float lo = std::min(dst_lse[i], src_lse[i]);
        const float merged_lse = hi + std::log(1.0 + std::exp(lo - hi));
        ggml_vec_scale_f32(head_dim, dst.data() + i * head_dim, std::exp(dst_lse[i] - merged_lse));
        ggml_vec_scale_f32(head_dim, src.data() + i * head_dim, std::exp(src_lse[i] - merged_lse));
        for (int j = 0; j < head_dim; ++j) dst[i * head_dim + j] += src[i * head_dim + j];
        dst_lse[i] = merged_lse;
      }
    }
  };

  // 空 batch / 全空 KV 不提交任务，避免线程池处理 task_num=0。
  if (task_offsets_[batch_size] > 0) {
    backend->do_work_stealing_job(
        task_offsets_[batch_size],
        [&](int thread_id) {
          thread_cur_head_idx_[thread_id] = {-1, -1};
        },
        [&](int task_id) {
          const int thread_id = WorkerPool::thread_local_id;
          // 重复前缀对应空序列，由 upper_bound 自动跳过。
          const int batch_id = int(std::upper_bound(task_offsets_.begin(),
              task_offsets_.begin() + batch_size + 1, task_id) - task_offsets_.begin()) - 1;
          const int chunks = (task_offsets_[batch_id + 1] - task_offsets_[batch_id]) / config_.kv_head_num;
          const int local_id = task_id - task_offsets_[batch_id];
          const int head_id = local_id / chunks;
          const int chunk_id = local_id % chunks;
          const int len = cache_seqlens_[batch_id];
          const int blocks = len / block_len + (len % block_len != 0);
          const int block_begin = chunk_id * kBlocksPerTask;
          const int block_end = block_begin + std::min(kBlocksPerTask, blocks - block_begin);
          for (int block_id = block_begin; block_id < block_end; ++block_id) {
            // 有效 block 数与 block 表行步长是两个不同的量。
            const int block_idx = block_table_[size_t(batch_id) * block_num_per_seq_ + block_id];
            if (block_idx < 0 || block_idx >= config_.max_block_num) {
              thread_local_failed_[thread_id] = 1;
              return;
            }
            const int valid_tokens = std::min(block_len, cache_seqlens_[batch_id] - block_id * block_len);
            auto& block_output = thread_local_output_fp32_[thread_id];
            auto& block_lse = thread_local_attn_lse_[thread_id];
            float* block_output_data = block_output.data();
            float* block_lse_data = block_lse.data();
            const bool ok = attn_with_kvcache_one_block_(
                head_dim, n_gqa_, q_in + (size_t(batch_id) * config_.kv_head_num + head_id) * output_size,
                block_len, valid_tokens,
                k_cache_fp16_[layer_id_][head_id][block_idx].data(),
                v_cache_fp16_[layer_id_][head_id][block_idx].data(),
                thread_local_attn_score_[thread_id].data(), block_output_data, block_lse_data,
                thread_local_probability_fp16_[thread_id].data());
            if (!ok) {
              // 不从工作线程抛异常；所有线程完成后，由调用线程报告错误。
              thread_local_failed_[thread_id] = 1;
              return;
            }

            const auto [cur_batch_id, cur_head_id] = thread_cur_head_idx_[thread_id];
            auto& cur_output = thread_local_cur_output_fp32_[thread_id];
            auto& cur_lse = thread_local_cur_attn_lse_[thread_id];
            if (parallel_reduce_ ? block_id != block_begin
                                : batch_id == cur_batch_id && head_id == cur_head_id) {
              // two_phase 每个 task 独立累计；locked 还可累计同一 head 的后续 task。
              for (int i = 0; i < n_gqa_; ++i) {
                const float hi = std::max(cur_lse[i], block_lse[i]);
                const float lo = std::min(cur_lse[i], block_lse[i]);
                const float merged_lse = hi + std::log(1.0 + std::exp(lo - hi));
                ggml_vec_scale_f32(head_dim, cur_output.data() + i * head_dim, std::exp(cur_lse[i] - merged_lse));
                ggml_vec_scale_f32(head_dim, block_output.data() + i * head_dim, std::exp(block_lse[i] - merged_lse));
                for (int j = 0; j < head_dim; ++j) cur_output[i * head_dim + j] += block_output[i * head_dim + j];
                cur_lse[i] = merged_lse;
              }
            } else {
              if (!parallel_reduce_) flush_thread_result(thread_id);
              thread_cur_head_idx_[thread_id] = {batch_id, head_id};
              std::copy(block_output.begin(), block_output.end(), cur_output.begin());
              std::copy(block_lse.begin(), block_lse.end(), cur_lse.begin());
            }
          }
          if (parallel_reduce_) {
            // task 内先归并，只有完整成功的 task 才提交一份独占结果。
            std::copy_n(thread_local_cur_output_fp32_[thread_id].data(), output_size,
                        reduce_task_output_.data() + size_t(task_id) * output_size);
            std::copy_n(thread_local_cur_attn_lse_[thread_id].data(), n_gqa_,
                        reduce_task_lse_.data() + size_t(task_id) * 32);
          }
        },
        [&](int thread_id) {
          if (!parallel_reduce_) flush_thread_result(thread_id);
        });
  }
  if (std::find(thread_local_failed_.begin(), thread_local_failed_.end(), 1) != thread_local_failed_.end()) {
    throw std::runtime_error("Dense attention failed: invalid physical block or unsupported FP16 GEMM");
  }

  // 上面的同步任务池调用已经等所有 block 完成；失败时不会读取未完成的块结果。
  if (parallel_reduce_ && task_offsets_[batch_size] > 0) {
    backend->do_work_stealing_job(batch_size * config_.q_head_num,
        [&](int task_id) { reduce_one_query_head_(task_id); });
  }

  // 空序列输出为 0、LSE 为 -inf。输出在此转为 FP16，LSE 始终为 FP32。
  for (int b = 0; b < batch_size; ++b) {
    for (int h = 0; h < config_.kv_head_num; ++h) {
      const size_t group = size_t(b) * config_.kv_head_num + h;
      for (int j = 0; j < output_size; ++j) {
        output[group * output_size + j] = GGML_FP32_TO_FP16(output_fp32_[b][h][j]);
      }
      if (attn_lse) std::copy(attn_lse_[b][h].begin(), attn_lse_[b][h].end(), attn_lse + group * n_gqa_);
    }
  }
}

// 每个 reduce task 独占一个 query head 的输出，动态领取；不进行 QK/PV 计算。
void KVCache::reduce_one_query_head_(int task_id) {
  int batch_id = task_id / config_.q_head_num;
  int query_head = task_id % config_.q_head_num;
  int head_id = query_head / n_gqa_;
  int group_id = query_head % n_gqa_;
  int chunks = (task_offsets_[batch_id + 1] - task_offsets_[batch_id]) / config_.kv_head_num;
  if (chunks == 0) return;  // 初始化已将空序列输出设为 0、LSE 设为 -inf。
  int first = task_offsets_[batch_id] + head_id * chunks;
  int head_dim = config_.head_dim;
  int output_size = n_gqa_ * head_dim;
  float* dst = output_fp32_[batch_id][head_id].data() + group_id * head_dim;

  // 用各 task LSE 的稳定 softmax 作为 task 输出权重。只读暂存结果，不修改其他 task 的数据。
  float max_lse = reduce_task_lse_[size_t(first) * 32 + group_id];
  for (int b = 1; b < chunks; ++b) {
    float value = reduce_task_lse_[size_t(first + b) * 32 + group_id];
    if (value > max_lse) max_lse = value;
  }
  float sum = 0.0f;
  for (int b = 0; b < chunks; ++b) {
    float weight = std::exp(reduce_task_lse_[size_t(first + b) * 32 + group_id] - max_lse);
    const float* src = reduce_task_output_.data() + size_t(first + b) * output_size + group_id * head_dim;
    sum += weight;
    for (int d = 0; d < head_dim; ++d) dst[d] += weight * src[d];
  }
  for (int d = 0; d < head_dim; ++d) dst[d] /= sum;
  attn_lse_[batch_id][head_id][group_id] = max_lse + std::log(sum);
}

// 在原有 batch 循环内构建任务前缀和，不分配临时任务对象。
void KVCache::attn_initialize_kvhead_(int batch_size, int layer_idx, const int* block_table,
                                    int block_table_stride, const int* cache_seqlens) {
  layer_id_ = layer_idx;
  block_table_ = block_table;
  block_num_per_seq_ = block_table_stride;
  task_offsets_[0] = 0;
  for (int b = 0; b < batch_size; ++b) {
    const int len = cache_seqlens[b];
    if (len < 0) throw std::invalid_argument("cache_seqlens must be nonnegative");
    const int blocks = len / config_.block_len + (len % config_.block_len != 0);
    if (blocks > block_table_stride) throw std::invalid_argument("block table is too short");
    const int chunks = blocks / kBlocksPerTask + (blocks % kBlocksPerTask != 0);
    const int64_t tasks = int64_t(task_offsets_[b]) + int64_t(chunks) * config_.kv_head_num;
    if (tasks > std::numeric_limits<int>::max()) throw std::overflow_error("too many attention tasks");
    task_offsets_[b + 1] = int(tasks);
    cache_seqlens_[b] = len;
    for (int h = 0; h < config_.kv_head_num; ++h) {
      output_valid_[b][h] = 0;
      std::fill(output_fp32_[b][h].begin(), output_fp32_[b][h].end(), 0.0f);
      std::fill(attn_lse_[b][h].begin(), attn_lse_[b][h].end(), -std::numeric_limits<float>::infinity());
    }
  }
  if (parallel_reduce_) {
    size_t output_count = size_t(task_offsets_[batch_size]) * n_gqa_ * config_.head_dim;
    size_t lse_count = size_t(task_offsets_[batch_size]) * 32;
    if (reduce_task_output_.size() < output_count) reduce_task_output_.resize(output_count);
    if (reduce_task_lse_.size() < lse_count) reduce_task_lse_.resize(lse_count);
  }
}

// 公开签名保留兼容性；generate_token_idx 不参与计算，q_len 只允许为 1。
void KVCache::attn(const ggml_fp16_t* q_in, ggml_fp16_t* output, float* attn_lse,
                   int layer_idx, int generate_token_idx, int q_len, int batch_size,
                   int max_block_num, int* block_table, int* cache_seqlens, WorkerPool* backend) {
  (void)generate_token_idx;
  if (q_len != 1) throw std::invalid_argument("Dense attention supports decode q_len=1 only");
  if (!backend || (backend->config.subpool_thread_count.empty() || backend->config.subpool_thread_count[0] <= 0 ||
      backend->config.subpool_thread_count[0] > config_.max_thread_num) || batch_size < 0 ||
      batch_size > config_.max_batch_size || layer_idx < 0 || layer_idx >= config_.layer_num ||
      max_block_num < 0) throw std::invalid_argument("invalid dense attention dimensions or capacity");
  if (batch_size > 0 && (!q_in || !output || !cache_seqlens || !block_table)) {
    throw std::invalid_argument("null dense attention input/output");
  }
  attn_initialize_kvhead_(batch_size, layer_idx, block_table, max_block_num, cache_seqlens);
  attention_kvhead_(q_in, output, attn_lse, batch_size, backend);
}

// cache_seqlens 为写入前长度，本函数每次增加 1；跨层调用需由调用方提供各层正确的长度。
void KVCache::attn_with_kvcache(
    const ggml_fp16_t* q_in, const ggml_fp16_t* k_in, const ggml_fp16_t* v_in,
    ggml_fp16_t* output, float* attn_lse, int layer_idx, int generate_token_idx,
    int q_len, int batch_size, int max_block_num, int* block_table,
    int* cache_seqlens, WorkerPool* backend) {
  if (q_len != 1) throw std::invalid_argument("Dense attention supports decode q_len=1 only");
  if (batch_size > 0 && (!q_in || !output)) throw std::invalid_argument("null query/output");
  update_kvcache_fp16(k_in, v_in, layer_idx, block_table, batch_size,
                      max_block_num, cache_seqlens, 1, backend);
  for (int b = 0; b < batch_size; ++b) ++cache_seqlens[b];
  attn(q_in, output, attn_lse, layer_idx, generate_token_idx, 1, batch_size,
       max_block_num, block_table, cache_seqlens, backend);
}

// Q [G,D], K [T,D], V [D,T] 均为 FP16，Q/K 已完成 Norm 和 RoPE。
// score/probability [G,T]，output [G,D] 为单块结果，不能与线程累计结果共用。
// 尾块只改变有效范围，两个 GEMM 仍使用完整 block_len 及原来的行步长。
bool KVCache::attn_with_kvcache_one_block_(
    int head_dim, int n_gqa, const ggml_fp16_t* q, int block_len, int valid_tokens,
    const ggml_fp16_t* k_cache, const ggml_fp16_t* v_cache, float* attn_score,
    float* output, float* lse, ggml_fp16_t* probability) {
  if (!llamafile_sgemm(block_len, n_gqa, head_dim, k_cache, head_dim, q, head_dim,
                       attn_score, block_len, 0, 1, GGML_TASK_TYPE_COMPUTE,
                       GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_PREC_DEFAULT)) return false;
  ggml_vec_scale_f32(n_gqa * block_len, attn_score, 1.0f / std::sqrt(float(head_dim)));
  for (int i = 0; i < n_gqa; ++i) {
    float* row = attn_score + i * block_len;
    // 只归一化有效前缀，避免 exp(score) 溢出；尾部 probability 明确置零。
    const float max_score = *std::max_element(row, row + valid_tokens);
    float sum_exp = 0.0f;
    for (int j = 0; j < valid_tokens; ++j) {
      row[j] = std::exp(row[j] - max_score);
      sum_exp += row[j];
    }
    for (int j = 0; j < valid_tokens; ++j) row[j] /= sum_exp;
    std::fill(row + valid_tokens, row + block_len, 0.0f);
    lse[i] = max_score + std::log(sum_exp);
  }
  for (int i = 0; i < n_gqa * block_len; ++i) probability[i] = GGML_FP32_TO_FP16(attn_score[i]);

  // PV 直接写当前线程的单块输出，省去 draft 中的 FP32 sum 及其复制。
  return llamafile_sgemm(head_dim, n_gqa, block_len, v_cache, block_len,
                         probability, block_len, output, head_dim, 0, 1,
                         GGML_TASK_TYPE_COMPUTE, GGML_TYPE_F16, GGML_TYPE_F16,
                         GGML_TYPE_F32, GGML_PREC_DEFAULT);
}

}  // namespace dense
