#include "dense_kvcache.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include "ggml-impl.h"
#include "llamafile/sgemm.h"

namespace dense {

// chrono 只在这里使用；其余计时代码都是 double，单位 ms。
static double profile_now_ms() {
  return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

void KVCache::profile_reset(int threads) {
  if ((threads != 32 && threads != 64) || threads > config_.max_thread_num) {
    throw std::invalid_argument("profiling requires 32 or 64 available threads");
  }
  profile_enabled_ = false;
  profile_threads_ = threads;
  profile_calls_ = 0;
  profile_wall_time_ = 0.0;
  profile_init_time_ = 0.0;
  profile_pool_time_ = 0.0;
  profile_output_time_ = 0.0;
  for (int t = 0; t < threads; ++t) {
    profile_task_count_[t][0] = 0;
    profile_flush_count_[t][0] = 0;
    profile_qk_time_[t][0] = 0.0;
    profile_softmax_time_[t][0] = 0.0;
    profile_convert_time_[t][0] = 0.0;
    profile_pv_time_[t][0] = 0.0;
    profile_merge_time_[t][0] = 0.0;
    profile_sync_time_[t][0] = 0.0;
    profile_writeback_time_[t][0] = 0.0;
    profile_task_time_[t][0] = 0.0;
    profile_final_flush_time_[t][0] = 0.0;
    profile_gap_time_[t][0] = 0.0;
    profile_last_end_[t][0] = 0.0;
  }
}

void KVCache::profile_enable(bool enabled) {
  if (enabled && profile_threads_ == 0) {
    throw std::invalid_argument("call profile_reset before profile_enable");
  }
  profile_enabled_ = enabled;
}

void KVCache::profile_get_times_(int t, double* times) {
  times[0] = profile_qk_time_[t][0];
  times[1] = profile_softmax_time_[t][0];
  times[2] = profile_convert_time_[t][0];
  times[3] = profile_pv_time_[t][0];
  times[4] = profile_merge_time_[t][0];
  times[5] = profile_sync_time_[t][0];
  times[6] = profile_writeback_time_[t][0];
  // other 是 task 和最后一次 flush 中尚未归入上述阶段的剩余时间。
  times[7] = profile_task_time_[t][0] + profile_final_flush_time_[t][0];
  for (int i = 0; i < 7; ++i) times[7] -= times[i];
  times[8] = profile_gap_time_[t][0];
}

void KVCache::profile_write(const char* path, bool append) {
  profile_enabled_ = false;
  long long tasks = 0;
  long long flushes = 0;
  double sums[9] = {0.0};
  double max_avg[9] = {0.0};
  const char* names[9] = {"qk", "softmax", "convert", "pv", "merge",
                          "sync", "writeback", "other", "gap"};
  for (int t = 0; t < profile_threads_; ++t) {
    double times[9];
    profile_get_times_(t, times);
    long long count = profile_task_count_[t][0];
    tasks += count;
    flushes += profile_flush_count_[t][0];
    for (int i = 0; i < 9; ++i) {
      sums[i] += times[i];
      if (count > 0 && times[i] / count > max_avg[i]) max_avg[i] = times[i] / count;
    }
  }
  if (profile_calls_ == 0 || tasks != profile_calls_ * 64 * 32 * config_.kv_head_num ||
      std::find(thread_local_failed_.begin(), thread_local_failed_.end(), 1) != thread_local_failed_.end()) {
    throw std::runtime_error("profile incomplete: no successful calls or task count mismatch");
  }
  FILE* file = fopen(path, append ? "a" : "w");
  if (!file) throw std::runtime_error("cannot open cpu profile report");
  if (!append) {
    fprintf(file,
        "CPU decode attention profile | 时间单位全部为 ms\n"
        "task = 一个 (batch, KV head, block)；tasks = 该线程实际完成的 task 次数。\n"
        "total_ms = 累计耗时；avg_ms/task = 累计耗时 / 该线程 tasks。无 task 显示 N/A。\n"
        "汇总 avg_ms/task = 全线程累计耗时 / 全线程 tasks；max_avg = 各线程平均值的最大值。\n"
        "qk: QK GEMM。softmax: 缩放、稳定 softmax、LSE 和尾块清零。\n"
        "convert: 概率矩阵 FP32 转 FP16。pv: PV GEMM。\n"
        "merge: 线程本地输出/LSE 合并，或切换 head 后复制新结果。\n"
        "sync: 输出写回前获取 mutex 的等待时间，包含 lock 调用开销。\n"
        "writeback: 获得 mutex 后的全局输出/LSE 合并与写回，包含解锁。\n"
        "flushes: 实际加锁写回次数；sync/writeback 包含线程结束时的最后一次写回，仍除以 tasks。\n"
        "other: task 与最后一次 flush 的剩余时间，含任务索引、检查和部分计时开销。\n"
        "gap: init 完成到首 task、相邻 task 间、末 task 到 finalize 的间隙；包含取任务和调度延迟。\n"
        "share_pct: 阶段累计耗时 / 所有阶段累计耗时；并行线程时间之和不是 attention 延迟。\n"
        "wall/init/pool/output: 每次 attention 的墙钟平均值，分母为 calls，与每 task 平均值分开。\n"
        "wall: 初始化到 FP16 输出完成；init: 清空输出和任务前缀；pool: 派发到所有线程结束；\n"
        "output: 工作完成检查、输出 FP16 转换和 LSE 复制。wall 不含输入验证、Python 和文件写入。\n"
        "使用 steady_clock；预热不计入。计时会扰动执行；各阶段包含线程被系统抢占的时间。\n\n");
  }
  fprintf(file, "B=64  Seq=4096  Block=128  Threads=%d  Hq=%d  Hkv=%d  D=%d\n",
          profile_threads_, config_.q_head_num, config_.kv_head_num, config_.head_dim);
  fprintf(file, "calls=%lld  tasks=%lld  flushes=%lld\n", profile_calls_, tasks, flushes);
  fprintf(file, "wall_avg_ms=%.6f  init_avg_ms=%.6f  pool_avg_ms=%.6f  output_avg_ms=%.6f\n\n",
          profile_wall_time_ / profile_calls_, profile_init_time_ / profile_calls_,
          profile_pool_time_ / profile_calls_, profile_output_time_ / profile_calls_);

  double total = 0.0;
  for (int i = 0; i < 9; ++i) total += sums[i];
  fprintf(file, "%-10s %14s %14s %14s %11s\n", "stage", "total_ms", "avg_ms/task", "max_avg", "share_pct");
  for (int i = 0; i < 9; ++i) {
    fprintf(file, "%-10s %14.6f %14.6f %14.6f %11.2f\n", names[i], sums[i],
            sums[i] / tasks, max_avg[i], total > 0.0 ? sums[i] * 100.0 / total : 0.0);
  }

  // 两张表使用相同列序：第一张累计时间，第二张每 task 平均时间。
  for (int average = 0; average < 2; ++average) {
    fprintf(file, "\n%s\n", average ? "每线程 avg_ms/task" : "每线程 total_ms");
    fprintf(file, "%6s %10s %10s", "thread", "tasks", "flushes");
    for (int i = 0; i < 9; ++i) fprintf(file, " %12s", names[i]);
    fprintf(file, "\n");
    for (int t = 0; t < profile_threads_; ++t) {
      double times[9];
      profile_get_times_(t, times);
      long long count = profile_task_count_[t][0];
      fprintf(file, "%6d %10lld %10lld", t, count, profile_flush_count_[t][0]);
      for (int i = 0; i < 9; ++i) {
        if (average && count == 0) fprintf(file, " %12s", "N/A");
        else fprintf(file, " %12.6f", average ? times[i] / count : times[i]);
      }
      fprintf(file, "\n");
    }
  }
  fprintf(file, "\n--------------------------------------------------------------------------\n\n");
  int failed = ferror(file);
  if (fclose(file) != 0) failed = 1;
  if (failed) throw std::runtime_error("cannot write cpu profile report");
}

// 一个任务仍处理一个 (batch, KV head, block)。先线程内累计，切换 head 或结束时再带锁提交。
void KVCache::attention_kvhead_(const ggml_fp16_t* q_in, ggml_fp16_t* output,
                              float* attn_lse, int batch_size, WorkerPool* backend) {
  const int head_dim = config_.head_dim;
  const int block_len = config_.block_len;
  const int output_size = n_gqa_ * head_dim;
  double pool_start = 0.0;
  if (profile_enabled_) pool_start = profile_now_ms();
  std::fill(thread_local_failed_.begin(), thread_local_failed_.end(), 0);

  // 两个提交位置共用代码；锁的对象、粒度和提交时机保持不变。
  auto flush_thread_result = [&](int thread_id) {
    const auto [batch_id, head_id] = thread_cur_head_idx_[thread_id];
    if (batch_id == -1) return;
    auto& dst = output_fp32_[batch_id][head_id];
    auto& dst_lse = attn_lse_[batch_id][head_id];
    auto& src = thread_local_cur_output_fp32_[thread_id];
    auto& src_lse = thread_local_cur_attn_lse_[thread_id];
    double wait_start = 0.0;
    double lock_acquired = 0.0;
    if (profile_enabled_) wait_start = profile_now_ms();
    {
      std::lock_guard<std::mutex> lock(*mutex_[batch_id][head_id]);
      if (profile_enabled_) lock_acquired = profile_now_ms();
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
    }  // 释放原来的锁后再累计，避免把统计数组写入放进临界区。
    if (profile_enabled_) {
      double end = profile_now_ms();
      profile_sync_time_[thread_id][0] += lock_acquired - wait_start;
      profile_writeback_time_[thread_id][0] += end - lock_acquired;
      ++profile_flush_count_[thread_id][0];
    }
  };

  // 空 batch / 全空 KV 不提交任务，避免线程池处理 task_num=0。
  if (task_offsets_[batch_size] > 0) {
    backend->do_work_stealing_job(
        task_offsets_[batch_size],
        [&](int thread_id) {
          thread_cur_head_idx_[thread_id] = {-1, -1};
          if (profile_enabled_) profile_last_end_[thread_id][0] = profile_now_ms();
        },
        [&](int task_id) {
          const int thread_id = WorkerPool::thread_local_id;
          double task_start = 0.0;
          if (profile_enabled_) {
            task_start = profile_now_ms();
            profile_gap_time_[thread_id][0] += task_start - profile_last_end_[thread_id][0];
          }
          // 重复前缀对应空序列，由 upper_bound 自动跳过。
          const int batch_id = int(std::upper_bound(task_offsets_.begin(),
              task_offsets_.begin() + batch_size + 1, task_id) - task_offsets_.begin()) - 1;
          const int blocks = (task_offsets_[batch_id + 1] - task_offsets_[batch_id]) / config_.kv_head_num;
          const int local_id = task_id - task_offsets_[batch_id];
          const int head_id = local_id / blocks;
          const int block_id = local_id % blocks;
          // 有效 block 数与 block 表行步长是两个不同的量。
          const int block_idx = block_table_[size_t(batch_id) * block_num_per_seq_ + block_id];
          if (block_idx < 0 || block_idx >= config_.max_block_num) {
            thread_local_failed_[thread_id] = 1;
            return;
          }
          const int valid_tokens = std::min(block_len, cache_seqlens_[batch_id] - block_id * block_len);
          auto& block_output = thread_local_output_fp32_[thread_id];
          auto& block_lse = thread_local_attn_lse_[thread_id];
          const bool ok = attn_with_kvcache_one_block_(
              head_dim, n_gqa_, q_in + (size_t(batch_id) * config_.kv_head_num + head_id) * output_size,
              block_len, valid_tokens,
              k_cache_fp16_[layer_id_][head_id][block_idx].data(),
              v_cache_fp16_[layer_id_][head_id][block_idx].data(),
              thread_local_attn_score_[thread_id].data(), block_output.data(), block_lse.data(),
              thread_local_probability_fp16_[thread_id].data());
          if (!ok) {
            // 不从工作线程抛异常；所有线程完成后，由调用线程报告错误。
            thread_local_failed_[thread_id] = 1;
            return;
          }

          const auto [cur_batch_id, cur_head_id] = thread_cur_head_idx_[thread_id];
          double merge_start = 0.0;
          if (profile_enabled_) merge_start = profile_now_ms();
          auto& cur_output = thread_local_cur_output_fp32_[thread_id];
          auto& cur_lse = thread_local_cur_attn_lse_[thread_id];
          if (batch_id == cur_batch_id && head_id == cur_head_id) {
            // 保留线程内归并，只修正 LSE 公式的溢出问题。
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
            flush_thread_result(thread_id);
            // flush 已单独计入 sync/writeback，不重复计入本地 merge。
            if (profile_enabled_) merge_start = profile_now_ms();
            thread_cur_head_idx_[thread_id] = {batch_id, head_id};
            std::copy(block_output.begin(), block_output.end(), cur_output.begin());
            std::copy(block_lse.begin(), block_lse.end(), cur_lse.begin());
          }
          if (profile_enabled_) {
            double end = profile_now_ms();
            profile_merge_time_[thread_id][0] += end - merge_start;
            profile_task_time_[thread_id][0] += end - task_start;
            ++profile_task_count_[thread_id][0];
            profile_last_end_[thread_id][0] = end;
          }
        },
        [&](int thread_id) {
          double start = 0.0;
          if (profile_enabled_) {
            start = profile_now_ms();
            profile_gap_time_[thread_id][0] += start - profile_last_end_[thread_id][0];
          }
          flush_thread_result(thread_id);
          if (profile_enabled_) {
            profile_final_flush_time_[thread_id][0] += profile_now_ms() - start;
          }
        });
  }
  double output_start = 0.0;
  if (profile_enabled_) {
    output_start = profile_now_ms();
    profile_pool_time_ += output_start - pool_start;
  }
  if (std::find(thread_local_failed_.begin(), thread_local_failed_.end(), 1) != thread_local_failed_.end()) {
    throw std::runtime_error("Dense attention failed: invalid physical block or unsupported FP16 GEMM");
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
  if (profile_enabled_) profile_output_time_ += profile_now_ms() - output_start;
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
    const int64_t tasks = int64_t(task_offsets_[b]) + int64_t(blocks) * config_.kv_head_num;
    if (tasks > std::numeric_limits<int>::max()) throw std::overflow_error("too many attention tasks");
    task_offsets_[b + 1] = int(tasks);
    cache_seqlens_[b] = len;
    for (int h = 0; h < config_.kv_head_num; ++h) {
      output_valid_[b][h] = 0;
      std::fill(output_fp32_[b][h].begin(), output_fp32_[b][h].end(), 0.0f);
      std::fill(attn_lse_[b][h].begin(), attn_lse_[b][h].end(), -std::numeric_limits<float>::infinity());
    }
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
  double start = 0.0;
  if (profile_enabled_) {
    if (batch_size != 64 || config_.block_len != 128 ||
        backend->config.subpool_thread_count[0] != profile_threads_) {
      throw std::invalid_argument("profile shape must be B=64, block=128, threads=32/64");
    }
    for (int b = 0; b < batch_size; ++b) {
      if (cache_seqlens[b] != 4096) throw std::invalid_argument("profile requires seq_len=4096");
    }
    start = profile_now_ms();
  }
  attn_initialize_kvhead_(batch_size, layer_idx, block_table, max_block_num, cache_seqlens);
  if (profile_enabled_) profile_init_time_ += profile_now_ms() - start;
  attention_kvhead_(q_in, output, attn_lse, batch_size, backend);
  if (profile_enabled_) {
    profile_wall_time_ += profile_now_ms() - start;
    ++profile_calls_;
  }
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
  int thread_id = WorkerPool::thread_local_id;
  double start = 0.0;
  if (profile_enabled_) start = profile_now_ms();
  if (!llamafile_sgemm(block_len, n_gqa, head_dim, k_cache, head_dim, q, head_dim,
                       attn_score, block_len, 0, 1, GGML_TASK_TYPE_COMPUTE,
                       GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_PREC_DEFAULT)) return false;
  if (profile_enabled_) {
    double end = profile_now_ms();
    profile_qk_time_[thread_id][0] += end - start;
    start = end;
  }
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
  if (profile_enabled_) {
    double end = profile_now_ms();
    profile_softmax_time_[thread_id][0] += end - start;
    start = end;
  }
  for (int i = 0; i < n_gqa * block_len; ++i) probability[i] = GGML_FP32_TO_FP16(attn_score[i]);
  if (profile_enabled_) {
    double end = profile_now_ms();
    profile_convert_time_[thread_id][0] += end - start;
    start = end;
  }

  // PV 直接写当前线程的单块输出，省去 draft 中的 FP32 sum 及其复制。
  bool ok = llamafile_sgemm(head_dim, n_gqa, block_len, v_cache, block_len,
                         probability, block_len, output, head_dim, 0, 1,
                         GGML_TASK_TYPE_COMPUTE, GGML_TYPE_F16, GGML_TYPE_F16,
                         GGML_TYPE_F32, GGML_PREC_DEFAULT);
  if (profile_enabled_) profile_pv_time_[thread_id][0] += profile_now_ms() - start;
  return ok;
}

}  // namespace dense
