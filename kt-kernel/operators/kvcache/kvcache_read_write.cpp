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

#include "ggml-impl.h"
#include "kvcache.h"

/**
 * @brief 准备读取一个物理 block 的 anchor 数据。
 *
 * 当前实现仅把层号、block 编号、block 长度和目标缓冲区保存到成员变量；实际从 anchor_ 复制数据的
 * 并行代码尚未启用，因此调用后不会填充 anchor 指向的内容。
 *
 * @param anchor 预期接收 FP16 anchor 的输出缓冲区，当前仅记录其地址。
 * @param layer_id 要读取的模型层编号。
 * @param block_idx 要读取的物理 block 编号。
 * @param backend 预留的工作线程池；当前实现未使用。
 */
void KVCache::get_anchor_one_block(ggml_fp16_t* anchor, int layer_id, int block_idx, WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();

  layer_id_ = layer_id;
  block_idx = block_idx;
  seq_len_ = config_.block_len;
  anchor_data_ = const_cast<uint16_t*>(anchor);

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  printf("layer %d block %d time of reading anchor: %f s\n", layer_id, block_idx, duration.count());
}

/**
 * @brief 准备更新一个物理 block 的 anchor 数据。
 *
 * 当前实现仅暂存层号、block 编号、block 长度和输入缓冲区；实际写入 anchor_ 的并行代码被注释掉，
 * 因此调用不会修改 cache 中的 anchor。
 *
 * @param anchor 包含新 FP16 anchor 的输入缓冲区，当前仅记录其地址。
 * @param layer_id 要更新的模型层编号。
 * @param block_idx 要更新的物理 block 编号。
 * @param backend 预留的工作线程池；当前实现未使用。
 */
void KVCache::update_anchor_one_block(const ggml_fp16_t* anchor, int layer_id, int block_idx, WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();

  layer_id_ = layer_id;
  block_idx = block_idx;
  seq_len_ = config_.block_len;
  anchor_data_ = const_cast<uint16_t*>(anchor);

  // Each task updates the anchor of a certain position
  // backend->do_work_stealing_job(config_.anchor_num, [&](int task_id) {
  //     int k = task_id % config_.anchor_num;
  //     int head_id = task_id / config_.anchor_num;
  //     memcpy(anchor_[layer_id_][head_id][block_idx].data() +
  //                k * config_.head_dim,
  //            anchor_data_ + k * config_.head_dim,
  //            sizeof(uint16_t) * config_.head_dim);
  // });

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  printf("layer %d block %d time of writting anchor: %f s\n", layer_id, block_idx, duration.count());
}

/**
 * @brief 把一个 block 的 FP16 token importance 写入内部 importance cache。
 *
 * 函数按 block 内 token 位置创建并行任务，把调用方缓冲区中的 importance 值复制到指定层和物理 block。
 * 当前每个任务复制一个 FP16 元素；该接口主要用于单 block importance 的导入。
 *
 * @param importance 输入 FP16 importance 缓冲区，至少包含 config_.block_len 个元素。
 * @param layer_id 目标模型层编号。
 * @param block_idx 目标物理 block 编号。
 * @param backend 用于按 token 位置并行复制的工作线程池。
 */
void KVCache::update_importance_one_block(const ggml_fp16_t* importance, int layer_id, int block_idx,
                                          WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();

  layer_id_ = layer_id;
  block_idx = block_idx;
  seq_len_ = config_.block_len;
  importance_data_ = const_cast<uint16_t*>(importance);

  // Each task updates the importance of a certain position
  backend->do_work_stealing_job(
      config_.block_len, nullptr,
      [&](int task_id) {
        int k = task_id;
        memcpy(importance_[layer_id_][block_idx].data() + k, importance_data_ + k, sizeof(uint16_t));
      },
      nullptr);

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  printf("layer %d block %d time of writting importance: %f s\n", layer_id, block_idx, duration.count());
}

/**
 * @brief 从内部 importance cache 导出一个 block 的 FP16 token importance。
 *
 * 函数按 block 内 token 位置创建并行任务，将指定层和物理 block 中的 importance 复制到调用方缓冲区。
 * 当前每个任务复制一个 FP16 元素。
 *
 * @param importance 接收结果的 FP16 输出缓冲区，至少包含 config_.block_len 个元素。
 * @param layer_id 源模型层编号。
 * @param block_idx 源物理 block 编号。
 * @param backend 用于按 token 位置并行复制的工作线程池。
 */
void KVCache::get_importance_one_block(ggml_fp16_t* importance, int layer_id, int block_idx, WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();

  layer_id_ = layer_id;
  block_idx = block_idx;
  seq_len_ = config_.block_len;
  importance_data_ = const_cast<uint16_t*>(importance);

  // Each task updates the importance of a certain position
  backend->do_work_stealing_job(
      config_.block_len, nullptr,
      [&](int task_id) {
        int k = task_id;
        memcpy(importance_data_ + k, importance_[layer_id_][block_idx].data() + k, sizeof(uint16_t));
      },
      nullptr);

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  printf("layer %d block %d time of reading importance: %f s\n", layer_id, block_idx, duration.count());
}

/**
 * @brief 将一个完整 FP16 K/V block 量化并写入内部 Q4_0 KV cache。
 *
 * 函数扩展目标层的 block 容量，然后按 KV head 和 K/V 类型并行。K 按 token、沿 head_dim 每 32 个元素
 * 量化；V 为适合 PV GEMM 的转置布局，按通道、沿 block_len 每 32 个 token 量化。写入完成后更新该层
 * past_block_num_。该旧接口直接操作 Q4_0 存储，不按 config_.kv_type 分支。
 *
 * @param k_in 完整 FP16 Key block，逻辑布局为 [kv_head_num, block_len, head_dim]。
 * @param v_in 完整 FP16 Value block，输入逻辑布局为 [kv_head_num, block_len, head_dim]。
 * @param layer_id 目标模型层编号。
 * @param block_idx 目标物理 block 编号。
 * @param backend 用于并行处理各 KV head 的工作线程池。
 */
void KVCache::update_kvcache_one_block_fp16(const ggml_fp16_t* k_in, const ggml_fp16_t* v_in, int layer_id,
                                            int block_idx, WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();

  layer_id_ = layer_id;
  block_idx = block_idx;
  seq_len_ = config_.block_len;
  k_data_ = const_cast<uint16_t*>(k_in);
  v_data_ = const_cast<uint16_t*>(v_in);

  int new_block_num = std::max((int)past_block_num_[layer_id], block_idx + 1);

  importance_[layer_id_].resize(new_block_num);

  for (int i = 0; i < config_.kv_head_num; i++) {
    k_cache_q4[layer_id][i].resize(new_block_num);
    v_cache_q4[layer_id][i].resize(new_block_num);
    // anchor_[layer_id][i].resize(new_block_num);
  }

  for (int i = 0; i < new_block_num; i++) {
    importance_[layer_id][i].resize(config_.block_len);
  }

  // Each task updates the k cache or v cache of a certain header
  backend->do_work_stealing_job(
      config_.kv_head_num * 2, nullptr,
      [&](int task_id) {
        std::vector<float> block_fp32(32);
        int head_id = task_id / 2;
        if (task_id & 1) {
          // fill k_cache_
          k_cache_q4[layer_id_][head_id][block_idx].resize(config_.block_len * config_.head_dim / 32);
          for (int k = 0; k < config_.block_len; k++) {
            for (int l = 0; l < config_.head_dim / 32; l++) {
              block_q4_0 block;
              for (int m = 0; m < 32; m++) {
                block_fp32[m] = GGML_FP16_TO_FP32(
                    k_data_[((0 * config_.kv_head_num + head_id) * seq_len_ + 0 * config_.block_len + k) *
                                config_.head_dim +
                            l * 32 + m]);
              }
              quantize_row_q4_0(block_fp32.data(), &block, 32);
              k_cache_q4[layer_id_][head_id][block_idx][k * config_.head_dim / 32 + l] = block;
            }
          }
        } else {
          // fill v_cache_
          v_cache_q4[layer_id_][head_id][block_idx].resize(config_.head_dim * config_.block_len / 32);
          for (int k = 0; k < config_.block_len / 32; k++) {
            for (int l = 0; l < config_.head_dim; l++) {
              block_q4_0 block;
              for (int m = 0; m < 32; m++) {
                block_fp32[m] = GGML_FP16_TO_FP32(
                    v_data_[((0 * config_.kv_head_num + head_id) * seq_len_ + 0 * config_.block_len + k * 32 + m) *
                                config_.head_dim +
                            l]);
              }
              quantize_row_q4_0(block_fp32.data(), &block, 32);
              v_cache_q4[layer_id_][head_id][block_idx][l * config_.block_len / 32 + k] = block;
            }
          }
        }
      },
      nullptr);
  past_block_num_[layer_id] = new_block_num;

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  printf("layer %d block %d time of writting KV Cache: %f s\n", layer_id, block_idx, duration.count());
  // printf("get_one_block_fp16 duration: %ld\n", duration);
}

/**
 * @brief 将一个内部 Q4_0 KV block 反量化为调用方可读的 FP16 K/V。
 *
 * 函数按 KV head 和 K/V 类型并行读取指定物理 block。K 从按 token 量化的布局恢复；V 从内部转置的
 * [head_dim, block_len] 量化布局恢复为调用方的 token-major FP16 布局。
 *
 * @param k_in 接收 FP16 Key 的输出缓冲区，逻辑布局为 [kv_head_num, block_len, head_dim]。
 * @param v_in 接收 FP16 Value 的输出缓冲区，逻辑布局为 [kv_head_num, block_len, head_dim]。
 * @param layer_id 源模型层编号。
 * @param block_idx 源物理 block 编号。
 * @param backend 用于并行处理各 KV head 的工作线程池。
 */
void KVCache::get_kvcache_one_block_fp16(ggml_fp16_t* k_in, ggml_fp16_t* v_in, int layer_id, int block_idx,
                                         WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();

  layer_id_ = layer_id;
  seq_len_ = config_.block_len;
  k_data_ = reinterpret_cast<uint16_t*>(k_in);
  v_data_ = reinterpret_cast<uint16_t*>(v_in);

  // printf("layer_id: %d, block_idx: %d\n", layer_id, block_idx);
  // Each task gets the k cache or v cache of a certain header
  backend->do_work_stealing_job(
      config_.kv_head_num * 2, nullptr,
      [&](int task_id) {
        std::vector<float> block_fp32(32);
        int head_id = task_id / 2;
        if (task_id & 1) {
          // get k_cache_
          for (int k = 0; k < config_.block_len; k++) {
            for (int l = 0; l < config_.head_dim / 32; l++) {
              block_q4_0 block = k_cache_q4[layer_id_][head_id][block_idx][k * config_.head_dim / 32 + l];
              dequantize_row_q4_0(&block, block_fp32.data(), 32);
              for (int m = 0; m < 32; m++) {
                k_data_[((0 * config_.kv_head_num + head_id) * seq_len_ + 0 * config_.block_len + k) *
                            config_.head_dim +
                        l * 32 + m] = GGML_FP32_TO_FP16(block_fp32[m]);
              }
            }
          }
        } else {
          // get v_cache_
          for (int k = 0; k < config_.block_len / 32; k++) {
            for (int l = 0; l < config_.head_dim; l++) {
              block_q4_0 block = v_cache_q4[layer_id_][head_id][block_idx][l * config_.block_len / 32 + k];
              dequantize_row_q4_0(&block, block_fp32.data(), 32);
              for (int m = 0; m < 32; m++) {
                v_data_[((0 * config_.kv_head_num + head_id) * seq_len_ + 0 * config_.block_len + k * 32 + m) *
                            config_.head_dim +
                        l] = GGML_FP32_TO_FP16(block_fp32[m]);
              }
            }
          }
        }
      },
      nullptr);

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  printf("layer %d block %d time of reading KV Cache: %f s\n", layer_id, block_idx, duration.count());
  // printf("get_one_block_fp16 duration: %ld\n", duration);
}

// k_in: (batch_size, seq_len, head_num, head_dim)
// v_in: (batch_size, seq_len, head_num, head_dim)
/**
 * @brief 在同一 token-major FP16 缓冲区中导出历史 KV，并把紧随其后的新 KV 写回 cache。
 *
 * 函数按 (batch, logical block, KV head) 并行。对 [0, cache_seqlens) 范围，从内部 FP16/Q4_0/Q8_0 cache
 * 读取并在需要时反量化到 k_in/v_in；对 [cache_seqlens, cache_seqlens + q_len) 范围，则从同一缓冲区读取
 * 新 K/V，按配置量化并写回物理 block。V 在内部始终转换为适合 PV GEMM 的转置布局。
 *
 * @param k_in 输入/输出 FP16 Key 缓冲区，布局为 [batch_size, max_tokens, kv_head_num, head_dim]。
 * @param v_in 输入/输出 FP16 Value 缓冲区，布局与 k_in 相同。
 * @param layer_id 要读取和更新的模型层编号。
 * @param block_table 行优先 [batch_size, max_block_num] 逻辑到物理 block 映射。
 * @param batch_size 序列数量。
 * @param max_block_num 每个序列的 block 表宽度，同时决定缓冲区最大 token 容量。
 * @param cache_seqlens 每个序列在写入新 token 前的有效 token 数。
 * @param q_len 紧随已有 cache、需要写回的新 token 数。
 * @param backend 用于并行处理 batch、block 和 KV head 的工作线程池。
 */
void KVCache::get_and_update_kvcache_fp16(ggml_fp16_t* k_in, ggml_fp16_t* v_in, int layer_id, int* block_table,
                                          int batch_size, int max_block_num, int* cache_seqlens, int q_len,
                                          WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();

  layer_id_ = layer_id;
  k_data_ = const_cast<uint16_t*>(k_in);
  v_data_ = const_cast<uint16_t*>(v_in);

  // Each task updates the k cache and v cache of a certain header
  backend->do_work_stealing_job(
      config_.kv_head_num * max_block_num * batch_size, nullptr,
      [&](int task_id) {
        // printf("block_idx: %d, task_id: %d\n", block_idx, task_id);
        std::vector<float> block_fp32(32);
        int batch_id = task_id / (config_.kv_head_num * max_block_num);
        int block_id = (task_id / config_.kv_head_num) % max_block_num;
        int head_id = task_id % config_.kv_head_num;
        int block_idx = block_table[batch_id * max_block_num + block_id];
        int seq_len = cache_seqlens[batch_id];
        int block_l = block_id * config_.block_len;
        int block_r = block_id * config_.block_len + config_.block_len;

        if (block_l < seq_len) {
          if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
            for (int k = 0; k < config_.block_len; k++) {
              if (block_id * config_.block_len + k >= seq_len) break;
              for (int l = 0; l < config_.head_dim; l++) {
                k_data_[batch_id * (max_block_num * config_.block_len * config_.kv_head_num * config_.head_dim) +
                        block_id * (config_.block_len * config_.kv_head_num * config_.head_dim) +
                        k * (config_.kv_head_num * config_.head_dim) + head_id * config_.head_dim + l] =
                    k_cache_fp16_[layer_id_][head_id][block_idx][k * config_.head_dim + l];
                v_data_[batch_id * (max_block_num * config_.block_len * config_.kv_head_num * config_.head_dim) +
                        block_id * (config_.block_len * config_.kv_head_num * config_.head_dim) +
                        k * (config_.kv_head_num * config_.head_dim) + head_id * config_.head_dim + l] =
                    v_cache_fp16_[layer_id_][head_id][block_idx][l * config_.block_len + k];
              }
            }
          } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
            // get k_cache_
            for (int k = 0; k < config_.block_len; k++) {
              if (block_id * config_.block_len + k >= seq_len) break;
              for (int l = 0; l < config_.head_dim / 32; l++) {
                block_q4_0 block = k_cache_q4[layer_id_][head_id][block_idx][k * config_.head_dim / 32 + l];
                dequantize_row_q4_0(&block, block_fp32.data(), 32);
                for (int m = 0; m < 32; m++) {
                  k_data_[batch_id * (max_block_num * config_.block_len * config_.kv_head_num * config_.head_dim) +
                          block_id * (config_.block_len * config_.kv_head_num * config_.head_dim) +
                          k * (config_.kv_head_num * config_.head_dim) + head_id * config_.head_dim + l * 32 + m] =
                      GGML_FP32_TO_FP16(block_fp32[m]);
                }
              }
            }
            // get v_cache_
            for (int k = 0; k < config_.block_len / 32; k++) {
              for (int l = 0; l < config_.head_dim; l++) {
                block_q4_0 block = v_cache_q4[layer_id_][head_id][block_idx][l * config_.block_len / 32 + k];
                dequantize_row_q4_0(&block, block_fp32.data(), 32);
                for (int m = 0; m < 32; m++) {
                  if (block_id * config_.block_len + k * 32 + m >= seq_len) break;
                  v_data_[batch_id * (max_block_num * config_.block_len * config_.kv_head_num * config_.head_dim) +
                          block_id * (config_.block_len * config_.kv_head_num * config_.head_dim) +
                          (k * 32 + m) * config_.kv_head_num * config_.head_dim + head_id * config_.head_dim + l] =
                      GGML_FP32_TO_FP16(block_fp32[m]);
                }
              }
            }
          } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
            // get k_cache_
            for (int k = 0; k < config_.block_len; k++) {
              if (block_id * config_.block_len + k >= seq_len) break;
              for (int l = 0; l < config_.head_dim / 32; l++) {
                block_q8_0 block = k_cache_q8[layer_id_][head_id][block_idx][k * config_.head_dim / 32 + l];
                dequantize_row_q8_0(&block, block_fp32.data(), 32);
                for (int m = 0; m < 32; m++) {
                  k_data_[batch_id * (max_block_num * config_.block_len * config_.kv_head_num * config_.head_dim) +
                          block_id * (config_.block_len * config_.kv_head_num * config_.head_dim) +
                          k * (config_.kv_head_num * config_.head_dim) + head_id * config_.head_dim + l * 32 + m] =
                      GGML_FP32_TO_FP16(block_fp32[m]);
                }
              }
            }
            // get v_cache_
            for (int k = 0; k < config_.block_len / 32; k++) {
              for (int l = 0; l < config_.head_dim; l++) {
                block_q8_0 block = v_cache_q8[layer_id_][head_id][block_idx][l * config_.block_len / 32 + k];
                dequantize_row_q8_0(&block, block_fp32.data(), 32);
                for (int m = 0; m < 32; m++) {
                  if (block_id * config_.block_len + k * 32 + m >= seq_len) break;
                  v_data_[batch_id * (max_block_num * config_.block_len * config_.kv_head_num * config_.head_dim) +
                          block_id * (config_.block_len * config_.kv_head_num * config_.head_dim) +
                          (k * 32 + m) * config_.kv_head_num * config_.head_dim + head_id * config_.head_dim + l] =
                      GGML_FP32_TO_FP16(block_fp32[m]);
                }
              }
            }
          }
        }
        if (block_r > seq_len && block_l < seq_len + q_len) {
          if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
            for (int k = 0; k < config_.block_len; k++) {
              if (block_id * config_.block_len + k >= seq_len + q_len || block_id * config_.block_len + k < seq_len)
                continue;
              for (int l = 0; l < config_.head_dim; l++) {
                k_cache_fp16_[layer_id_][head_id][block_idx][k * config_.head_dim + l] =
                    k_data_[batch_id * (max_block_num * config_.block_len * config_.kv_head_num * config_.head_dim) +
                            block_id * (config_.block_len * config_.kv_head_num * config_.head_dim) +
                            k * (config_.kv_head_num * config_.head_dim) + head_id * config_.head_dim + l];
                v_cache_fp16_[layer_id_][head_id][block_idx][l * config_.block_len + k] =
                    v_data_[batch_id * (max_block_num * config_.block_len * config_.kv_head_num * config_.head_dim) +
                            block_id * (config_.block_len * config_.kv_head_num * config_.head_dim) +
                            k * (config_.kv_head_num * config_.head_dim) + head_id * config_.head_dim + l];
              }
            }
          } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
            // fill k_cache_
            for (int k = 0; k < config_.block_len; k++) {
              if (block_id * config_.block_len + k >= seq_len + q_len || block_id * config_.block_len + k < seq_len)
                continue;
              for (int l = 0; l < config_.head_dim / 32; l++) {
                block_q4_0 block;
                for (int m = 0; m < 32; m++) {
                  block_fp32[m] = GGML_FP16_TO_FP32(
                      k_data_[batch_id * (max_block_num * config_.block_len * config_.kv_head_num * config_.head_dim) +
                              block_id * (config_.block_len * config_.kv_head_num * config_.head_dim) +
                              k * (config_.kv_head_num * config_.head_dim) + head_id * config_.head_dim + l * 32 + m]);
                }
                quantize_row_q4_0(block_fp32.data(), &block, 32);
                k_cache_q4[layer_id_][head_id][block_idx][k * config_.head_dim / 32 + l] = block;
              }
            }

            // fill v_cache_
            for (int k = 0; k < config_.block_len / 32; k++) {
              for (int l = 0; l < config_.head_dim; l++) {
                block_q4_0 block;
                for (int m = 0; m < 32; m++) {
                  if (block_id * config_.block_len + k * 32 + m >= seq_len + q_len) {
                    block_fp32[m] = 0;
                    continue;
                  }
                  block_fp32[m] = GGML_FP16_TO_FP32(
                      v_data_[batch_id * (max_block_num * config_.block_len * config_.kv_head_num * config_.head_dim) +
                              block_id * (config_.block_len * config_.kv_head_num * config_.head_dim) +
                              (k * 32 + m) * config_.kv_head_num * config_.head_dim + head_id * config_.head_dim + l]);
                }
                quantize_row_q4_0(block_fp32.data(), &block, 32);
                v_cache_q4[layer_id_][head_id][block_idx][l * config_.block_len / 32 + k] = block;
              }
            }
          } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
            // fill k_cache_
            for (int k = 0; k < config_.block_len; k++) {
              if (block_id * config_.block_len + k >= seq_len + q_len || block_id * config_.block_len + k < seq_len)
                continue;
              for (int l = 0; l < config_.head_dim / 32; l++) {
                block_q8_0 block;
                for (int m = 0; m < 32; m++) {
                  block_fp32[m] = GGML_FP16_TO_FP32(
                      k_data_[batch_id * (max_block_num * config_.block_len * config_.kv_head_num * config_.head_dim) +
                              block_id * (config_.block_len * config_.kv_head_num * config_.head_dim) +
                              k * (config_.kv_head_num * config_.head_dim) + head_id * config_.head_dim + l * 32 + m]);
                }
                quantize_row_q8_0(block_fp32.data(), &block, 32);
                k_cache_q8[layer_id_][head_id][block_idx][k * config_.head_dim / 32 + l] = block;
              }
            }

            // fill v_cache_
            for (int k = 0; k < config_.block_len / 32; k++) {
              for (int l = 0; l < config_.head_dim; l++) {
                block_q8_0 block;
                for (int m = 0; m < 32; m++) {
                  if (block_id * config_.block_len + k * 32 + m >= seq_len + q_len) {
                    block_fp32[m] = 0;
                    continue;
                  }
                  block_fp32[m] = GGML_FP16_TO_FP32(
                      v_data_[batch_id * (max_block_num * config_.block_len * config_.kv_head_num * config_.head_dim) +
                              block_id * (config_.block_len * config_.kv_head_num * config_.head_dim) +
                              (k * 32 + m) * config_.kv_head_num * config_.head_dim + head_id * config_.head_dim + l]);
                }
                quantize_row_q8_0(block_fp32.data(), &block, 32);
                v_cache_q8[layer_id_][head_id][block_idx][l * config_.block_len / 32 + k] = block;
              }
            }
          }
        }
      },
      nullptr);

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;

  // printf("layer %d time of reading and updating KV Cache: %f s\n",
  // layer_id,
  //        duration.count());
}

/**
 * @brief 把一段 FP16 token importance 累加到 paged importance cache。
 *
 * 函数按 batch 和逻辑 block 并行，通过 block_table 定位物理 block，再把输入窗口覆盖到的每个 token、
 * 每个 Query head 的 importance 与内部旧值相加。该累积结果随后可用于 DYNAMIC anchor 选择重要 token。
 *
 * @param importance 输入 FP16 importance，布局为 [batch_size, max_block_num * block_len, q_head_num]。
 * @param layer_id 要更新的模型层编号。
 * @param block_table 行优先 [batch_size, max_block_num] 逻辑到物理 block 映射。
 * @param batch_size 序列数量。
 * @param max_block_num 每个序列的 block 表宽度。
 * @param offset 每个 batch 的 importance 窗口起始 token 偏移。
 * @param width 每个 batch 本次更新覆盖的 token 数。
 * @param backend 用于按 batch 和 block 并行累加的工作线程池。
 */
void KVCache::update_importance(const ggml_fp16_t* importance, int layer_id, int* block_table, int batch_size,
                                int max_block_num, int* offset, int width, WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();

  layer_id_ = layer_id;
  importance_data_ = const_cast<uint16_t*>(importance);

  // Each task updates the importance of a certain position
  backend->do_work_stealing_job(
      max_block_num * batch_size, nullptr,
      [&](int task_id) {
        int block_id = task_id % max_block_num;
        int batch_id = task_id / max_block_num;
        int block_idx = block_table[batch_id * max_block_num + block_id];
        if (block_id > (offset[batch_id] + width) / config_.block_len) {
          return;
        }
        for (int k = 0; k < config_.block_len; k++) {
          for (int head_id = 0; head_id < config_.q_head_num; head_id++) {
            importance_[layer_id_][block_idx][k][head_id] = GGML_FP32_TO_FP16(
                GGML_FP16_TO_FP32(importance_data_[batch_id * max_block_num * config_.block_len * config_.q_head_num +
                                                   (block_id * config_.block_len + k) * config_.q_head_num + head_id]) +
                GGML_FP16_TO_FP32(importance_[layer_id_][block_idx][k][head_id]));
          }
        }
      },
      nullptr);

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;

  // printf("layer %d time of updating importance: %f s\n", layer_id,
  //        duration.count());
}

/**
 * @brief 按逻辑顺序把一层 paged KV cache 导出为连续 FP16 K/V。
 *
 * 函数按 (batch, logical block, KV head) 并行，通过 block_table 读取物理 block。FP16 cache 直接复制，
 * Q4_0/Q8_0 cache 先反量化；内部转置保存的 V 会恢复为调用方使用的 token-major 布局。超出各序列
 * cache_seqlens 的位置不会写入。
 *
 * @param k_in 接收 Key 的 FP16 输出缓冲区，布局为 [batch_size, max_tokens, kv_head_num, head_dim]。
 * @param v_in 接收 Value 的 FP16 输出缓冲区，布局与 k_in 相同。
 * @param layer_id 要导出的模型层编号。
 * @param block_table 行优先 [batch_size, max_block_num] 逻辑到物理 block 映射。
 * @param batch_size 序列数量。
 * @param max_block_num 每个序列的 block 表宽度和输出最大 block 数。
 * @param cache_seqlens 每个序列的有效 token 数，用于裁剪尾块。
 * @param backend 用于并行读取 batch、block 和 KV head 的工作线程池。
 */
void KVCache::get_kvcache_fp16(ggml_fp16_t* k_in, ggml_fp16_t* v_in, int layer_id, int* block_table, int batch_size,
                               int max_block_num, int* cache_seqlens, WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();

  layer_id_ = layer_id;
  k_data_ = const_cast<uint16_t*>(k_in);
  v_data_ = const_cast<uint16_t*>(v_in);

  // Each task updates the k cache and v cache of a certain header
  backend->do_work_stealing_job(
      config_.kv_head_num * max_block_num * batch_size, nullptr,
      [&](int task_id) {
        // printf("block_idx: %d, task_id: %d\n", block_idx, task_id);
        std::vector<float> block_fp32(32);
        int batch_id = task_id / (config_.kv_head_num * max_block_num);
        int block_id = (task_id / config_.kv_head_num) % max_block_num;
        int head_id = task_id % config_.kv_head_num;
        int block_idx = block_table[batch_id * max_block_num + block_id];
        int seq_len = cache_seqlens[batch_id];
        int block_l = block_id * config_.block_len;
        int block_r = block_id * config_.block_len + config_.block_len;

        if (block_l < seq_len) {
          if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
            for (int k = 0; k < config_.block_len; k++) {
              if (block_id * config_.block_len + k >= seq_len) break;
              for (int l = 0; l < config_.head_dim; l++) {
                k_data_[batch_id * (max_block_num * config_.block_len * config_.kv_head_num * config_.head_dim) +
                        block_id * (config_.block_len * config_.kv_head_num * config_.head_dim) +
                        k * (config_.kv_head_num * config_.head_dim) + head_id * config_.head_dim + l] =
                    k_cache_fp16_[layer_id_][head_id][block_idx][k * config_.head_dim + l];
                v_data_[batch_id * (max_block_num * config_.block_len * config_.kv_head_num * config_.head_dim) +
                        block_id * (config_.block_len * config_.kv_head_num * config_.head_dim) +
                        k * (config_.kv_head_num * config_.head_dim) + head_id * config_.head_dim + l] =
                    v_cache_fp16_[layer_id_][head_id][block_idx][l * config_.block_len + k];
              }
            }
          } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
            // get k_cache_
            for (int k = 0; k < config_.block_len; k++) {
              if (block_id * config_.block_len + k >= seq_len) break;
              for (int l = 0; l < config_.head_dim / 32; l++) {
                block_q4_0 block = k_cache_q4[layer_id_][head_id][block_idx][k * config_.head_dim / 32 + l];
                dequantize_row_q4_0(&block, block_fp32.data(), 32);
                for (int m = 0; m < 32; m++) {
                  k_data_[batch_id * (max_block_num * config_.block_len * config_.kv_head_num * config_.head_dim) +
                          block_id * (config_.block_len * config_.kv_head_num * config_.head_dim) +
                          k * (config_.kv_head_num * config_.head_dim) + head_id * config_.head_dim + l * 32 + m] =
                      GGML_FP32_TO_FP16(block_fp32[m]);
                }
              }
            }
            // get v_cache_
            for (int k = 0; k < config_.block_len / 32; k++) {
              for (int l = 0; l < config_.head_dim; l++) {
                block_q4_0 block = v_cache_q4[layer_id_][head_id][block_idx][l * config_.block_len / 32 + k];
                dequantize_row_q4_0(&block, block_fp32.data(), 32);
                for (int m = 0; m < 32; m++) {
                  if (block_id * config_.block_len + k * 32 + m >= seq_len) break;
                  v_data_[batch_id * (max_block_num * config_.block_len * config_.kv_head_num * config_.head_dim) +
                          block_id * (config_.block_len * config_.kv_head_num * config_.head_dim) +
                          (k * 32 + m) * config_.kv_head_num * config_.head_dim + head_id * config_.head_dim + l] =
                      GGML_FP32_TO_FP16(block_fp32[m]);
                }
              }
            }
          } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
            // get k_cache_
            for (int k = 0; k < config_.block_len; k++) {
              if (block_id * config_.block_len + k >= seq_len) break;
              for (int l = 0; l < config_.head_dim / 32; l++) {
                block_q8_0 block = k_cache_q8[layer_id_][head_id][block_idx][k * config_.head_dim / 32 + l];
                dequantize_row_q8_0(&block, block_fp32.data(), 32);
                for (int m = 0; m < 32; m++) {
                  k_data_[batch_id * (max_block_num * config_.block_len * config_.kv_head_num * config_.head_dim) +
                          block_id * (config_.block_len * config_.kv_head_num * config_.head_dim) +
                          k * (config_.kv_head_num * config_.head_dim) + head_id * config_.head_dim + l * 32 + m] =
                      GGML_FP32_TO_FP16(block_fp32[m]);
                }
              }
            }
            // get v_cache_
            for (int k = 0; k < config_.block_len / 32; k++) {
              for (int l = 0; l < config_.head_dim; l++) {
                block_q8_0 block = v_cache_q8[layer_id_][head_id][block_idx][l * config_.block_len / 32 + k];
                dequantize_row_q8_0(&block, block_fp32.data(), 32);
                for (int m = 0; m < 32; m++) {
                  if (block_id * config_.block_len + k * 32 + m >= seq_len) break;
                  v_data_[batch_id * (max_block_num * config_.block_len * config_.kv_head_num * config_.head_dim) +
                          block_id * (config_.block_len * config_.kv_head_num * config_.head_dim) +
                          (k * 32 + m) * config_.kv_head_num * config_.head_dim + head_id * config_.head_dim + l] =
                      GGML_FP32_TO_FP16(block_fp32[m]);
                }
              }
            }
          }
        }
      },
      nullptr);

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
}

/**
 * @brief 把连续 FP16 K/V token 追加写入一层 paged KV cache。
 *
 * 函数按 (batch, KV head, Query token) 并行，根据 cache_seqlens 和 token 偏移计算目标逻辑 block、物理
 * block 及 block 内位置。FP16 模式直接写入；Q4_0/Q8_0 模式对 K 按 head_dim 分组量化，并对 V 所在的
 * 32-token 量化块执行读改写，保持内部 [head_dim, block_len] 转置布局。
 *
 * @param k_in 新增 FP16 Key，布局为 [batch_size, q_len, kv_head_num, head_dim]。
 * @param v_in 新增 FP16 Value，布局与 k_in 相同。
 * @param layer_id 目标模型层编号。
 * @param block_table 行优先 [batch_size, max_block_num] 逻辑到物理 block 映射。
 * @param batch_size 序列数量。
 * @param max_block_num 每个序列的 block 表宽度。
 * @param cache_seqlens 每个序列写入前的有效 token 数。
 * @param q_len 每个序列要追加的 token 数；decode 路径通常为 1。
 * @param backend 用于并行写入 batch、KV head 和 token 的工作线程池。
 */
void KVCache::update_kvcache_fp16(const ggml_fp16_t* k_in, const ggml_fp16_t* v_in, int layer_id, int* block_table,
                                  int batch_size, int max_block_num, int* cache_seqlens, int q_len,
                                  WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();

  layer_id_ = layer_id;
  k_data_ = const_cast<uint16_t*>(k_in);
  v_data_ = const_cast<uint16_t*>(v_in);
  // Each task updates the k cache and v cache of a certain header
  backend->do_work_stealing_job(
      batch_size * config_.kv_head_num * q_len, nullptr,
      [&](int task_id) {
        int batch_id = task_id / (config_.kv_head_num * q_len);
        int head_id = task_id / q_len % config_.kv_head_num;
        int seq_len = cache_seqlens[batch_id] + task_id % q_len;
        int q_offset = task_id % q_len;

        int block_id = seq_len / config_.block_len;
        int block_idx = block_table[batch_id * max_block_num + block_id];
        int pos_in_block = seq_len % config_.block_len;

        if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
          for (int l = 0; l < config_.head_dim; l++) {
            k_cache_fp16_[layer_id_][head_id][block_idx][pos_in_block * config_.head_dim + l] =
                k_data_[batch_id * (q_len * config_.kv_head_num * config_.head_dim) +
                        q_offset * config_.kv_head_num * config_.head_dim + head_id * config_.head_dim + l];
            v_cache_fp16_[layer_id_][head_id][block_idx][l * config_.block_len + pos_in_block] =
                v_data_[batch_id * (q_len * config_.kv_head_num * config_.head_dim) +
                        q_offset * config_.kv_head_num * config_.head_dim + head_id * config_.head_dim + l];
          }
        } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
          std::vector<float> block_fp32(32);
          // fill k_cache_
          for (int l = 0; l < config_.head_dim / 32; l++) {
            block_q4_0 block;
            for (int m = 0; m < 32; m++) {
              block_fp32[m] = GGML_FP16_TO_FP32(k_data_[batch_id * (q_len * config_.kv_head_num * config_.head_dim) +
                                                        head_id * config_.head_dim + l * 32 + m]);
            }
            quantize_row_q4_0(block_fp32.data(), &block, 32);

            k_cache_q4[layer_id_][head_id][block_idx][pos_in_block * config_.head_dim / 32 + l] = block;
          }

          // fill v_cache_
          for (int l = 0; l < config_.head_dim; l++) {
            block_q4_0 block =
                v_cache_q4[layer_id_][head_id][block_idx][l * config_.block_len / 32 + pos_in_block / 32];
            dequantize_row_q4_0(&block, block_fp32.data(), 32);
            block_fp32[pos_in_block % 32] = GGML_FP16_TO_FP32(
                v_data_[batch_id * (q_len * config_.kv_head_num * config_.head_dim) + head_id * config_.head_dim + l]);
            quantize_row_q4_0(block_fp32.data(), &block, 32);
            v_cache_q4[layer_id_][head_id][block_idx][l * config_.block_len / 32 + pos_in_block / 32] = block;
          }
        } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
          std::vector<float> block_fp32(32);
          // fill k_cache_
          for (int l = 0; l < config_.head_dim / 32; l++) {
            block_q8_0 block;
            for (int m = 0; m < 32; m++) {
              block_fp32[m] = GGML_FP16_TO_FP32(k_data_[batch_id * (q_len * config_.kv_head_num * config_.head_dim) +
                                                        head_id * config_.head_dim + l * 32 + m]);
            }
            quantize_row_q8_0(block_fp32.data(), &block, 32);

            k_cache_q8[layer_id_][head_id][block_idx][pos_in_block * config_.head_dim / 32 + l] = block;
          }

          // fill v_cache_
          for (int l = 0; l < config_.head_dim; l++) {
            block_q8_0 block =
                v_cache_q8[layer_id_][head_id][block_idx][l * config_.block_len / 32 + pos_in_block / 32];
            dequantize_row_q8_0(&block, block_fp32.data(), 32);
            block_fp32[pos_in_block % 32] = GGML_FP16_TO_FP32(
                v_data_[batch_id * (q_len * config_.kv_head_num * config_.head_dim) + head_id * config_.head_dim + l]);
            quantize_row_q8_0(block_fp32.data(), &block, 32);
            v_cache_q8[layer_id_][head_id][block_idx][l * config_.block_len / 32 + pos_in_block / 32] = block;
          }
        }
      },
      nullptr);

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  // printf("layer %d time of reading KV Cache: %f s\n", layer_id,
  //        duration.count());
}

/**
 * @brief 将一层内部 Q4_0 KV cache 的全部有效 token 导出为连续 FP16 数据。
 *
 * 函数按 KV head、物理 block 和 K/V 类型并行，遍历 past_block_num_ 中属于 cache_total_len_ 的有效范围，
 * 反量化 Q4_0 K/V，并把内部转置的 V 恢复为 token-major 输出。该旧接口直接读取 Q4_0 存储，不按
 * config_.kv_type 分支。
 *
 * @param layer_id 要导出的模型层编号。
 * @param k_in 接收 FP16 Key 的输出缓冲区，逻辑布局为 [kv_head_num, cache_total_len, head_dim]。
 * @param v_in 接收 FP16 Value 的输出缓冲区，逻辑布局与 k_in 相同。
 * @param backend 用于并行处理 KV head、block 和 K/V 的工作线程池。
 */
void KVCache::get_all_kvcache_one_layer(int layer_id, ggml_fp16_t* k_in, ggml_fp16_t* v_in, WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();

  layer_id_ = layer_id;
  seq_len_ = config_.block_len;
  block_num_ = get_cache_total_block_num();
  k_data_ = reinterpret_cast<uint16_t*>(k_in);
  v_data_ = reinterpret_cast<uint16_t*>(v_in);

  // Each task gets the k cache or v cache of a certain header
  backend->do_work_stealing_job(
      config_.kv_head_num * past_block_num_[layer_id] * 2, nullptr,
      [&](int task_id) {
        std::vector<float> block_fp32(32);
        int head_id = task_id / 2 / past_block_num_[layer_id];
        int block_idx = task_id / 2 % past_block_num_[layer_id];
        if (block_idx >= block_num_) return;

        int max_offset = 0;
        if (task_id & 1) {
          // get k_cache_
          for (int k = 0; k < config_.block_len; k++) {
            if (block_idx * seq_len_ + k >= cache_total_len_) break;
            for (int l = 0; l < config_.head_dim / 32; l++) {
              block_q4_0 block = k_cache_q4[layer_id_][head_id][block_idx][k * config_.head_dim / 32 + l];
              dequantize_row_q4_0(&block, block_fp32.data(), 32);
              for (int m = 0; m < 32; m++) {
                k_data_[(head_id * cache_total_len_ + block_idx * config_.block_len + k) * config_.head_dim + l * 32 +
                        m] = GGML_FP32_TO_FP16(block_fp32[m]);
                max_offset =
                    std::max(max_offset,
                             (int)(head_id * cache_total_len_ + block_idx * config_.block_len + k) * config_.head_dim +
                                 l * 32 + m);
              }
            }
          }
        } else {
          // get v_cache_
          for (int k = 0; k < config_.block_len / 32; k++) {
            for (int l = 0; l < config_.head_dim; l++) {
              block_q4_0 block = v_cache_q4[layer_id_][head_id][block_idx][l * config_.block_len / 32 + k];
              dequantize_row_q4_0(&block, block_fp32.data(), 32);
              for (int m = 0; m < 32; m++) {
                if (block_idx * seq_len_ + k * 32 + m >= cache_total_len_) break;
                v_data_[(head_id * cache_total_len_ + block_idx * config_.block_len + k * 32 + m) * config_.head_dim +
                        l] = GGML_FP32_TO_FP16(block_fp32[m]);
                max_offset = std::max(
                    max_offset,
                    (int)((head_id * cache_total_len_ + block_idx * config_.block_len + k * 32 + m) * config_.head_dim +
                          l));
              }
            }
          }
        }
      },
      nullptr);

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  // printf("layer %d block num %d time of reading all KV Cache: %f s\n",
  //        layer_id, block_num_, duration.count());
}
