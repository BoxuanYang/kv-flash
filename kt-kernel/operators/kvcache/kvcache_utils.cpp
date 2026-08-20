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
 * @brief 将 KV cache 支持的 GGML 数据类型转换为可读字符串。
 *
 * 该辅助函数用于构造配置日志；未识别的枚举值返回 "UNDIFINED"。
 *
 * @param type 要转换的 GGML 数据类型枚举。
 * @return 与 type 对应的固定字符串。
 */
std::string ggml_type_to_string(ggml_type type) {
  switch (type) {
    case GGML_TYPE_F32:
      return "GGML_TYPE_F32";
    case GGML_TYPE_F16:
      return "GGML_TYPE_F16";
    case GGML_TYPE_Q4_0:
      return "GGML_TYPE_Q4_0";
    case GGML_TYPE_Q8_0:
      return "GGML_TYPE_Q8_0";
  }
  return "UNDIFINED";
}
/**
 * @brief 将 block anchor 算法类型转换为可读字符串。
 *
 * @param type 要转换的 AnchorType 枚举。
 * @return DYNAMIC、BLOCK_MEAN、BLOCK_MAX、FIXED_ANCHOR 或 QUEST；未知值返回 "UNDIFINED"。
 */
std::string AnchorTypeToString(AnchorType type) {
  switch (type) {
    case AnchorType::DYNAMIC:
      return "DYNAMIC";
    case AnchorType::BLOCK_MEAN:
      return "BLOCK_MEAN";
    case AnchorType::BLOCK_MAX:
      return "BLOCK_MAX";
    case AnchorType::FIXED_ANCHOR:
      return "FIXED_ANCHOR";
    case AnchorType::QUEST:
      return "QUEST";
  }
  return "UNDIFINED";
}
/**
 * @brief 将 block 检索粒度转换为 Python 配置使用的可读名称。
 *
 * LAYER 映射为 SHARED，KVHEAD 映射为 SEPARATE，QHEAD 映射为 INDIVIDUAL。
 *
 * @param type 要转换的 RetrievalType 枚举。
 * @return 对应的检索模式字符串；未知值返回 "UNDIFINED"。
 */
std::string RetrievalTypeToString(RetrievalType type) {
  switch (type) {
    case RetrievalType::LAYER:
      return "SHARED";
    case RetrievalType::KVHEAD:
      return "SEPARATE";
    case RetrievalType::QHEAD:
      return "INDIVIDUAL";
  }
  return "UNDIFINED";
}
/**
 * @brief 构造并校验 KV cache 的静态配置。
 *
 * 构造函数保存模型形状、block/anchor 策略、检索复用步长和容量上限，打印完整配置，并校验 Query head
 * 数能够被 KV head 数整除，以保证 GQA 分组 n_gqa 为整数。
 *
 * @param layer_num 模型层数，也是 KV cache 的层维度。
 * @param kv_head_num 每层 Key/Value head 数。
 * @param q_head_num 每层 Query head 数。
 * @param head_dim 每个 Attention head 的特征维度。
 * @param block_len 每个物理 cache block 容纳的 token 数。
 * @param anchor_num 每个 block 保存的检索 anchor 数。
 * @param anchor_type 从 block K 和 importance 生成 anchor 的算法。
 * @param kv_type 内部 K/V 存储类型，当前支持 FP16、Q4_0 和 Q8_0。
 * @param retrieval_type block 选择粒度：整层共享、按 KV head 或按 Query head。
 * @param layer_step 每隔多少层重新计算一次 block 选择。
 * @param token_step 每隔多少个 decode token 重新计算一次 block 选择。
 * @param layer_offset 刷新层在 layer_step 周期中的偏移量。
 * @param max_block_num 每层最多预分配的物理 block 数。
 * @param max_batch_size 最多预分配的 batch 数。
 * @param max_thread_num 最多预分配的工作线程及线程局部缓冲区数量。
 */
KVCacheConfig::KVCacheConfig(int layer_num, int kv_head_num, int q_head_num, int head_dim, int block_len,
                             int anchor_num, AnchorType anchor_type, ggml_type kv_type, RetrievalType retrieval_type,
                             int layer_step, int token_step, int layer_offset, int max_block_num, int max_batch_size,
                             int max_thread_num)
    : layer_num(layer_num),
      kv_head_num(kv_head_num),
      q_head_num(q_head_num),
      head_dim(head_dim),
      block_len(block_len),
      anchor_num(anchor_num),
      anchor_type(anchor_type),
      kv_type(kv_type),
      retrieval_type(retrieval_type),
      layer_step(layer_step),
      token_step(token_step),
      layer_offset(layer_offset),
      max_block_num(max_block_num),
      max_batch_size(max_batch_size),
      max_thread_num(max_thread_num) {
  printf(
      "layer_num: %d, kv_head_num: %d, q_head_num: %d, head_dim: %d, "
      "block_len: %d, anchor_num: %d, anchor_type: %s, kv_type: %s, "
      "retrieval_type: %s, layer_step: %d, token_step: %d, layer_offset: %d,"
      "max_block_num: %d, max_batch_size: %d, max_thread_num: %d\n",
      layer_num, kv_head_num, q_head_num, head_dim, block_len, anchor_num, AnchorTypeToString(anchor_type).c_str(),
      ggml_type_to_string(kv_type).c_str(), RetrievalTypeToString(retrieval_type).c_str(), layer_step, token_step,
      layer_offset, max_block_num, max_batch_size, max_thread_num);
  assert(q_head_num % kv_head_num == 0);
}
/**
 * @brief 按配置创建 KV cache，并预分配检索、量化和 Attention 所需状态。
 *
 * 构造函数计算 GQA 分组大小，按 kv_type 创建对应的 K/V 容器，分配 anchor、importance 和历史选择状态，
 * 再调用 ThreadResize()、BatchResize() 和 BlockResize() 建立全部线程、batch 和 block 维度的缓冲区。
 *
 * @param config 已完成校验的 KVCacheConfig；其容量字段决定本实例的预分配上限。
 */
KVCache::KVCache(KVCacheConfig config) {
  this->config_ = config;

  n_gqa_ = config_.q_head_num / config_.kv_head_num;
  if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
    // TODO: Elegant implement
    k_cache_fp16_.resize(config_.layer_num);
    v_cache_fp16_.resize(config_.layer_num);
    selected_blocks_num_history_.resize(config_.layer_num / config_.layer_step);
    if (config_.retrieval_type == RetrievalType::LAYER) {
      selected_blocks_history_.resize(config_.layer_num / config_.layer_step);
    } else if (config_.retrieval_type == RetrievalType::KVHEAD) {
      selected_blocks_history_kvhead_.resize(config_.layer_num / config_.layer_step);
    } else if (config_.retrieval_type == RetrievalType::QHEAD) {
    }
  } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
    k_cache_q4.resize(config.layer_num);
    v_cache_q4.resize(config.layer_num);
  } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
    k_cache_q8.resize(config.layer_num);
    v_cache_q8.resize(config.layer_num);
  } else {
    assert(false);
  }
  anchor_.resize(config.layer_num * config.max_block_num * config.anchor_num * config.q_head_num * config.head_dim);
  importance_.resize(config.layer_num);
  past_block_num_.resize(config.layer_num);
  for (int i = 0; i < config.layer_num; i++) {
    past_block_num_[i] = 0;
  }

  ThreadResize(config.max_thread_num);
  BatchResize(config.max_batch_size);
  BlockResize(config.max_block_num);
  q_fp32.resize(n_gqa_ * config.head_dim);
}

/**
 * @brief 调整每个工作线程独享的 Attention 临时缓冲区数量和大小。
 *
 * 每个线程获得 Q8 输出、Attention score、FP32 输出、LSE、在线 block 归并状态、尾块 mask 和 kernel
 * draft 工作区，避免并行 block 计算时发生临时内存竞争。
 *
 * @param thread_num 需要支持的最大并行工作线程数；应不小于 WorkerPool 的线程数。
 */
void KVCache::ThreadResize(int thread_num) {
  thread_local_output_q8_0_.resize(thread_num);
  thread_local_attn_score_.resize(thread_num);
  thread_local_output_fp32_.resize(thread_num);
  thread_local_attn_lse_.resize(thread_num);
  thread_local_cur_output_fp32_.resize(thread_num);
  thread_local_cur_attn_lse_.resize(thread_num);
  thread_local_draft_.resize(thread_num);
  thread_cur_head_idx_.resize(thread_num);
  thread_local_attn_mask_.resize(thread_num);
  for (int i = 0; i < thread_num; i++) {
    thread_local_output_q8_0_[i].resize(n_gqa_ * config_.head_dim / QK8_0);
    thread_local_attn_score_[i].resize(n_gqa_ * config_.block_len);
    thread_local_output_fp32_[i].resize(n_gqa_ * config_.head_dim);
    thread_local_attn_lse_[i].resize(n_gqa_);
    thread_local_cur_output_fp32_[i].resize(n_gqa_ * config_.head_dim);
    thread_local_cur_attn_lse_[i].resize(n_gqa_);
    thread_local_draft_[i].resize(2 * n_gqa_ * config_.block_len + 6 * n_gqa_ * config_.head_dim +
                                  2 * config_.block_len * config_.head_dim +
                                  config_.block_len * config_.head_dim / QK4_0);
    thread_local_attn_mask_[i].resize(config_.block_len / 8);
  }
}
/**
 * @brief 调整所有按 batch 索引的 Query、输出、检索表和同步状态。
 *
 * 函数按 retrieval_type 分配 layer 共享或 per-head block 表与历史，给每个 (batch, KV head) 创建 mutex、
 * Query 量化缓冲区、FP32 输出和 LSE，并分配每个 batch 的平均 Query 与稀疏度缓冲区。
 *
 * @param batch_size 需要预分配的最大 batch 数；运行时 batch 不得超过该值。
 */
void KVCache::BatchResize(int batch_size) {
  mutex_.resize(batch_size);
  q_q8_0_.resize(batch_size);
  q_fp32_.resize(batch_size);
  output_fp32_.resize(batch_size);
  attn_lse_.resize(batch_size);
  block_lse_.resize(batch_size);
  attn_sparsity_.resize(batch_size);

  if (config_.retrieval_type == RetrievalType::LAYER) {
    block_table_before_retrieval_.resize(batch_size);
    block_table_after_retrieval_.resize(batch_size);

    for (int i = 0; i < config_.layer_num / config_.layer_step; i++) {
      selected_blocks_history_[i].resize(batch_size);
    }

  } else if (config_.retrieval_type == RetrievalType::KVHEAD) {
    block_table_before_retrieval_kvhead_.resize(batch_size);
    block_table_after_retrieval_kvhead_.resize(batch_size);
    for (int i = 0; i < config_.layer_num / config_.layer_step; i++) {
      selected_blocks_history_kvhead_[i].resize(batch_size);
    }
  } else if (config_.retrieval_type == RetrievalType::QHEAD) {
    block_table_before_retrieval_qhead_.resize(batch_size);
    block_table_after_retrieval_qhead_.resize(batch_size);
  }
  cache_seqlens_.resize(batch_size);
  if (config_.retrieval_type == RetrievalType::LAYER) {
    block_similar_.resize(batch_size);
  } else if (config_.retrieval_type == RetrievalType::KVHEAD) {
    block_similar_kv_head_.resize(batch_size);
  } else if (config_.retrieval_type == RetrievalType::QHEAD) {
    block_similar_q_head_.resize(batch_size);
  }
  for (int i = 0; i < batch_size; i++) {
    top_similar_block_.resize(batch_size);

    mutex_[i].resize(config_.kv_head_num);
    q_q8_0_[i].resize(config_.kv_head_num);
    q_fp32_[i].resize(config_.kv_head_num);
    output_fp32_[i].resize(config_.kv_head_num);
    attn_lse_[i].resize(config_.kv_head_num);

    for (int j = 0; j < config_.kv_head_num; j++) {
      if (!mutex_[i][j]) {
        mutex_[i][j] = std::make_unique<std::mutex>();
      }
      q_q8_0_[i][j].resize(n_gqa_ * config_.head_dim / QK8_0);
      q_fp32_[i][j].resize(n_gqa_ * config_.head_dim);
      output_fp32_[i][j].resize(n_gqa_ * config_.head_dim);
      attn_lse_[i][j].resize(n_gqa_);
    }
  }
  avg_q.resize(batch_size);
  avg_q_fp16.resize(batch_size);
  for (int i = 0; i < batch_size; i++) {
    attn_sparsity_[i].resize(config_.q_head_num);
    avg_q[i].resize(config_.q_head_num * config_.head_dim);
    avg_q_fp16[i].resize(config_.q_head_num * config_.head_dim);
  }
}

/**
 * @brief 调整全部按物理 block 索引的 KV、anchor 辅助状态和检索缓冲区。
 * 
 * 初始化KV Cache
 *
 * 函数分配 RoPE 表、历史选择、每层每 KV head 的 FP16/Q4_0/Q8_0 K/V block、每个 token 的 importance、
 * similarity 和逻辑 block 表。K 使用 token-major 布局，V 使用适合 PV GEMM 的 [head_dim, block_len] 转置
 * 布局；量化类型每 32 个元素保存一个量化块。
 *
 * @param max_block_num 每层要支持的最大物理 block 数，通常等于 config_.max_block_num。
 */
void KVCache::BlockResize(int max_block_num) {
  sin_.resize(max_block_num * config_.block_len);
  cos_.resize(max_block_num * config_.block_len);
  for (int i = 0; i < max_block_num * config_.block_len; i++) {
    sin_[i].resize(config_.head_dim);
    cos_[i].resize(config_.head_dim);
  }

  for (int i = 0; i < config_.layer_num / config_.layer_step; i++) {
    for (int j = 0; j < config_.max_batch_size; j++) {
      if (config_.retrieval_type == RetrievalType::LAYER) {
        selected_blocks_history_[i][j].resize(max_block_num);
      } else if (config_.retrieval_type == RetrievalType::KVHEAD) {
        selected_blocks_history_kvhead_[i][j].resize(max_block_num);
        for (int k = 0; k < config_.max_block_num; k++) {
          selected_blocks_history_kvhead_[i][j][k].resize(config_.kv_head_num);
        }
      } else if (config_.retrieval_type == RetrievalType::QHEAD) {
      }
    }
  }

  for (int layer_id = 0; layer_id < config_.layer_num; layer_id++) {
    importance_[layer_id].resize(max_block_num);

    // 这是 KVCache CPU 内存 buffer 的实际分配位置。每个 [layer][kv_head][physical_block] 都拥有一段
    // 独立连续的 block_data；K 按 [token][head_dim] 排列，V 按 [head_dim][token] 转置排列。
    if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
      // TODO: Elegant implement
      
      // 初始化该层
      k_cache_fp16_[layer_id].resize(config_.kv_head_num);
      v_cache_fp16_[layer_id].resize(config_.kv_head_num);

      // 对于一层的KVCache, 每一个KV Head分配max_block_num个块
      for (int i = 0; i < config_.kv_head_num; i++) {
        k_cache_fp16_[layer_id][i].resize(max_block_num);
        v_cache_fp16_[layer_id][i].resize(max_block_num);


        for (int j = 0; j < max_block_num; j++) {
          k_cache_fp16_[layer_id][i][j].resize(config_.block_len * config_.head_dim);
          v_cache_fp16_[layer_id][i][j].resize(config_.block_len * config_.head_dim);
        }
      }

    } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
      k_cache_q4[layer_id].resize(config_.kv_head_num);
      v_cache_q4[layer_id].resize(config_.kv_head_num);
      for (int i = 0; i < config_.kv_head_num; i++) {
        k_cache_q4[layer_id][i].resize(max_block_num);
        v_cache_q4[layer_id][i].resize(max_block_num);

        for (int j = 0; j < max_block_num; j++) {
          k_cache_q4[layer_id][i][j].resize(config_.block_len * config_.head_dim / 32);
          v_cache_q4[layer_id][i][j].resize(config_.block_len * config_.head_dim / 32);
        }
      }
    } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
      k_cache_q8[layer_id].resize(config_.kv_head_num);
      v_cache_q8[layer_id].resize(config_.kv_head_num);
      for (int i = 0; i < config_.kv_head_num; i++) {
        k_cache_q8[layer_id][i].resize(max_block_num);
        v_cache_q8[layer_id][i].resize(max_block_num);

        for (int j = 0; j < max_block_num; j++) {
          k_cache_q8[layer_id][i][j].resize(config_.block_len * config_.head_dim / 32);
          v_cache_q8[layer_id][i][j].resize(config_.block_len * config_.head_dim / 32);
        }
      }
    } else {
      assert(false);
    }

    // 分配KVCache结束
    
    for (int i = 0; i < config_.max_batch_size; i++) {
      if (config_.retrieval_type == RetrievalType::LAYER) {
        block_similar_[i].resize(max_block_num);
        block_table_before_retrieval_[i].resize(max_block_num);
        block_table_after_retrieval_[i].resize(max_block_num);
      } else if (config_.retrieval_type == RetrievalType::KVHEAD) {
        block_similar_kv_head_[i].resize(max_block_num);
        block_table_before_retrieval_kvhead_[i].resize(max_block_num);
        block_table_after_retrieval_kvhead_[i].resize(max_block_num);
        for (int j = 0; j < max_block_num; j++) {
          block_similar_kv_head_[i][j].resize(config_.kv_head_num);
          block_table_before_retrieval_kvhead_[i][j].resize(config_.kv_head_num);
          block_table_after_retrieval_kvhead_[i][j].resize(config_.kv_head_num);
        }
      } else if (config_.retrieval_type == RetrievalType::QHEAD) {
        block_similar_q_head_[i].resize(max_block_num);
        block_table_before_retrieval_qhead_[i].resize(max_block_num);
        block_table_after_retrieval_qhead_[i].resize(max_block_num);
        for (int j = 0; j < max_block_num; j++) {
          block_similar_q_head_[i][j].resize(config_.q_head_num);
          block_table_before_retrieval_qhead_[i][j].resize(config_.q_head_num);
          block_table_after_retrieval_qhead_[i][j].resize(config_.q_head_num);
        }
      }
      block_lse_[i].resize(max_block_num);
      for (int j = 0; j < max_block_num; j++) {
        block_lse_[i][j].resize(config_.q_head_num);
      }
    }

    for (int i = 0; i < max_block_num; i++) {
      importance_[layer_id][i].resize(config_.block_len);
      for (int j = 0; j < config_.block_len; j++) {
        importance_[layer_id][i][j].resize(config_.q_head_num);
      }
    }
  }
}

/**
 * @brief 为全部层和有效物理 block 生成用于稀疏检索的 FP16 anchor。
 *
 * 函数按 (layer, batch, logical block) 并行，并通过 block_table 定位物理 block。DYNAMIC 根据累计
 * importance 选择重要 token 的 K 并聚合；BLOCK_MEAN 对 block 内 K 求均值；BLOCK_MAX 逐维取最大值；
 * FIXED_ANCHOR 按固定步长采样并聚合；QUEST 为每个维度保存 block 内 K 的最大值与最小值。量化 cache
 * 会先按 32 个元素反量化，再写入统一的 FP16 anchor_ 数组。
 *
 * @param block_table 行优先 [batch_size, max_block_num] 逻辑到物理 block 映射。
 * @param cache_seqlens 每个 batch 的有效 token 数；超出范围的 block 会被跳过。
 * @param batch_size 序列数量。
 * @param max_block_num 每个序列参与遍历的 block 表宽度。
 * @param backend 用于并行处理层、batch 和 block 的工作线程池。
 */
void KVCache::calc_anchor_all_layers(int* block_table, int* cache_seqlens, int batch_size, int max_block_num,
                                     WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();

  // Each task updates the importance of a certain block
  seq_len_ = config_.block_len;
  backend->do_work_stealing_job(
      config_.layer_num * batch_size * max_block_num, nullptr,
      [&](int task_id) {
        int layer_id = task_id / (batch_size * max_block_num);
        int batch_id = (task_id / max_block_num) % batch_size;
        int block_id = task_id % max_block_num;
        // If the block is out of the sequence length, skip it. In
        // particular, the last block of the sequence that is shorter than
        // the block length should be skipped.

        if (cache_seqlens[batch_id] / config_.block_len < block_id) {
          return;
        }
        int block_idx = block_table[batch_id * max_block_num + block_id];

        std::vector<float> block_fp32(32);
        if (config_.anchor_type == AnchorType::DYNAMIC) {
          // clear anchor_
          for (int anchor_id = 0; anchor_id < 1; anchor_id++) {
            for (int head_id = 0; head_id < config_.q_head_num; head_id++) {
              for (int l = 0; l < config_.head_dim; l++) {
                anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num * config_.head_dim +
                        block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                        anchor_id * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + l] = 0;
              }
            }
          }

          // find top anchor_num importances and their corresponding
          // positions in the importance_ tensor
          // TODO: Move top_importances to the class member to avoid
          // repeated memory allocation
          std::priority_queue<std::pair<float, std::pair<int, int>>, std::vector<std::pair<float, std::pair<int, int>>>,
                              std::greater<>>
              top_importances;
          for (int head_id = 0; head_id < config_.q_head_num; head_id++) {
            for (int k = 0; k < seq_len_; k++) {
              top_importances.push(std::make_pair(GGML_FP16_TO_FP32(importance_[layer_id][block_idx][k][head_id]),
                                                  std::make_pair(block_idx, k)));
              // TODO: change to config_ item
              if (top_importances.size() > config_.anchor_num) {
                top_importances.pop();
              }
            }

            // fill anchor_

            for (int l = 0; l < config_.head_dim; l++) {
              anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num * config_.head_dim +
                      block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                      0 * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + l] = 0;
            }
            for (int k = 0; k < config_.anchor_num; k++) {
              int top_indice = top_importances.top().second.second;
              int top_block_idx = top_importances.top().second.first;

              if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
                for (int l = 0; l < config_.head_dim; l++) {
                  anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num *
                              config_.head_dim +
                          top_block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                          0 * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + l] =
                      GGML_FP32_TO_FP16(
                          GGML_FP16_TO_FP32(
                              anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num *
                                          config_.head_dim +
                                      top_block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                                      0 * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + l]) +
                          GGML_FP16_TO_FP32(k_cache_fp16_[layer_id][head_id / n_gqa_][top_block_idx]
                                                         [top_indice * config_.head_dim + l]));
                }

              } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
                for (int l = 0; l < config_.head_dim / 32; l++) {
                  block_q4_0 block =
                      k_cache_q4[layer_id][head_id / n_gqa_][top_block_idx][top_indice * config_.head_dim / 32 + l];
                  dequantize_row_q4_0(&block, block_fp32.data(), 32);
                  for (int m = 0; m < 32; m++) {
                    anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num *
                                config_.head_dim +
                            top_block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                            0 * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + l * 32 + m] =
                        GGML_FP32_TO_FP16(
                            block_fp32[m] / 4 +
                            GGML_FP16_TO_FP32(
                                anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num *
                                            config_.head_dim +
                                        top_block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                                        0 * config_.q_head_num * config_.head_dim + head_id * config_.head_dim +
                                        l * 32 + m]));
                  }
                }
              } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
                for (int l = 0; l < config_.head_dim / 32; l++) {
                  block_q8_0 block =
                      k_cache_q8[layer_id][head_id / n_gqa_][top_block_idx][top_indice * config_.head_dim / 32 + l];
                  dequantize_row_q8_0(&block, block_fp32.data(), 32);
                  for (int m = 0; m < 32; m++) {
                    anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num *
                                config_.head_dim +
                            top_block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                            0 * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + l * 32 + m] =
                        GGML_FP32_TO_FP16(
                            block_fp32[m] / 4 +
                            GGML_FP16_TO_FP32(
                                anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num *
                                            config_.head_dim +
                                        top_block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                                        0 * config_.q_head_num * config_.head_dim + head_id * config_.head_dim +
                                        l * 32 + m]));
                  }
                }
              }
              top_importances.pop();
            }
          }
        } else if (config_.anchor_type == AnchorType::BLOCK_MEAN) {
          // clear anchor_
          for (int anchor_id = 0; anchor_id < config_.anchor_num; anchor_id++) {
            for (int head_id = 0; head_id < config_.q_head_num; head_id++) {
              for (int l = 0; l < config_.head_dim; l++) {
                anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num * config_.head_dim +
                        block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                        anchor_id * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + l] = 0;
              }
            }
          }

          // fill anchor_
          if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
            for (int head_id = 0; head_id < config_.q_head_num; head_id++) {
              for (int k = 0; k < config_.block_len; k++) {
                for (int l = 0; l < config_.head_dim; l++) {
                  anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num *
                              config_.head_dim +
                          block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                          0 * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + l] =
                      GGML_FP32_TO_FP16(
                          GGML_FP16_TO_FP32(
                              anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num *
                                          config_.head_dim +
                                      block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                                      0 * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + l]) +
                          GGML_FP16_TO_FP32(
                              k_cache_fp16_[layer_id][head_id / n_gqa_][block_idx][k * config_.head_dim + l]) /
                              config_.block_len);
                }
              }
            }
          }
        } else if (config_.anchor_type == AnchorType::BLOCK_MAX) {
          // clear anchor_
          for (int anchor_id = 0; anchor_id < config_.anchor_num; anchor_id++) {
            for (int head_id = 0; head_id < config_.q_head_num; head_id++) {
              for (int l = 0; l < config_.head_dim; l++) {
                anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num * config_.head_dim +
                        block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                        anchor_id * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + l] = 0;
              }
            }
          }

          // fill anchor_
          if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
            for (int head_id = 0; head_id < config_.q_head_num; head_id++) {
              for (int k = 0; k < config_.block_len; k++) {
                for (int l = 0; l < config_.head_dim; l++) {
                  anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num *
                              config_.head_dim +
                          block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                          0 * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + l] =
                      GGML_FP32_TO_FP16(std::max(
                          GGML_FP16_TO_FP32(
                              anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num *
                                          config_.head_dim +
                                      block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                                      0 * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + l]),
                          GGML_FP16_TO_FP32(
                              k_cache_fp16_[layer_id][head_id / n_gqa_][block_idx][k * config_.head_dim + l])));
                }
              }
            }
          }
        } else if (config_.anchor_type == AnchorType::FIXED_ANCHOR) {
          // clear anchor_
          for (int anchor_id = 0; anchor_id < 1; anchor_id++) {
            for (int head_id = 0; head_id < config_.q_head_num; head_id++) {
              for (int l = 0; l < config_.head_dim; l++) {
                anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num * config_.head_dim +
                        block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                        anchor_id * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + l] = 0;
              }
            }
          }

          // fill anchor_
          if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
            int stride = config_.block_len / config_.anchor_num;
            for (int head_id = 0; head_id < config_.q_head_num; head_id++) {
              for (int k = 0, tot = 0; k < config_.block_len, tot < config_.anchor_num; k += stride, tot++) {
                for (int l = 0; l < config_.head_dim; l++) {
                  anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num *
                              config_.head_dim +
                          block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                          0 * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + l] =
                      GGML_FP32_TO_FP16(
                          GGML_FP16_TO_FP32(
                              anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num *
                                          config_.head_dim +
                                      block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                                      0 * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + l]) +
                          GGML_FP16_TO_FP32(
                              k_cache_fp16_[layer_id][head_id / n_gqa_][block_idx][k * config_.head_dim + l]) /
                              config_.anchor_num);
                }
              }
            }
          }

        } else if (config_.anchor_type == AnchorType::QUEST) {
          // clear anchor_
          for (int head_id = 0; head_id < config_.q_head_num; head_id++) {
            for (int l = 0; l < config_.head_dim; l++) {
              anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num * config_.head_dim +
                      block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                      1 * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + l] =
                  GGML_FP32_TO_FP16(std::numeric_limits<float>::max());

              anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num * config_.head_dim +
                      block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                      0 * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + l] =
                  GGML_FP32_TO_FP16(std::numeric_limits<float>::min());
            }
          }

          // fill anchor_

          if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
            for (int indice = 0; indice < seq_len_; indice++) {
              for (int head_id = 0; head_id < config_.kv_head_num; head_id++) {
                for (int l = 0; l < config_.head_dim; l++) {
                  anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num *
                              config_.head_dim +
                          block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                          0 * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + l] =
                      GGML_FP32_TO_FP16(std::max(
                          GGML_FP16_TO_FP32(k_cache_fp16_[layer_id][head_id][block_idx][indice * config_.head_dim + l]),
                          GGML_FP16_TO_FP32(
                              anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num *
                                          config_.head_dim +
                                      block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                                      0 * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + l])));

                  anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num *
                              config_.head_dim +
                          block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                          1 * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + l] =
                      GGML_FP32_TO_FP16(std::min(
                          GGML_FP16_TO_FP32(k_cache_fp16_[layer_id][head_id][block_idx][indice * config_.head_dim + l]),
                          GGML_FP16_TO_FP32(
                              anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num *
                                          config_.head_dim +
                                      block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                                      1 * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + l])));
                }
              }
            }

          } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
            for (int indice = 0; indice < seq_len_; indice++) {
              for (int head_id = 0; head_id < config_.kv_head_num; head_id++) {
                for (int l = 0; l < config_.head_dim / 32; l++) {
                  block_q4_0 block = k_cache_q4[layer_id][head_id][block_idx][indice * config_.head_dim / 32 + l];
                  dequantize_row_q4_0(&block, block_fp32.data(), 32);

                  for (int m = 0; m < 32; m++) {
                    for (int gqa_idx = 0; gqa_idx < n_gqa_; gqa_idx++) {
                      anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num *
                                  config_.head_dim +
                              block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                              0 * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + l * 32 + m] =
                          GGML_FP32_TO_FP16(std::max(
                              block_fp32[m],
                              GGML_FP16_TO_FP32(
                                  anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num *
                                              config_.head_dim +
                                          block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                                          0 * config_.q_head_num * config_.head_dim + head_id * config_.head_dim +
                                          l * 32 + m])));

                      anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num *
                                  config_.head_dim +
                              block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                              1 * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + l * 32 + m] =
                          GGML_FP32_TO_FP16(std::min(
                              block_fp32[m],
                              GGML_FP16_TO_FP32(
                                  anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num *
                                              config_.head_dim +
                                          block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                                          1 * config_.q_head_num * config_.head_dim + head_id * config_.head_dim +
                                          l * 32 + m])));
                    }
                  }
                }
              }
            }
          } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
            for (int indice = 0; indice < seq_len_; indice++) {
              for (int head_id = 0; head_id < config_.kv_head_num; head_id++) {
                for (int l = 0; l < config_.head_dim / 32; l++) {
                  block_q8_0 block = k_cache_q8[layer_id][head_id][block_idx][indice * config_.head_dim / 32 + l];
                  dequantize_row_q8_0(&block, block_fp32.data(), 32);

                  for (int m = 0; m < 32; m++) {
                    for (int gqa_idx = 0; gqa_idx < n_gqa_; gqa_idx++) {
                      anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num *
                                  config_.head_dim +
                              block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                              0 * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + l * 32 + m] =
                          GGML_FP32_TO_FP16(std::max(
                              block_fp32[m],
                              GGML_FP16_TO_FP32(
                                  anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num *
                                              config_.head_dim +
                                          block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                                          0 * config_.q_head_num * config_.head_dim + head_id * config_.head_dim +
                                          l * 32 + m])));

                      anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num *
                                  config_.head_dim +
                              block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                              1 * config_.q_head_num * config_.head_dim + head_id * config_.head_dim + l * 32 + m] =
                          GGML_FP32_TO_FP16(std::min(
                              block_fp32[m],
                              GGML_FP16_TO_FP32(
                                  anchor_[layer_id * config_.max_block_num * config_.anchor_num * config_.q_head_num *
                                              config_.head_dim +
                                          block_idx * config_.anchor_num * config_.q_head_num * config_.head_dim +
                                          1 * config_.q_head_num * config_.head_dim + head_id * config_.head_dim +
                                          l * 32 + m])));
                    }
                  }
                }
              }
            }
          }
        } else {
          assert(false);
        }
      },
      nullptr);

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  //    printf("time of calc_anchor_all_layers: %f s\n", duration.count());
}

/**
 * @brief 清零全部层有效 block 的动态 token importance。
 *
 * 仅在 anchor_type 为 DYNAMIC 时执行实际清零。函数按 (layer, batch, logical block) 并行，通过
 * block_table 定位物理 block，并把该 block 内所有 token、所有 Query head 的 importance 设为零。
 *
 * @param block_table 行优先 [batch_size, max_block_num] 逻辑到物理 block 映射。
 * @param cache_seqlens 每个 batch 的有效 token 数，用于跳过 cache 范围外的 block。
 * @param batch_size 序列数量。
 * @param max_block_num 每个序列参与遍历的 block 表宽度。
 * @param backend 用于并行清零各层和 block 的工作线程池。
 */
void KVCache::clear_importance_all_layers(int* block_table, int* cache_seqlens, int batch_size, int max_block_num,
                                          WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();

  // Each task updates the importance of a certain block
  seq_len_ = config_.block_len;
  backend->do_work_stealing_job(
      config_.layer_num * batch_size * max_block_num, nullptr,
      [&](int task_id) {
        int layer_id = task_id / (batch_size * max_block_num);
        int batch_id = (task_id / max_block_num) % batch_size;
        int block_id = task_id % max_block_num;
        // If the block is out of the sequence length, skip it. In
        // particular, the last block of the sequence that is shorter than
        // the block length should be skipped.

        if (cache_seqlens[batch_id] / config_.block_len < block_id) {
          return;
        }
        int block_idx = block_table[batch_id * max_block_num + block_id];

        if (config_.anchor_type == AnchorType::DYNAMIC) {
          // clear anchor_
          for (int head_id = 0; head_id < config_.q_head_num; head_id++) {
            for (int l = 0; l < config_.block_len; l++) {
              importance_[layer_id][block_idx][l][head_id] = 0;
            }
          }
        }
      },
      nullptr);

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  //    printf("time of clear_importance_all_layerssssss: %f s\n",
  //    duration.count());
}

/**
 * @brief 清零全部层、全部 KV head 中由 block 表引用的有效 K/V block。
 *
 * 函数按 (layer, batch, logical block, KV head) 并行。FP16 模式把 K/V 元素全部设零；Q4_0/Q8_0 模式
 * 把每个量化块的缩放因子 d 设零，使反量化结果为零。只处理 cache_seqlens 覆盖的逻辑 block。
 *
 * @param block_table 行优先 [batch_size, max_block_num] 逻辑到物理 block 映射。
 * @param cache_seqlens 每个 batch 的有效 token 数，用于确定应清零的 block 范围。
 * @param batch_size 序列数量。
 * @param max_block_num 每个序列参与遍历的 block 表宽度。
 * @param backend 用于并行清零层、block 和 KV head 的工作线程池。
 */
void KVCache::clear_kvcache_all_layers(int* block_table, int* cache_seqlens, int batch_size, int max_block_num,
                                       WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();

  // Each task updates the importance of a certain block
  seq_len_ = config_.block_len;
  backend->do_work_stealing_job(
      config_.layer_num * batch_size * max_block_num * config_.kv_head_num, nullptr,
      [&](int task_id) {
        int layer_id = task_id / (batch_size * max_block_num * config_.kv_head_num);
        int batch_id = (task_id / (max_block_num * config_.kv_head_num)) % batch_size;
        int block_id = task_id / config_.kv_head_num % max_block_num;
        int head_id = task_id % config_.kv_head_num;
        // If the block is out of the sequence length, skip it. In
        // particular, the last block of the sequence that is shorter than
        // the block length should be skipped.
        if (cache_seqlens[batch_id] / config_.block_len < block_id) {
          return;
        }
        int block_idx = block_table[batch_id * max_block_num + block_id];

        if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
          for (int l = 0; l < config_.block_len * config_.head_dim; l++) {
            k_cache_fp16_[layer_id][head_id][block_idx][l] = 0;
            v_cache_fp16_[layer_id][head_id][block_idx][l] = 0;
          }
        } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
          for (int l = 0; l < config_.block_len * config_.head_dim / 32; l++) {
            k_cache_q4[layer_id][head_id][block_idx][l].d = 0;
            v_cache_q4[layer_id][head_id][block_idx][l].d = 0;
          }
        } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
          for (int l = 0; l < config_.block_len * config_.head_dim / 32; l++) {
            k_cache_q8[layer_id][head_id][block_idx][l].d = 0;
            v_cache_q8[layer_id][head_id][block_idx][l].d = 0;
          }
        }
      },
      nullptr);

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  //    printf("time of clear_kvcache_all_layers: %f s\n", duration.count());
}

/**
 * @brief 把连续 FP16 RoPE sin/cos 表复制到 KVCache 的二维内部存储。
 *
 * 内部表随后可由 block Attention kernel 在需要在线旋转 Key 时按位置读取。
 *
 * @param sin 输入 FP16 sin 表，布局为 [seqlen, head_dim]。
 * @param cos 输入 FP16 cos 表，布局为 [seqlen, head_dim]。
 * @param seqlen 要复制的位置数量，不得超过 BlockResize() 预分配的 token 容量。
 */
void KVCache::get_sincos(ggml_fp16_t* sin, ggml_fp16_t* cos, int seqlen) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();

  const uint16_t* sin_data = const_cast<const uint16_t*>(sin);
  const uint16_t* cos_data = const_cast<const uint16_t*>(cos);

  for (int i = 0; i < seqlen; i++) {
    for (int j = 0; j < config_.head_dim; j++) {
      sin_[i][j] = sin_data[i * config_.head_dim + j];
      cos_[i][j] = cos_data[i * config_.head_dim + j];
    }
  }

  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  printf("time of get_sincos: %f s\n", duration.count());
}

/**
 * @brief 将一个连续 FP32 向量原地乘以标量。
 *
 * 实现优先使用 Accelerate 或 GGML SIMD 向量指令，并用标量循环处理 SIMD 尾部；无加速后端时退化为
 * 普通循环。Attention 的跨 block LSE 归并使用该函数重缩放局部输出。
 *
 * @param n 向量 y 中要缩放的元素数量。
 * @param y 输入/输出 FP32 向量。
 * @param v 乘到每个元素上的缩放系数。
 */
void ggml_vec_scale_f32(const int n, float* y, const float v) {
#if defined(GGML_USE_ACCELERATE)
  vDSP_vsmul(y, 1, &v, y, 1, n);
#elif defined(GGML_SIMD)
  const int np = (n & ~(GGML_F32_STEP - 1));

  GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

  GGML_F32_VEC ay[GGML_F32_ARR];

  for (int i = 0; i < np; i += GGML_F32_STEP) {
    for (int j = 0; j < GGML_F32_ARR; j++) {
      ay[j] = GGML_F32_VEC_LOAD(y + i + j * GGML_F32_EPR);
      ay[j] = GGML_F32_VEC_MUL(ay[j], vx);

      GGML_F32_VEC_STORE(y + i + j * GGML_F32_EPR, ay[j]);
    }
  }

  // leftovers
  for (int i = np; i < n; ++i) {
    y[i] *= v;
  }
#else
  // scalar
  for (int i = 0; i < n; ++i) {
    y[i] *= v;
  }
#endif
}
