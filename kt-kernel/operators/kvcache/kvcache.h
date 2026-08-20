/**
 * @说明         :
 * @作者         : Jianwei Dong
 * @日期         : 2024-08-26 22:47:06
 * @版本         : 1.0.0
 * @最后编辑者   : Jianwei Dong
 * @最后编辑时间 : 2024-08-26 22:47:06
 * @版权所有 (c) 2024 KVCache.AI，保留所有权利。
 **/

#ifndef CPUINFER_OPERATOR_KVCACHE_H
#define CPUINFER_OPERATOR_KVCACHE_H

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

#include "../../cpu_backend/worker_pool.h"
#include "llama.cpp/ggml-common.h"
#include "llama.cpp/ggml-quants.h"
#include "llama.cpp/ggml.h"

#define CHUNK_SIZE 32

/**
 * @brief 将 ggml_type 枚举值转换为对应的字符串表示。
 *
 * 此函数为给定的 ggml_type 枚举值提供便于阅读的字符串表示，可用于日志记录、
 * 调试或在用户界面中显示信息。
 *
 * @param type 要转换的 ggml_type 枚举值。
 * @return 枚举值对应的字符串表示。
 */
std::string ggml_type_to_string(ggml_type type);

/**
 * @enum AnchorType
 * @brief 定义 Attention 机制使用的锚点类型。
 *
 * 此枚举指定 Attention 机制可使用的不同锚点类型，包括固定锚点、动态锚点，
 * 以及 QUEST、BLOCK_MEAN、BLOCK_MAX 等特殊锚点。
 */
enum AnchorType {
  FIXED_ANCHOR, /**< 始终不变的固定锚点。 */
  DYNAMIC,      /**< 可随时间变化的动态锚点。 */
  QUEST,        /**< QUEST（查询与嵌入空间变换）使用的特殊锚点类型。 */
  BLOCK_MEAN,   /**< 根据一个数据块的均值生成的锚点。 */
  BLOCK_MAX     /**< 根据一个数据块内最大值生成的锚点。
                 */
};

/**
 * @brief 将 AnchorType 枚举值转换为对应的字符串表示。
 *
 * 此函数为给定的 AnchorType 枚举值提供便于阅读的字符串表示，可用于日志记录、
 * 调试或在用户界面中显示信息。
 *
 * @param anchor_type 要转换的 AnchorType 枚举值。
 * @return 枚举值对应的字符串表示。
 */
std::string AnchorTypeToString(AnchorType anchor_type);

/**
 * @enum RetrievalType
 * @brief 定义 Attention 机制中的检索策略类型。
 *
 * 此枚举指定 Attention 机制可使用的不同检索策略，包括层级检索、KV Head 级检索
 * 和 Query Head 级检索。
 */
enum RetrievalType {
  LAYER,  /**< 按层检索。 */
  KVHEAD, /**< 按 KV Head 检索。 */
  QHEAD   /**< 按 Query Head 检索。 */
};

/**
 * @brief 将 RetrievalType 枚举值转换为对应的字符串表示。
 *
 * 此函数为给定的 RetrievalType 枚举值提供便于阅读的字符串表示，可用于日志记录、
 * 调试或在用户界面中显示信息。
 *
 * @param retrieval_type 要转换的 RetrievalType 枚举值。
 * @return 枚举值对应的字符串表示。
 */
std::string RetrievalTypeToString(RetrievalType retrieval_type);

/**
 * @struct KVCacheConfig
 * @brief KV Cache 的配置结构体。
 *
 * 此结构体保存创建和管理 Attention 机制所用 KV Cache 的配置参数，包括层数、
 * Head 数、每个 Head 的维度、Block 长度、锚点信息以及内存相关设置。
 */
struct KVCacheConfig {
  int layer_num;   /**< 模型的层数。 */
  int kv_head_num; /**< KV Cache 中的 Head 数。 */
  int q_head_num;  /**< Query 的 Head 数。 */
  int head_dim;    /**< 每个 Head 的维度。 */
  int block_len;   /**< Cache 中每个 Block 的长度。 */
  int anchor_num;  /**< Attention 使用的锚点数量。 */

  ggml_type kv_type; /**< KV Cache 的数据类型，例如 fp16、q8_0。 */

  // 控制预分配内存的大小
  int max_block_num;  /**< 可分配的最大 Block 数量。 */
  int max_batch_size; /**< 可处理的最大 Batch Size。 */
  int max_thread_num; /**< 可使用的最大线程数。 */

  AnchorType anchor_type;       /**< Attention 机制使用的锚点类型。 */
  RetrievalType retrieval_type; /**< Cache 使用的检索策略类型。 */

  int layer_step;   /**< 层之间的步长。 */
  int token_step;   /**< Token 之间的步长。 */
  int layer_offset; /**< 层偏移量。 */

  /**
   * @brief KVCacheConfig 的默认构造函数。
   *
   * 使用默认值初始化配置。此构造函数不显式初始化任何成员变量。
   */
  KVCacheConfig() = default;

  /**
   * @brief KVCacheConfig 的带参构造函数。
   *
   * 使用指定值初始化全部成员变量。
   *
   * @param layer_num 模型的层数。
   * @param kv_head_num KV Cache 中的 Head 数。
   * @param q_head_num Query 的 Head 数。
   * @param head_dim 每个 Head 的维度。
   * @param block_len Cache 中每个 Block 的长度。
   * @param anchor_num Attention 使用的锚点数量。
   * @param anchor_type Attention 机制使用的锚点类型。
   * @param kv_type KV Cache 的数据类型，例如 fp16、q8_0。
   * @param retrieval_type Cache 使用的检索策略类型。
   * @param layer_step 层之间的步长。
   * @param token_step Token 之间的步长。
   * @param layer_offset 层偏移量。
   * @param max_block_num 可分配的最大 Block 数量。
   * @param max_batch_size 可处理的最大 Batch Size。
   * @param max_thread_num 可使用的最大线程数。
   */
  KVCacheConfig(int layer_num, int kv_head_num, int q_head_num, int head_dim, int block_len, int anchor_num,
                AnchorType anchor_type, ggml_type kv_type, RetrievalType retrieval_type, int layer_step, int token_step,
                int layer_offset, int max_block_num, int max_batch_size, int max_thread_num);
};

/**
 * @class KVCache
 * @brief 管理 Attention 机制使用的 KV Cache。
 *
 * KVCache 类提供管理 KV Cache 的功能，包括调整 Cache 大小、读取配置参数以及
 * 更新内部状态。该类通常用于 Transformer 模型，存储并管理历史 Key、Value 状态，
 * 以便高效计算 Attention。
 */
class KVCache {
 public:
  /**
   * @brief 使用给定配置构造 KVCache 对象。
   *
   * 使用指定配置参数初始化 KVCache，包括层数、Head 数、Head 维度及其他相关设置。
   *
   * @param config 包含各项初始化参数的配置对象。
   */
  KVCache(KVCacheConfig config);

  /**
   * @brief 调整 Cache 使用的线程数。
   *
   * 调整 Cache 可使用的线程数量，从而根据当前工作负载或系统资源动态配置并行处理能力。
   *
   * @param thread_num 调整后的线程数。
   */
  void ThreadResize(int thread_num);

  /**
   * @brief 调整 Cache 管理的 Batch Size。
   *
   * 调整 Cache 可处理的 Batch Size。当输入 Batch Size 动态变化时，可用此函数
   * 相应地重新配置 Cache。
   *
   * @param batch_size 调整后的 Batch Size。
   */
  void BatchResize(int batch_size);

  /**
   * @brief 调整 Cache 管理的 Block 数量。
   *
   * 调整 Cache 可管理的 Block 数量，从而根据当前序列长度或其他因素动态配置
   * Block 结构。
   *
   * @param block_num 调整后的 Block 数量。
   */
  void BlockResize(int block_num);

  /**
   * @brief 获取 Cache 的层数。
   *
   * @return Cache 配置的层数。
   */
  int get_layer_num() { return config_.layer_num; }

  /**
   * @brief 获取 Cache 的 KV Head 数量。
   *
   * @return Cache 配置的 KV Head 数量。
   */
  int get_kv_head_num() { return config_.kv_head_num; }

  /**
   * @brief 获取 Cache 的 Query Head 数量。
   *
   * @return Cache 配置的 Query Head 数量。
   */
  int get_q_head_num() { return config_.q_head_num; }

  /**
   * @brief 获取 Cache 中每个 Head 的维度。
   *
   * @return 每个 Head 的维度。
   */
  int get_head_dim() { return config_.head_dim; }

  /**
   * @brief 获取 Cache 中每个 Block 的长度。
   *
   * @return 每个 Block 的长度。
   */
  int get_block_len() { return config_.block_len; }

  /**
   * @brief 获取指定层的 Block 数量。
   *
   * @param layer_id 要查询 Block 数量的层 ID。
   * @return 指定层中的 Block 数量。
   */
  int get_block_num(int layer_id) { return past_block_num_[layer_id]; }

  /**
   * @brief 获取 Cache 中的锚点数量。
   *
   * @return Cache 配置的锚点数量。
   */
  int get_anchor_num() { return config_.anchor_num; }

  /**
   * @brief 获取 Cache 的总长度。
   *
   * @return Cache 的总长度。
   */
  int get_cache_total_len() { return cache_total_len_; }

  /**
   * @brief 获取 Cache 中的 Block 总数。
   *
   * 根据 Cache 总长度和 Block 长度配置计算并返回 Cache 中的 Block 总数。
   *
   * @return Cache 中的 Block 总数。
   */
  int get_cache_total_block_num() { return (cache_total_len_ + config_.block_len - 1) / config_.block_len; }

  /**
   * @brief 更新 Cache 的总长度。
   *
   * 为 Cache 设置新的总长度，以便在运行时动态调整 Cache 大小。
   *
   * @param cache_total_len Cache 的新总长度。
   */
  void update_cache_total_len(int cache_total_len) { cache_total_len_ = cache_total_len; }
  void attn(const ggml_fp16_t* q_in, ggml_fp16_t* output, float* attn_lse, int layer_idx, int generate_token_idx,
            int q_len, int batch_size, int max_block_num, int* block_table, int* cache_seqlens, int pick_block_num,
            int init_block_num, int local_block_num, WorkerPool* backend);

  void update_kvcache_one_block_fp16(const ggml_fp16_t* k_in, const ggml_fp16_t* v_in, int layer_id, int block_idx,
                                     WorkerPool* backend);

  void get_kvcache_one_block_fp16(ggml_fp16_t* k_in, ggml_fp16_t* v_in, int layer_id, int block_idx,
                                  WorkerPool* backend);

  void update_importance_one_block(const ggml_fp16_t* importance, int layer_id, int block_idx, WorkerPool* backend);
  void get_importance_one_block(ggml_fp16_t* importance, int layer_id, int block_idx, WorkerPool* backend);

  void get_anchor_one_block(ggml_fp16_t* anchor, int layer_id, int block_idx, WorkerPool* backend);

  void update_anchor_one_block(const ggml_fp16_t* anchor, int layer_id, int block_idx, WorkerPool* backend);

  void calc_anchor_all_layers(int* block_table, int* cache_seqlens, int batch_size, int max_block_num,
                              WorkerPool* backend);

  void load_kvcache(std::string tensor_file_path, WorkerPool* backend);
  void dump_kvcache(int* block_table, int cache_total_len, std::string tensor_file_path, WorkerPool* backend);

  void get_and_update_kvcache_fp16(ggml_fp16_t* k_in, ggml_fp16_t* v_in, int layer_id, int* block_table, int batch_size,
                                   int max_block_num, int* cache_seqlens, int q_len, WorkerPool* backend);

  void get_kvcache_fp16(ggml_fp16_t* k_in, ggml_fp16_t* v_in, int layer_id, int* block_table, int batch_size,
                        int max_block_num, int* cache_seqlens, WorkerPool* backend);

  void update_kvcache_fp16(const ggml_fp16_t* k_in, const ggml_fp16_t* v_in, int layer_id, int* block_table,
                           int batch_size, int max_block_num, int* cache_seqlens, int q_len, WorkerPool* backend);

  void update_importance(const ggml_fp16_t* importance, int layer_id, int* block_table, int batch_size,
                         int max_block_num, int* offset, int width, WorkerPool* backend);

  void attn_with_kvcache(const ggml_fp16_t* q_in, const ggml_fp16_t* k_in, const ggml_fp16_t* v_in, ggml_fp16_t* output,
                         float* attn_lse, int layer_idx, int generate_token_idx, int q_len, int batch_size,
                         int max_block_num, int* block_table, int* cache_seqlens, int topk, int local,
                         WorkerPool* backend);

  void clear_importance_all_layers(int* block_table, int* cache_seqlens, int batch_size, int max_block_num,
                                   WorkerPool* backend);

  void clear_kvcache_all_layers(int* block_table, int* cache_seqlens, int batch_size, int max_block_num,
                                WorkerPool* backend);

  void get_sincos(ggml_fp16_t* sin, ggml_fp16_t* cos, int seqlen);

  void get_attn_sparsity(const ggml_fp16_t* q_in, float* attn_sparsity, int layer_idx, int generate_token_idx,
                         int q_len, int batch_size, int max_block_num, int* block_table, int* cache_seqlens,
                         int* block_table_origin, int* cache_seqlens_origin, int max_block_num_origin, int topk,
                         int local, WorkerPool* backend);

  void get_all_kvcache_one_layer(int layer_id, ggml_fp16_t* k_in, ggml_fp16_t* v_in, WorkerPool* backend);

 private:
  // 持久化数据
  KVCacheConfig config_;
  int n_gqa_;                             // q_head_num / kv_head_num
  int cache_total_len_;                   // Cache 中的 Token 数量
  std::vector<uint64_t> past_block_num_;  // [layer_num]

  // 这是 KVCache 在 CPU 内存中的持久化 buffer。
  // 外层使用四级 vector，统一索引为 [layer][kv_head][physical_block][block_data]；因此每个物理 block 的
  // block_data 连续，但不同 block、KV head 和 layer 之间不保证物理连续。
  //
  // K block 采用 token-major 布局：
  //   FP16: K[token, dim] -> token * head_dim + dim
  //   Q4/Q8: K[token, dim_group] -> token * (head_dim / 32) + dim_group
  // V block 为了让 Attention probability x V 使用连续内存，采用转置后的 channel-major 布局：
  //   FP16: V[dim, token] -> dim * block_len + token
  //   Q4/Q8: V[dim, token_group] -> dim * (block_len / 32) + token_group
  // 其中一个 block_q4_0/block_q8_0 表示连续 32 个元素。逻辑 token block 通过 block_table 映射到
  // physical_block；未量化时每个 K/V block 都包含 block_len * head_dim 个 FP16 元素。
  std::vector<std::vector<std::vector<std::vector<block_q4_0>>>>
      k_cache_q4;  // [layer_num][kv_head_num][physical_block][block_len * (head_dim / 32)]
  std::vector<std::vector<std::vector<std::vector<block_q4_0>>>>
      v_cache_q4;  // [layer_num][kv_head_num][physical_block][head_dim * (block_len / 32)]
  std::vector<std::vector<std::vector<std::vector<block_q8_0>>>>
      k_cache_q8;  // [layer_num][kv_head_num][physical_block][block_len * (head_dim / 32)]
  std::vector<std::vector<std::vector<std::vector<block_q8_0>>>>
      v_cache_q8;  // [layer_num][kv_head_num][physical_block][head_dim * (block_len / 32)]

  std::vector<std::vector<std::vector<std::vector<ggml_fp16_t>>>>
      k_cache_fp16_;  // [layer_num][kv_head_num][physical_block][block_len * head_dim]
  std::vector<std::vector<std::vector<std::vector<ggml_fp16_t>>>>
      v_cache_fp16_;  // [layer_num][kv_head_num][physical_block][head_dim * block_len]

  std::vector<std::vector<std::vector<std::vector<ggml_fp16_t>>>> importance_;  // [layer_num, past_block_num,
                                                                                // block_len, attention_head_num]

  std::vector<ggml_fp16_t> anchor_;  // [layer_num * past_block_num * anchor_num *
                                     // attention_head_num * head_dim]

  // 运行时数据
  int64_t layer_id_;
  int64_t block_idx_;
  int* block_table_;
  uint64_t block_num_;
  int max_block_num_after_retrieval_;

  // 旋转位置编码
  std::vector<std::vector<ggml_fp16_t>> sin_;  // [seq_len, head_dim]
  std::vector<std::vector<ggml_fp16_t>> cos_;  // [seq_len, head_dim]

  // 更新与读取
  int seq_len_;
  uint16_t* k_scales_;         // q4_0
  uint8_t* k_in_;              // q4_0
  uint16_t* v_scales_;         // q4_0
  uint8_t* v_in_;              // q4_0
  uint16_t* k_data_;           // fp16
  uint16_t* v_data_;           // fp16
  uint16_t* importance_data_;  // fp16
  uint16_t* anchor_data_;      // fp16

  // 稀疏度 = (sigma(block lse / lse))
  std::vector<std::vector<std::vector<float>>> block_lse_;  // [batch_size, max_block_num, q_head_num]
  std::vector<std::vector<float>> attn_sparsity_;           // [batch_size, q_head_num]

  // Attention 运行时数据
  std::vector<std::vector<float>> avg_q;  // [batch_size, q_head_num * head_dim]

  std::vector<std::vector<ggml_fp16_t>> avg_q_fp16;  // [batch_size, q_head_num * head_dim]
  std::vector<std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<>>>
      top_similar_block_;

  std::vector<std::vector<float>> block_similar_;
  std::vector<std::vector<std::vector<float>>> block_similar_kv_head_;
  std::vector<std::vector<std::vector<float>>> block_similar_q_head_;

  std::vector<int> cache_seqlens_;                // [batch_size]
  std::vector<int> selected_blocks_num_history_;  // [layer_num // layer_step]

  std::vector<std::vector<std::vector<int>>> selected_blocks_history_;
  // [layer_num // layer_step, batch_size, max_block_num]

  std::vector<std::vector<std::vector<std::vector<int>>>>
      selected_blocks_history_kvhead_;  // [layer_num // layer_step,
                                        // batch_size, max_block_num,
                                        // kv_head_num]

  std::vector<std::vector<int>> block_table_before_retrieval_;  // [batch_size, max_block_num]
  std::vector<std::vector<int>> block_table_after_retrieval_;   // [batch_size, pick_block_num]

  std::vector<std::vector<std::vector<int>>> block_table_before_retrieval_qhead_;  // [batch_size, max_block_num,
                                                                                   // q_head_num]
  std::vector<std::vector<std::vector<int>>> block_table_after_retrieval_qhead_;   // [batch_size, pick_block_num,
                                                                                   // q_head_num]

  std::vector<std::vector<std::vector<int>>> block_table_before_retrieval_kvhead_;  // [batch_size, max_block_num,
                                                                                    // kv_head_num]
  std::vector<std::vector<std::vector<int>>> block_table_after_retrieval_kvhead_;   // [batch_size, pick_block_num,
                                                                                    // kv_head_num]

  std::vector<std::vector<std::unique_ptr<std::mutex>>> mutex_;  // [batch_size, kv_head_num]
  std::vector<std::vector<std::vector<block_q8_0>>> q_q8_0_;     // [batch_size, kv_head_num, n_gqa * head_dim / QK8_0]
  std::vector<std::vector<std::vector<float>>> q_fp32_;          // [batch_size, kv_head_num, n_gqa * head_dim]

  std::vector<std::vector<std::vector<float>>> output_fp32_;  // [batch_size, kv_head_num, n_gqa * head_dim]
  std::vector<std::vector<std::vector<float>>> attn_lse_;     // [batch_size, kv_head_num, n_gqa]

  std::vector<std::pair<int, int>> thread_cur_head_idx_;  // [thread_num]

  std::vector<std::vector<block_q8_0>> thread_local_output_q8_0_;  // [thread_num, n_gqa * head_dim / QK8_0]
  std::vector<std::vector<float>> thread_local_attn_score_;        // [thread_num, n_gqa * block_len]
  std::vector<std::vector<float>> thread_local_output_fp32_;       // [thread_num, n_gqa * head_dim]
  std::vector<std::vector<float>> thread_local_attn_lse_;          // [thread_num, n_gqa]
  std::vector<std::vector<float>> thread_local_cur_output_fp32_;   // [thread_num, n_gqa * head_dim]
  std::vector<std::vector<float>> thread_local_cur_attn_lse_;      // [thread_num, n_gqa]
  std::vector<std::vector<uint8_t>> thread_local_attn_mask_;       // [thread_num, block_len // 8]
  std::vector<std::vector<char>> thread_local_draft_;              // [thread_num, 2 * n_gqa * block_len + 6 * n_gqa *
                                                                   // head_dim + 2 * block_len * head_dim]

  // 临时空间
  std::vector<float> q_fp32;  // [n_gqa * head_dim]

  void quantize_q_(const uint16_t* q_in_data, int batch_size);
  void attn_initialize_layer_(int batch_size, int layer_idx, int* block_table, int& max_block_num, int* cache_seqlens);
  void attn_initialize_kvhead_(int batch_size, int layer_idx, int* block_table, int& max_block_num, int* cache_seqlens);
  void retrieval_kvcache_layer_(const uint16_t* q_in_data, int init_block_num, int local_block_num, int pick_block_num,
                                int q_len, int generate_token_idx, int batch_size, int layer_idx, int* cache_seqlens,
                                int& max_block_num, WorkerPool* backend);
  void retrieval_kvcache_kvhead_(const uint16_t* q_in_data, int init_block_num, int local_block_num, int pick_block_num,
                                 int q_len, int generate_token_idx, int batch_size, int layer_idx, int* cache_seqlens,
                                 int& max_block_num, WorkerPool* backend);

  void calculate_block_similarity_layer_(const uint16_t* q_in_data, int batch_size, int layer_idx, int q_len,
                                         int max_block_num, int* cache_seqlens, int init_block_num, int local_block_num,
                                         int pick_block_num, WorkerPool* backend);
  void calculate_block_similarity_kvhead_(const uint16_t* q_in_data, int batch_size, int layer_idx, int q_len,
                                          int max_block_num, int* cache_seqlens, int init_block_num,
                                          int local_block_num, int pick_block_num, WorkerPool* backend);

  void select_block_layer_(int batch_size, int layer_idx, int max_block_num, int init_block_num, int local_block_num,
                           int pick_block_num);
  void select_block_kvhead_(int batch_size, int layer_idx, int max_block_num, int init_block_num, int local_block_num,
                            int pick_block_num);

  void calculate_sparsity_layer_(const uint16_t* q_in_data, float* attn_sparsity, int batch_size, int max_block_num,
                                 int* block_table, int* cache_seqlens, WorkerPool* backend);
  void calculate_sparsity_kvhead_(const uint16_t* q_in_data, float* attn_sparsity, int batch_size, int max_block_num,
                                  int* block_table, int* cache_seqlens, WorkerPool* backend);

  void attention_kvhead_(const uint16_t* q_in_data, ggml_fp16_t* output, float* attn_lse, int batch_size,
                         WorkerPool* backend);
  void attention_layer_(const uint16_t* q_in_data, ggml_fp16_t* output, float* attn_lse, int batch_size,
                        WorkerPool* backend);

  /**
   * @brief 使用 KV Cache 计算单个 Block 的 Attention。
   *
   * 此函数使用 KV Cache 对单个 Block 执行 Attention 计算。函数支持多种 Q、K、V Cache
   * 数据类型，并提供量化选项。函数内部不执行任何动态内存分配，因此所需 Buffer 必须
   * 由调用方预先分配。
   *
   * @param head_dim Head 的维度。
   * @param bsz Batch Size。
   * @param q_type Q 的 GGML 数据类型，仅支持 fp16 和 q8_0。
   * @param q 指向 Q Tensor [bsz, head_dim]。量化始终沿 head_dim 维度进行，大小必须为
   *          bsz * head_dim/32 * qtype_size；若 head_dim % 32 != 0，则报错。
   * @param past_kv_len 历史 KV Cache 的长度。
   * @param past_kv_offset 历史 KV Cache 中的偏移量。
   * @param is_full_attn 是否使用全量 Attention；true 表示 Mask 全为 1。
   * @param attn_mask 指向 Attention Mask [bsz, past_kv_len]。若 is_full_attn = false，
   *                  则传入表示 Mask 的位矩阵。
   * @param k_type K Cache 的 GGML 数据类型，仅支持 fp16、q4_0 和 q8_0。
   * @param k_quant_type K Cache 的量化类型：0 表示 per_token，1 表示 per_channel，
   *                     其他值会报错。
   * @param k_cache 指向 K Cache Tensor [seq_len, head_dim]。quant_type == 0 时，
   *                head_dim % 32 必须为 0；quant_type == 1 时，seq_len % 32 必须为 0。
   * @param num_k_anchor K 锚点数量。num_k_anchor == 0 表示没有锚点。
   * @param k_cache_anchors 指向 K Cache 锚点 [num_k_anchor, head_dim]，
   *                        k_anchor_type 必须为 fp16。
   * @param k_cache_anchor_pos 指向 K Cache 锚点位置。每个 Token 与其前方距离最近的锚点位置关联。
   * @param v_type V Cache 的 GGML 数据类型。
   * @param v_quant_type V Cache 的量化类型。
   * @param v_cache 指向 V Cache Tensor [head_dim, seq_len]。
   * @param num_v_anchor V 锚点数量。
   * @param v_cache_anchors 指向 V Cache 锚点。
   * @param v_cache_anchor_pos 指向 V Cache 锚点位置。
   * @param attn_score 预分配的 Attention Score Buffer [bsz, past_kv_len]。
   * @param output 输出 Tensor [bsz, head_dim]，数据类型与 q_type 相同。
   * @param lse 预分配的 Buffer [bsz]，用于保存 Attention Score 的 log-sum-exp。
   * @param draft 预分配的临时 Buffer，其容量应至少为
   *              (2 * bsz * past_kv_len + 6 * bsz * head_dim + 2 *
   *              past_kv_len * head_dim + past_kv_len * head_dim / 32) 字节。
   * @param rotary_angle 指向旋转角 Tensor。
   * @param rotary_cos 指向旋转位置编码的余弦值。
   * @param rotary_sin 指向旋转位置编码的正弦值。
   */
  void attn_with_kvcache_one_block_(int head_dim, int bsz,
                                    ggml_type q_type,  // Q 的 GGML 数据类型，仅支持 fp16 和 q8_0
                                    // [bsz, head_dim]
                                    // 量化始终沿 head_dim 维度进行（per_token）。若 head_dim % 32 != 0，
                                    // 则报错。大小必须为 bsz *
                                    // head_dim/32 * qtype_size.
                                    const void* q,

                                    int past_kv_len, int past_kv_offset,
                                    bool is_full_attn,  // true 表示 Mask 全为 1
                                    // 若 is_full_attn = false，则传入表示 Mask 的位矩阵。
                                    // [bsz, past_kv_len]
                                    const uint8_t* attn_mask,

                                    ggml_type k_type,  // K Cache 的 GGML 数据类型，仅支持 fp16、
                                                       // q4_0, q8_0
                                    int k_quant_type,  // 0 表示 per_token，1 表示 per_channel，其他值
                                                       // 会报错
                                    // [seq_len, head_dim]
                                    // quant_type == 0 时，head_dim % 32 必须为 0。
                                    // quant_type == 1 时，seq_len % 32 必须为 0。
                                    const void* k_cache,

                                    // k_anchor_type 必须为 fp16
                                    int num_k_anchor,  // num_k_anchor == 0 表示没有锚点
                                    // [num_k_anchor, head_dim]
                                    const void* k_cache_anchors,
                                    // 每个 Token 与其前方距离最近的锚点位置关联，并保持相同距离。
                                    const int* k_cache_anchor_pos,

                                    // v_cache 与 k_cache 类似
                                    ggml_type v_type, int v_quant_type,
                                    // [head_dim, seq_len]
                                    const void* v_cache, int num_v_anchor, const void* v_cache_anchors,
                                    const int* v_cache_anchor_pos,

                                    // 为中间计算预分配的 Buffer [bsz, past_kv_len]；
                                    // 此函数内部不调用 malloc。
                                    float* attn_score,

                                    // 输出：[bsz, head_dim]，数据类型与 q_type 相同
                                    void* output,
                                    // [bsz]
                                    float* lse,

                                    // 预分配且容量足够的临时 Buffer：
                                    // (2 * bsz * past_kv_len + 6 * bsz * head_dim + 2 * past_kv_len *
                                    // head_dim + past_kv_len * head_dim / 32) 字节。
                                    void* draft,

                                    // 在线应用旋转位置编码
                                    const int* rotary_angle, const void* rotary_cos, const void* rotary_sin
                                    // rotary_cos=None,
                                    // rotary_sin=None,
                                    // cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
                                    // cache_batch_idx: Optional[torch.Tensor] = None,
                                    // rotary_interleaved=True,

                                    // // 暂不支持
                                    // window_size=(-1, -1),  # -1 表示无限上下文窗口
                                    // alibi_slopes=None,
  );
};

/**
 * @brief 使用给定标量缩放 float32 向量。
 *
 * 此函数将输入向量 `y` 的每个元素乘以标量 `v`。如果可用，会采用平台专用优化，
 * 例如 Apple Accelerate 框架或 SIMD 指令；若没有可用的专用优化，则回退到简单的
 * 标量乘法循环。
 *
 * @param n 向量 `y` 中的元素数量。
 * @param y 要缩放的输入向量，结果写回同一向量。
 * @param v 用于缩放向量的标量值。
 */
void ggml_vec_scale_f32(const int n, float* y, const float v);
#endif
