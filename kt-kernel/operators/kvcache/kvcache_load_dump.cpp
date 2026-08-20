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
#include <fstream>
#include <iostream>

#include "kvcache.h"

/**
 * @brief 从二进制文件恢复完整的 KV cache、anchor 和 token importance。
 *
 * 文件开头保存有效 token 总数，随后依次保存全部 anchor、各层各 KV head 的有效 K/V block，以及各层
 * 的 token importance。函数根据有效 token 数恢复每层 past_block_num_，并按当前 KVCacheConfig 的数据
 * 类型和预分配形状读取数据；因此加载文件时必须使用与写出时一致的模型和 cache 配置。
 *
 * @param tensor_file_path 要读取的二进制 KV cache 文件路径。
 * @param backend 预留的工作线程池参数；当前实现采用串行文件读取，未使用该参数。
 */
void KVCache::load_kvcache(std::string tensor_file_path, WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();
  std::ifstream ifs_tensor(tensor_file_path, std::ios::binary);
  if (!ifs_tensor) {
    throw std::runtime_error("Failed to open tensor file");
  }
  ifs_tensor.read(reinterpret_cast<char*>(&cache_total_len_), sizeof(cache_total_len_));
  int past_block_num = (cache_total_len_ + config_.block_len - 1) / config_.block_len;
  printf("cache_total_len: %d, past_block_num: %d\n", cache_total_len_, past_block_num);
  for (int i = 0; i < config_.layer_num; ++i) {
    past_block_num_[i] = past_block_num;
  }
  ifs_tensor.read(reinterpret_cast<char*>(anchor_.data()), anchor_.size() * sizeof(ggml_fp16_t));
  for (int i = 0; i < config_.layer_num; ++i) {
    for (int j = 0; j < config_.kv_head_num; ++j) {
      for (int k = 0; k < past_block_num_[i]; ++k) {
        if (config_.kv_type == GGML_TYPE_F16) {
          ifs_tensor.read(reinterpret_cast<char*>(k_cache_fp16_[i][j][k].data()),
                          k_cache_fp16_[i][j][k].size() * sizeof(ggml_fp16_t));
          ifs_tensor.read(reinterpret_cast<char*>(v_cache_fp16_[i][j][k].data()),
                          v_cache_fp16_[i][j][k].size() * sizeof(ggml_fp16_t));
        } else if (config_.kv_type == GGML_TYPE_Q4_0) {
          ifs_tensor.read(reinterpret_cast<char*>(k_cache_q4[i][j][k].data()),
                          k_cache_q4[i][j][k].size() * sizeof(block_q4_0));
          ifs_tensor.read(reinterpret_cast<char*>(v_cache_q4[i][j][k].data()),
                          v_cache_q4[i][j][k].size() * sizeof(block_q4_0));
        }
      }
    }
    for (int k = 0; k < past_block_num_[i]; ++k) {
      for (int l = 0; l < config_.block_len; l++) {
        ifs_tensor.read(reinterpret_cast<char*>(importance_[i][k][l].data()),
                        importance_[i][k][l].size() * sizeof(ggml_fp16_t));
      }
    }
  }
  ifs_tensor.close();
  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  printf("time of load: %f s\n", diff.count());
}
/**
 * @brief 按逻辑 block 顺序把 KV cache、anchor 和 token importance 写入二进制文件。
 *
 * 函数先写有效 token 总数和完整 anchor 数组，再通过 block_table 将逻辑 block 映射为物理 block，按层、
 * KV head 和 block 顺序写出 K/V，最后以相同逻辑顺序写出 importance。输出可由 load_kvcache() 在配置
 * 完全一致的 KVCache 实例中恢复。
 *
 * @param block_table 一维逻辑到物理 block 映射；至少包含覆盖 cache_total_len 的 block 数。
 * @param cache_total_len 要持久化的有效 token 总数，用于计算应写出的 block 数和尾块范围。
 * @param tensor_file_path 输出二进制文件路径；文件存在时会被覆盖。
 * @param backend 预留的工作线程池参数；当前实现采用串行文件写入，未使用该参数。
 */
void KVCache::dump_kvcache(int* block_table, int cache_total_len, std::string tensor_file_path, WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();
  std::ofstream ofs(tensor_file_path, std::ios::binary);
  printf("dump_kvcache: %s\n", tensor_file_path.c_str());
  if (!ofs.is_open()) {
    std::cerr << "Cannot open file " << tensor_file_path << std::endl;
    return;
  }
  ofs.write(reinterpret_cast<const char*>(&cache_total_len), sizeof(cache_total_len));
  int past_block_num = (cache_total_len + config_.block_len - 1) / config_.block_len;
  printf("cache_total_len: %d, past_block_num: %d\n", cache_total_len, past_block_num);
  ofs.write(reinterpret_cast<const char*>(anchor_.data()), anchor_.size() * sizeof(ggml_fp16_t));
  for (int i = 0; i < config_.layer_num; ++i) {
    for (int j = 0; j < config_.kv_head_num; ++j) {
      for (int k = 0; k < past_block_num; ++k) {
        int block_idx = block_table[k];
        if (config_.kv_type == GGML_TYPE_F16) {
          ofs.write(reinterpret_cast<const char*>(k_cache_fp16_[i][j][block_idx].data()),
                    k_cache_fp16_[i][j][block_idx].size() * sizeof(ggml_fp16_t));
          ofs.write(reinterpret_cast<const char*>(v_cache_fp16_[i][j][block_idx].data()),
                    v_cache_fp16_[i][j][block_idx].size() * sizeof(ggml_fp16_t));

        } else if (config_.kv_type == GGML_TYPE_Q4_0) {
          ofs.write(reinterpret_cast<const char*>(k_cache_q4[i][j][block_idx].data()),
                    k_cache_q4[i][j][block_idx].size() * sizeof(block_q4_0));
          ofs.write(reinterpret_cast<const char*>(v_cache_q4[i][j][block_idx].data()),
                    v_cache_q4[i][j][block_idx].size() * sizeof(block_q4_0));
        }
      }
    }
    for (int k = 0; k < past_block_num; ++k) {
      int block_idx = block_table[k];
      for (int l = 0; l < config_.block_len; l++) {
        ofs.write(reinterpret_cast<const char*>(importance_[i][block_idx][l].data()),
                  importance_[i][block_idx][l].size() * sizeof(ggml_fp16_t));
      }
    }
  }
  ofs.close();
  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  printf("time of dump: %f s\n", diff.count());
}
