#include "qwen3_cpu_ops.h"

#include <cassert>
#include <cmath>

#include "ggml-impl.h"

namespace dense {

void rms_norm_fp16(const ggml_fp16_t* input, const ggml_fp16_t* weight,
                   ggml_fp16_t* output, int rows, int hidden_dim, float eps,
                   WorkerPool* backend) {
  assert(input != nullptr);
  assert(weight != nullptr);
  assert(output != nullptr);
  assert(backend != nullptr);
  assert(rows >= 0);
  assert(hidden_dim > 0);
  assert(eps >= 0.0f);
  if (rows == 0) return;

  backend->do_work_stealing_job(rows, [&](int row) {
    const ggml_fp16_t* input_row = input + row * hidden_dim;
    ggml_fp16_t* output_row = output + row * hidden_dim;

    float sum_squares = 0.0f;
    for (int i = 0; i < hidden_dim; ++i) {
      const float value = GGML_FP16_TO_FP32(input_row[i]);
      sum_squares += value * value;
    }

    const float inv_rms =
        1.0f / std::sqrt(sum_squares / static_cast<float>(hidden_dim) + eps);
    for (int i = 0; i < hidden_dim; ++i) {
      const float value = GGML_FP16_TO_FP32(input_row[i]);
      const float scale = GGML_FP16_TO_FP32(weight[i]);
      output_row[i] = GGML_FP32_TO_FP16(value * inv_rms * scale);
    }
  });
}

namespace {

void apply_qwen3_rope_to_head(ggml_fp16_t* head, const ggml_fp16_t* cos_row,
                              const ggml_fp16_t* sin_row, int head_dim) {
  const int half_dim = head_dim / 2;
  for (int i = 0; i < half_dim; ++i) {
    const float first = GGML_FP16_TO_FP32(head[i]);
    const float second = GGML_FP16_TO_FP32(head[i + half_dim]);
    const float cos_first = GGML_FP16_TO_FP32(cos_row[i]);
    const float sin_first = GGML_FP16_TO_FP32(sin_row[i]);
    const float cos_second = GGML_FP16_TO_FP32(cos_row[i + half_dim]);
    const float sin_second = GGML_FP16_TO_FP32(sin_row[i + half_dim]);

    head[i] = GGML_FP32_TO_FP16(first * cos_first - second * sin_first);
    head[i + half_dim] =
        GGML_FP32_TO_FP16(second * cos_second + first * sin_second);
  }
}

}  // namespace

void qwen3_rope_fp16(ggml_fp16_t* q, ggml_fp16_t* k,
                     const int* position_ids, const ggml_fp16_t* cos,
                     const ggml_fp16_t* sin, int batch_size, int q_len,
                     int q_head_num, int kv_head_num, int head_dim,
                     WorkerPool* backend) {
  assert(q != nullptr);
  assert(k != nullptr);
  assert(position_ids != nullptr);
  assert(cos != nullptr);
  assert(sin != nullptr);
  assert(backend != nullptr);
  assert(batch_size >= 0);
  assert(q_len >= 0);
  assert(q_head_num > 0);
  assert(kv_head_num > 0);
  assert(head_dim > 0 && head_dim % 2 == 0);

  const int token_count = batch_size * q_len;
  if (token_count == 0) return;
  const int heads_per_token = q_head_num + kv_head_num;
  backend->do_work_stealing_job(token_count * heads_per_token, [&](int task_id) {
    const int token_id = task_id / heads_per_token;
    const int head_id = task_id % heads_per_token;
    const int position = position_ids[token_id];
    assert(position >= 0);

    const ggml_fp16_t* cos_row = cos + position * head_dim;
    const ggml_fp16_t* sin_row = sin + position * head_dim;
    if (head_id < q_head_num) {
      ggml_fp16_t* q_head =
          q + (token_id * q_head_num + head_id) * head_dim;
      apply_qwen3_rope_to_head(q_head, cos_row, sin_row, head_dim);
    } else {
      const int kv_head_id = head_id - q_head_num;
      ggml_fp16_t* k_head =
          k + (token_id * kv_head_num + kv_head_id) * head_dim;
      apply_qwen3_rope_to_head(k_head, cos_row, sin_row, head_dim);
    }
  });
}

}  // namespace dense
