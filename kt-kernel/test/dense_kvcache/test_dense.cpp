#ifndef DENSE_TEST_HEADER
#define DENSE_TEST_HEADER "operators/dense_kvcache/dense_kvcache.h"
#endif
#include DENSE_TEST_HEADER
#include "ggml-impl.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

extern std::atomic<int> dense_test_gemm_calls;
extern std::atomic<bool> dense_test_fail_gemm;
using Half = ggml_fp16_t;
static Half half(float x) { return GGML_FP32_TO_FP16(x); }
static float real(Half x) { return GGML_COMPUTE_FP16_TO_FP32(x); }
static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
static bool test_parallel_reduce = true;

struct Case {
  int heads, block, batch, stride, tokens;
  WorkerPool pool;
  dense::KVCache cache;
  std::vector<int> table, lengths;
  std::vector<Half> keys, values, queries, output;
  std::vector<float> lse;
  Case(int h, int t, int threads, std::vector<int> lens, int extra_stride = 7)
      : heads(h), block(t), batch(lens.size()),
        stride((*std::max_element(lens.begin(), lens.end()) + 3 + t - 1) / t + extra_stride),
        tokens(((*std::max_element(lens.begin(), lens.end()) + 3 + t - 1) / t) * t), pool(threads, 0),
        cache(dense::KVCacheConfig(2, 4, h, 128, t, GGML_TYPE_F16, batch * (tokens / t), batch, threads)),
        table(batch * stride, -1), lengths(lens),
        keys(size_t(batch) * tokens * 4 * 128), values(keys.size()),
        queries(size_t(batch) * h * 128), output(queries.size()), lse(batch * h) {
    cache.set_parallel_reduce(test_parallel_reduce);
    std::mt19937 rng(83);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& x : keys) x = half(dist(rng));
    for (auto& x : values) x = half(dist(rng));
    for (auto& x : queries) x = half(dist(rng));
    std::vector<int> physical(batch * (tokens / t));
    std::iota(physical.begin(), physical.end(), 0);
    std::shuffle(physical.begin(), physical.end(), rng);
    // Unused table slots intentionally stay -1, including the slot after an exact full block.
    int next = 0;
    for (int b = 0; b < batch; ++b) {
      int count = (lengths[b] + block - 1) / block;
      for (int j = 0; j < count; ++j) table[b * stride + j] = physical[next++];
    }
    free_blocks.assign(physical.begin() + next, physical.end());
  }
  std::vector<int> free_blocks;
  size_t kv(int b, int token, int h, int d) const {
    return ((size_t(b) * tokens + token) * 4 + h) * 128 + d;
  }
  void import(int layer) {
    // Bulk import each ragged sequence through the existing public update API.
    for (int b = 0; b < batch; ++b) {
      int zero = 0;
      cache.update_kvcache_fp16(keys.data() + kv(b, 0, 0, 0), values.data() + kv(b, 0, 0, 0),
          layer, table.data() + b * stride, 1, stride, &zero, lengths[b], &pool);
    }
  }
  void attention(int layer = 0) {
    dense_test_gemm_calls = 0;
    cache.attn(queries.data(), output.data(), lse.data(), layer, 0, 1, batch, stride,
               table.data(), lengths.data(), &pool);
    int blocks = 0;
    for (int len : lengths) blocks += (len + block - 1) / block;
    require(dense_test_gemm_calls == 2 * 4 * blocks, "wrong number of successful block GEMMs");
  }
  void reference() {
    // Full-sequence FP64 attention; independent of block boundaries and LSE merge order.
    for (int b = 0; b < batch; ++b) for (int h = 0; h < heads; ++h) {
      int len = lengths[b], kh = h / (heads / 4);
      size_t out = (size_t(b) * heads + h) * 128;
      if (!len) {
        require(std::isinf(lse[b * heads + h]) && lse[b * heads + h] < 0, "empty LSE");
        for (int d = 0; d < 128; ++d) require(real(output[out + d]) == 0, "empty output");
        continue;
      }
      std::vector<double> scores(len);
      for (int t = 0; t < len; ++t) {
        double dot = 0;
        for (int d = 0; d < 128; ++d) dot += double(real(queries[out + d])) * real(keys[kv(b,t,kh,d)]);
        scores[t] = dot / std::sqrt(128.0);
      }
      double mx = *std::max_element(scores.begin(), scores.end()), sum = 0;
      for (auto& x : scores) { x = std::exp(x - mx); sum += x; }
      double expected_lse = mx + std::log(sum);
      require(std::isfinite(lse[b * heads + h]) && std::abs(lse[b * heads + h] - expected_lse) < 0.004,
              "LSE differs from FP64 reference");
      for (int d = 0; d < 128; ++d) {
        double expected = 0;
        for (int t = 0; t < len; ++t) expected += scores[t] / sum * real(values[kv(b,t,kh,d)]);
        float actual = real(output[out + d]);
        if (!std::isfinite(actual) || std::abs(actual - expected) > 0.002 + 0.002 * std::abs(expected)) {
          std::cerr << "b=" << b << " h=" << h << " d=" << d << " actual=" << actual << " expected=" << expected << '\n';
          throw std::runtime_error("output differs from FP64 reference");
        }
      }
    }
  }
  void append() {
    std::vector<Half> k(size_t(batch) * 4 * 128), v(k.size());
    for (int b = 0; b < batch; ++b) {
      int logical = lengths[b] / block;
      if (table[b * stride + logical] < 0) {
        table[b * stride + logical] = free_blocks.back(); free_blocks.pop_back();
      }
      std::copy_n(keys.data() + kv(b, lengths[b], 0, 0), 4 * 128, k.data() + b * 4 * 128);
      std::copy_n(values.data() + kv(b, lengths[b], 0, 0), 4 * 128, v.data() + b * 4 * 128);
    }
    auto before = lengths;
    // Both layers must receive the same pre-append lengths.
    for (int layer = 0; layer < 2; ++layer) {
      auto layer_lengths = before;
      cache.attn_with_kvcache(queries.data(), k.data(), v.data(), output.data(), lse.data(),
          layer, 0, 1, batch, stride, table.data(), layer_lengths.data(), &pool);
      for (int b = 0; b < batch; ++b) require(layer_lengths[b] == before[b] + 1, "append length");
      lengths = layer_lengths;
      reference();
    }
  }
};

template<class F> void expect_error(F f) {
  bool caught = false;
  try { f(); } catch (const std::exception&) { caught = true; }
  require(caught, "expected exception");
}

static void correctness() {
  int cases = 0;
  // 长短序列混合、完整 4096-token 序列和尾块；在同一个 cache 上切换两种 reduce。
  for (int h = 32; h <= 64; h += 32) {
    Case long_case(h, 128, 4, {0, 1, 127, 128, 129, 4096});
    long_case.import(0);
    for (int mode = 0; mode < 2; ++mode) {
      long_case.cache.set_parallel_reduce(mode == 0);
      long_case.attention();
      long_case.reference();
    }
    ++cases;
  }
  for (int h : {32, 64}) for (int threads : {1, 4}) for (int t : {8, 32, 128}) {
    // Exercise 1/2/3-block tail tasks, exact four-block boundaries and partial tail blocks.
    Case c(h, t, threads, {0, 1, t-1, t, t+1, 2*t, 3*t+7, 4*t-1, 4*t,
                          4*t+1, 5*t, 6*t, 7*t, 8*t, 8*t+1, 0});
    c.import(0); c.import(1);
    c.attention(); c.reference();
    for (int i = 0; i < 3; ++i) c.append();
    // Refresh with a different batch length pattern to detect stale task prefixes/state.
    std::fill(c.lengths.begin(), c.lengths.end(), 0);
    c.attention(); c.reference();
    c.cache.attn(nullptr, nullptr, nullptr, 0, 0, 1, 0, 0, nullptr, nullptr, &c.pool);
    ++cases;
  }
  for (int h : {32, 64}) for (bool large : {false, true}) {
    Case c(h, 32, 4, {64, 65, 96, 128, 129, 256, 257, 1});
    std::fill(c.queries.begin(), c.queries.end(), half(0));
    for (int b = 0; b < c.batch; ++b) {
      for (int qh = 0; qh < h; ++qh) c.queries[(size_t(b)*h+qh)*128] = half(1);
      for (int t = 0; t < c.tokens; ++t) for (int kh = 0; kh < 4; ++kh) {
        for (int d = 0; d < 128; ++d) {
          // false: one score=0 per block and others=-200 -> valid block LSE=0.
          // true: negative/positive block scores exercise softmax and merge overflow.
          float score = large ? ((t / 128) % 2 == 0 ? -200.0f : 200.0f)
                              : (t % 32 == 0 ? 0.0f : -200.0f);
          c.keys[c.kv(b,t,kh,d)] = half(d == 0 ? score * std::sqrt(128.0f) : 0);
          c.values[c.kv(b,t,kh,d)] = half(float(t / 32 + 1));
        }
      }
    }
    c.import(0);
    for (int repeat = 0; repeat < 12; ++repeat) { c.attention(); c.reference(); }
    ++cases;
  }
  // Fail after earlier blocks have been accumulated, in both full and tail tasks.
  // Check both physical-page bounds and recovery without stale partial results.
  for (bool parallel : {false, true}) {
    Case boundary(32, 32, 4, {0, 7*32, 0});
    boundary.cache.set_parallel_reduce(parallel);
    boundary.import(0);
    for (int block : {1, 3, 4, 6}) {
      int& entry = boundary.table[boundary.stride + block];
      int saved = entry;
      for (int invalid : {-1, boundary.batch * (boundary.tokens / boundary.block)}) {
        entry = invalid;
        expect_error([&] { boundary.attention(); });
        entry = saved;
        boundary.attention(); boundary.reference();
      }
    }
    // A table shorter than the effective logical block count must fail before dispatch.
    expect_error([&] { boundary.cache.attn(boundary.queries.data(), boundary.output.data(),
        boundary.lse.data(), 0, 0, 1, boundary.batch, 6, boundary.table.data(),
        boundary.lengths.data(), &boundary.pool); });
    boundary.attention(); boundary.reference();
    ++cases;
  }
  Case c(32, 32, 4, {32, 0, 1});
  c.import(0);
  // Exact-block clear must not touch the following unallocated table entry.
  c.cache.clear_kvcache_all_layers(c.table.data(), c.lengths.data(), c.batch, c.stride, &c.pool);
  c.import(0); c.attention(); c.reference();
  expect_error([&] { c.cache.attn(c.queries.data(), c.output.data(), c.lse.data(), 0, 0, 2,
      c.batch, c.stride, c.table.data(), c.lengths.data(), &c.pool); });
  int saved = c.table[0]; c.table[0] = -1;
  expect_error([&] { c.attention(); }); c.table[0] = saved;
  dense_test_fail_gemm = true;
  expect_error([&] { c.attention(); }); dense_test_fail_gemm = false;
  c.attention(); c.reference();
  c.lengths[0] = -1; expect_error([&] { c.attention(); }); c.lengths[0] = 32;
  c.cache.ThreadResize(1);
  expect_error([&] { c.attention(); });
  c.cache.ThreadResize(4);
  auto before_lengths = c.lengths;
  std::vector<Half> new_k(size_t(c.batch) * 4 * 128, half(1));
  std::vector<Half> new_v(new_k.size(), half(2));
  // Sequence 0 is exactly full and its next block is unallocated: reject before writing.
  expect_error([&] { c.cache.attn_with_kvcache(c.queries.data(), new_k.data(), new_v.data(),
      c.output.data(), c.lse.data(), 0, 0, 1, c.batch, c.stride,
      c.table.data(), c.lengths.data(), &c.pool); });
  require(c.lengths == before_lengths, "failed append changed lengths");
  c.attention(); c.reference();
  c.cache.BatchResize(c.batch + 2); c.cache.BlockResize(c.batch * c.stride + 2); c.cache.ThreadResize(4);
  c.attention(); c.reference();
  expect_error([&] { c.cache.attn(c.queries.data(), c.output.data(), c.lse.data(), 0, 0, 1,
      c.batch, 0, c.table.data(), c.lengths.data(), &c.pool); });
  std::cout << "PASS: " << cases << " shape/thread/boundary scenarios, repeated decode, FP64 reference, empty KV, LSE=0, extreme scores, failure recovery and capacity checks\n";
}

static void benchmark() {
  for (int heads : {32, 64}) for (bool ragged : {false, true}) {
    std::vector<int> lengths(64, 512);
    if (ragged) for (int b = 0; b < 64; ++b) lengths[b] = 1 + (b * 131) % 512;
    Case c(heads, 128, 8, lengths, ragged ? 60 : 0);
    c.import(0);
    auto run = [&] { c.cache.attn(c.queries.data(), c.output.data(), c.lse.data(), 0, 0, 1,
          c.batch, c.stride, c.table.data(), c.lengths.data(), &c.pool); };
    for (int i = 0; i < 20; ++i) run();
    std::vector<double> samples;
    for (int i = 0; i < 101; ++i) {
      auto begin = std::chrono::steady_clock::now(); run();
      samples.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count());
    }
    std::sort(samples.begin(), samples.end());
    std::cout << "BENCH heads=" << heads << " ragged=" << ragged << " batch=64 threads=8 median_us=" << samples[50]
              << " p95_us=" << samples[95] << '\n';
  }
}
// 共享只读物理 block，低内存验证 profiling；这里的耗时不是实验室性能数据。
static void profile_smoke() {
  static Half query[64 * 32 * 128] = {0};
  static Half output[64 * 32 * 128];
  static float lse[64 * 32];
  static Half keys[128 * 4 * 128] = {0};
  static Half values[128 * 4 * 128];
  int table[64 * 32] = {0};
  int lengths[64];
  for (int i = 0; i < 128 * 4 * 128; ++i) values[i] = half(0.25f);
  for (int i = 0; i < 64; ++i) lengths[i] = 4096;
  for (int threads = 32; threads <= 64; threads += 32) {
    WorkerPool pool(threads, 0);
    dense::KVCache cache(dense::KVCacheConfig(1, 4, 32, 128, 128, GGML_TYPE_F16, 1, 64, threads));
    cache.set_parallel_reduce(test_parallel_reduce);
    int zero = 0;
    cache.update_kvcache_fp16(keys, values, 0, table, 1, 32, &zero, 128, &pool);
    cache.profile_reset(threads);
    cache.profile_enable(true);
    cache.attn(query, output, lse, 0, 0, 1, 64, 32, table, lengths, &pool);
    cache.profile_reset(threads);  // 上一次采样必须清除；reset 同时关闭计时。
    for (int repeat = 0; repeat < 5; ++repeat) {
      cache.profile_enable(repeat == 1 || repeat == 2 || repeat == 4);
      cache.attn(query, output, lse, 0, 0, 1, 64, 32, table, lengths, &pool);
      for (int i = 0; i < 64 * 32 * 128; ++i) {
        require(std::abs(real(output[i]) - 0.25f) < 0.002f, "profile changed output");
      }
      for (int i = 0; i < 64 * 32; ++i) {
        require(std::abs(lse[i] - std::log(4096.0f)) < 0.004f, "profile changed LSE");
      }
    }
    cache.profile_enable(false);
    cache.profile_write(test_parallel_reduce ? "build/dense_validation/profile_smoke.txt"
                                            : "build/dense_validation/profile_smoke_locked.txt", threads == 64);
  }
  std::cout << "PASS: 32/64-thread profile, reset, warmup exclusion, output and LSE; expected calls=3 tasks=6144 per group\n";
}

int main(int argc, char** argv) {
  try {
    if (argc > 1 && std::string(argv[1]) == "--legacy-reduce") {
      test_parallel_reduce = false;
      --argc;
      ++argv;
    }
    if (argc > 1 && std::string(argv[1]) == "--bench") benchmark();
    else if (argc > 1 && std::string(argv[1]) == "--profile-smoke") profile_smoke();
    else correctness();
  } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
