// Use the repository's actual AVX2 GEMM, including its IQK dispatch, and count calls.
#include <atomic>
#define llamafile_sgemm dense_test_sgemm
#include "llamafile/tinyblas_cpu_sgemm.inc"
#undef llamafile_sgemm
std::atomic<int> dense_test_gemm_calls{0};
std::atomic<bool> dense_test_fail_gemm{false};
extern "C" bool llamafile_sgemm(long m, long n, long k, const void* a, long lda,
    const void* b, long ldb, void* c, long ldc, int ith, int nth, int task,
    int at, int bt, int ct, int precision) {
  ++dense_test_gemm_calls;
  if (dense_test_fail_gemm) return false;
  return dense_test_sgemm(m, n, k, a, lda, b, ldb, c, ldc, ith, nth, task, at, bt, ct, precision);
}
