#!/usr/bin/env bash
# Standalone Linux x86-64 AVX2/F16C test; does not require CUDA, Python or pybind11.
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode="${1:-release}"
out="build/dense_validation/$mode"
mkdir -p "$out"
common=(-D_GNU_SOURCE -mavx2 -mfma -mf16c -pthread -ffunction-sections -fdata-sections -Ithird_party -Ithird_party/llama.cpp -Ikt-kernel)
prod=(-O3 -ffast-math)
link=(-Wl,--gc-sections -lnuma -lhwloc)
if [[ "$mode" == sanitize ]]; then
  prod=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  link+=(-fsanitize=address,undefined)
fi
# Build backend dependencies once per mode.
for src in third_party/llama.cpp/ggml.c third_party/llama.cpp/ggml-quants.c; do
  obj="$out/$(basename "$src").o"
  if [[ ! -f "$obj" || "$src" -nt "$obj" || "kt-kernel/test/dense_kvcache/run.sh" -nt "$obj" ]]; then
    gcc -std=gnu11 "${common[@]}" -O2 -c "$src" -o "$obj"
  fi
done
for src in third_party/llamafile/iqk_mul_mat_amd_avx2.cpp third_party/llamafile/flags.cpp kt-kernel/cpu_backend/worker_pool.cpp; do
  obj="$out/$(basename "$src").o"
  if [[ ! -f "$obj" || "$src" -nt "$obj" || "kt-kernel/test/dense_kvcache/run.sh" -nt "$obj" ]]; then
    g++ -std=c++20 "${common[@]}" -O2 -include cstring -c "$src" -o "$obj"
  fi
done
test_flags=()
sources=(kt-kernel/operators/dense_kvcache/dense_kvcache_{attn,utils,read_write,load_dump}.cpp)
if [[ -n "${DENSE_SOURCE_DIR:-}" ]]; then
  sources=("$DENSE_SOURCE_DIR"/dense_kvcache_{attn,utils,read_write,load_dump}.cpp)
  test_flags+=("-DDENSE_TEST_HEADER=\"$(realpath "$DENSE_SOURCE_DIR/dense_kvcache.h")\"")
fi
for src in "${sources[@]}" kt-kernel/test/dense_kvcache/gemm_driver.cpp; do
  g++ -std=c++20 "${common[@]}" "${prod[@]}" -c "$src" -o "$out/$(basename "$src").o"
done
# Reference checks must not be compiled with finite-math assumptions.
g++ -std=c++20 "${common[@]}" -O2 -fno-fast-math "${test_flags[@]}" -c kt-kernel/test/dense_kvcache/test_dense.cpp -o "$out/test.o"
g++ -pthread "$out"/*.o "${link[@]}" -o "$out/test_dense"
if [[ "${2:-}" == --build-only ]]; then exit 0; fi
if [[ "$mode" == sanitize ]]; then
  # The existing WorkerPool leaves hwloc topology allocations alive; check bounds/UB here.
  ASAN_OPTIONS=detect_leaks=0 "$out/test_dense"
else
  "$out/test_dense" "${@:2}"
fi
