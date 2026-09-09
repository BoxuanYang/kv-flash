#!/usr/bin/env python
# coding=utf-8
"""Run exactly two CPU profiles: B=64, S=4096, block=128, threads=32/64.

From the repository root on the lab Linux machine, after rebuilding kt-kernel:
  numactl --cpunodebind=0 --membind=0 python kt-kernel/test/dense_kvcache/test_cpu_profile.py

Each compute task covers up to four consecutive blocks of one sequence/KV head.
C++ writes both groups to cpu_profile.txt in the repository root.
"""

import argparse
from pathlib import Path
import platform

import torch
from test_cpu_perf import prepare_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-heads", type=int, choices=[32, 64], default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0", help="GPU for correctness checks only")
    parser.add_argument("--reduce-mode", choices=["two-phase", "locked"], default="two-phase")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[3] / "cpu_profile.txt")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.rounds <= 0:
        parser.error("warmup must be nonnegative; iterations and rounds must be positive")
    if platform.system() != "Linux":
        raise RuntimeError("Run on the lab Linux machine")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required for the FlashAttention correctness check")

    from flash_attn import flash_attn_with_kvcache
    from kt_kernel import kt_kernel_ext

    if not hasattr(kt_kernel_ext.dense_kvcache.KVCache, "set_parallel_reduce"):
        raise RuntimeError("Rebuild kt-kernel first: the loaded extension has no reduce mode switch")
    torch.set_num_threads(1)
    report = args.output.resolve()
    with torch.inference_mode():
        for threads in (32, 64):
            print(f"Profiling B=64 S=4096 block=128 Hq={args.q_heads} threads={threads} reduce={args.reduce_mode}", flush=True)
            # Reuse existing inputs, shuffled physical pages, NUMA-0 pool and GPU reference.
            case = prepare_case(kt_kernel_ext, flash_attn_with_kvcache, args,
                                64, 4096, args.q_heads, 128, threads)
            case.cache.profile_reset(threads)
            for _ in range(args.rounds):
                for _ in range(args.warmup):
                    case.run()
                case.cache.profile_enable(True)
                for _ in range(args.iterations):
                    case.run()
                case.cache.profile_enable(False)
            case.validate()
            torch.testing.assert_close(case.lengths, torch.full_like(case.lengths, 4096))
            # All division and file writing happen in C++, once per configuration.
            case.cache.profile_write(str(report), threads == 64)
            print(f"PASS: threads={threads}; C++ report saved to {report}", flush=True)
            del case


if __name__ == "__main__":
    main()
