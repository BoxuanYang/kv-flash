#!/usr/bin/env python
# coding=utf-8
"""Profile steady-state CPU decode attention; write results to kv-flash/large_batch_cpuperf.txt.

Run from the repository root on Linux:
  numactl --cpunodebind=0 --membind=0 python kt-kernel/test/dense_kvcache/test_cpu_perf.py

Inputs and KV cache stay unchanged during measurement. FlashAttention validates
each configuration before measurement; there are no GPU calls in the timed loop.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
import platform
import statistics
import time

import torch
from test_flash_attention import check_result


REPORT_PATH = Path(__file__).resolve().parents[3] / "large_batch_cpuperf.txt"
KV_HEADS = 4
HEAD_DIM = 128


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[128, 4096])
    parser.add_argument("--threads", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--q-heads", type=int, nargs="+", choices=[32, 64], default=[32])
    parser.add_argument("--block-lens", type=int, nargs="+", default=[128])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0", help="GPU used only for correctness validation")
    parser.add_argument("--pause-before-timing", action="store_true",
                        help="Single configuration only: wait for Enter so perf can attach after setup")
    args = parser.parse_args()
    if any(x <= 0 for x in args.batch_sizes + args.sequence_lengths + args.threads):
        parser.error("batch sizes, sequence lengths and thread counts must be positive")
    if any(x <= 0 or x % 8 for x in args.block_lens):
        parser.error("block lengths must be positive multiples of 8")
    if args.warmup < 0 or args.iterations <= 0 or args.rounds <= 0:
        parser.error("warmup must be nonnegative; iterations and rounds must be positive")
    configurations = (len(args.batch_sizes) * len(args.sequence_lengths) * len(args.threads)
                      * len(args.q_heads) * len(args.block_lens))
    if args.pause_before_timing and configurations != 1:
        parser.error("--pause-before-timing requires exactly one B/S/H/block/thread configuration")
    return args


@dataclass
class CpuCase:
    # Explicit ownership keeps the worker pool, cache and all raw-pointer inputs alive.
    backend_owner: object
    cache: object
    call_arguments: dict
    query: torch.Tensor
    block_table: torch.Tensor
    lengths: torch.Tensor
    output: torch.Tensor
    lse: torch.Tensor
    expected_output: torch.Tensor
    expected_lse: torch.Tensor

    def run(self):
        self.cache.attn(**self.call_arguments)

    def validate(self):
        check_result(self.output, self.lse, self.expected_output, self.expected_lse,
                     "CPU performance case")


def prepare_case(ext, flash_attention, args, batch, length, heads, block, threads):
    """Allocate one layer, import history, and validate against GPU before timing."""
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    kv_shape = (batch, length, KV_HEADS, HEAD_DIM)
    keys = torch.randn(kv_shape, generator=generator, dtype=torch.float16)
    values = torch.randn(kv_shape, generator=generator, dtype=torch.float16)
    query = torch.randn(batch, 1, heads, HEAD_DIM, generator=generator, dtype=torch.float16)
    lengths = torch.full((batch,), length, dtype=torch.int32)

    blocks_per_sequence = (length + block - 1) // block
    capacity = batch * blocks_per_sequence
    block_table = torch.full((batch, blocks_per_sequence + 7), -1, dtype=torch.int32)
    block_table[:, :blocks_per_sequence] = torch.randperm(
        capacity, generator=generator, dtype=torch.int32).reshape(batch, blocks_per_sequence)

    worker_config = ext.WorkerPoolConfig()
    worker_config.subpool_count = 1
    worker_config.subpool_numa_map = [0]
    worker_config.subpool_thread_count = [threads]
    backend_owner = ext.CPUInfer(worker_config)
    config = ext.dense_kvcache.KVCacheConfig(
        layer_num=1, kv_head_num=KV_HEADS, q_head_num=heads, head_dim=HEAD_DIM,
        block_len=block, kv_type=ext.kvcache.ggml_type.FP16,
        max_block_num=capacity, max_batch_size=batch, max_thread_num=threads,
    )
    cache = ext.dense_kvcache.KVCache(config)
    past_lengths = torch.zeros(batch, dtype=torch.int32)
    cache.update_kvcache_fp16(
        k_in=keys.data_ptr(), v_in=values.data_ptr(), layer_id=0,
        block_table=block_table.data_ptr(), batch_size=batch,
        max_block_num=block_table.shape[1], cache_seqlens=past_lengths.data_ptr(),
        q_len=length, backend=backend_owner.backend_,
    )

    gpu_output, gpu_lse = flash_attention(
        q=query.to(args.device), k_cache=keys.to(args.device), v_cache=values.to(args.device),
        cache_seqlens=lengths.to(args.device), softmax_scale=HEAD_DIM ** -0.5,
        causal=True, return_softmax_lse=True,
    )
    expected_output = gpu_output.float().cpu()
    expected_lse = gpu_lse.transpose(1, 2).float().cpu()
    torch.cuda.synchronize(args.device)

    output = torch.full_like(query, torch.nan)
    lse = torch.full(query.shape[:3], torch.nan, dtype=torch.float32)
    call_arguments = dict(
        q_in=query.data_ptr(), output=output.data_ptr(), attn_lse=lse.data_ptr(),
        layer_idx=0, generate_token_idx=0, q_len=1, batch_size=batch,
        max_block_num=block_table.shape[1], block_table=block_table.data_ptr(),
        cache_seqlens=lengths.data_ptr(), backend=backend_owner.backend_,
    )
    case = CpuCase(backend_owner, cache, call_arguments, query, block_table, lengths,
                   output, lse, expected_output, expected_lse)
    case.run()
    case.validate()
    return case


def measure_round(case, warmup, iterations):
    """Synchronous pybind wall time: includes scheduling, merge and output conversion."""
    run = case.run
    for _ in range(warmup):
        run()
    # Preallocate timing storage; do not print, validate, copy tensors or call CUDA here.
    samples = [0.0] * iterations
    for index in range(iterations):
        start = time.perf_counter_ns()
        run()
        samples[index] = (time.perf_counter_ns() - start) / 1_000_000
    return samples


def describe_affinity():
    """Report allowed CPUs, not a claim that KV pages were individually inspected."""
    lines = []
    task_directory = Path("/proc/self/task")
    for task in sorted(task_directory.iterdir(), key=lambda path: int(path.name)):
        try:
            fields = dict(line.split(":", 1) for line in (task / "status").read_text().splitlines()
                          if ":" in line)
            lines.append(f"  TID={task.name} name={fields['Name'].strip()} "
                         f"CPUs={fields['Cpus_allowed_list'].strip()} "
                         f"allowed_nodes={fields['Mems_allowed_list'].strip()}")
        except FileNotFoundError:
            continue  # A runtime helper thread may exit while /proc is being read.
    return "\n".join(lines)


def run_experiments(ext, flash_attention, args, log):
    rows = []
    failures = 0
    for batch in args.batch_sizes:
        for length in args.sequence_lengths:
            for heads in args.q_heads:
                for block in args.block_lens:
                    for threads in args.threads:
                        label = f"B={batch} S={length} Hq={heads} block={block} threads={threads}"
                        case = None
                        try:
                            log(f"\nPreparing {label}")
                            case = prepare_case(ext, flash_attention, args, batch, length, heads, block, threads)
                            log("Correctness before timing: PASS\n" + describe_affinity())
                            if args.pause_before_timing:
                                log(f"Attach perf to PID {os.getpid()} now; press Enter to start CPU rounds.")
                                input()
                            samples = []
                            round_medians = []
                            for round_index in range(args.rounds):
                                log(f"CPU round {round_index + 1}/{args.rounds} BEGIN: {label}")
                                round_samples = measure_round(case, args.warmup, args.iterations)
                                round_medians.append(statistics.median(round_samples))
                                samples.extend(round_samples)
                                log(f"CPU round END: median={round_medians[-1]:.6f} ms")
                            case.validate()
                            torch.testing.assert_close(case.lengths, torch.full_like(case.lengths, length))
                            median = statistics.median(samples)
                            p95 = sorted(samples)[max(0, (95 * len(samples) + 99) // 100 - 1)]
                            row = (f"{batch:6} {length:6} {heads:4} {block:6} {threads:7} "
                                   f"{median:11.6f} {p95:11.6f} {batch * 1000 / median:12.1f} PASS")
                            rows.append(row)
                            log(row)
                            log("Round medians (ms): " + ", ".join(f"{x:.6f}" for x in round_medians))
                            log("Samples (ms): " + ",".join(f"{x:.6f}" for x in samples), terminal=False)
                        except Exception as error:
                            failures += 1
                            row = f"{label} FAIL: {type(error).__name__}: {error}"
                            rows.append(row)
                            log(row)
                        finally:
                            # Release this configuration before allocating the next one.
                            case = None
    log("\nFINAL RESULTS")
    log(" Batch    Seq   Hq  Block Threads   Median ms      P95 ms   Seq/s/layer Check")
    for row in rows:
        log(row)
    log("恭喜！！正确性测试通过！" if not failures else "测试Fail！！")
    return 1 if failures else 0


def main():
    args = parse_arguments()
    # Each invocation replaces the previous report; flush progress to preserve partial results.
    with REPORT_PATH.open("w", encoding="utf-8") as report:
        def log(message, terminal=True):
            report.write(message + "\n")
            report.flush()
            if terminal:
                print(message, flush=True)

        log(f"CPU attention performance report: {datetime.now().astimezone().isoformat()}")
        log(f"Host={platform.node()} PID={os.getpid()} PyTorch={torch.__version__}")
        log(f"Arguments: {vars(args)}\nReport: {REPORT_PATH}")
        log("NUMA target=0; launch with numactl --cpunodebind=0 --membind=0.")
        log("Affinity diagnostics show allowed nodes, NOT actual KV page residency or memory policy.")
        log("One layer, uniform FP16 inputs, shuffled pages, unchanged Q/K/V reused each iteration.")
        log("Timing excludes setup, KV import/append, GPU validation and tensor transfers.")
        log("Seq/s/layer is batch/latency, not full-model token throughput.")
        try:
            if platform.system() != "Linux":
                raise RuntimeError("Run this benchmark in the target Linux environment")
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA PyTorch is required for the initial correctness check")
            from flash_attn import flash_attn_with_kvcache
            from kt_kernel import kt_kernel_ext
            torch.set_num_threads(1)
            log(f"Extension: {kt_kernel_ext.__file__}")
            with torch.inference_mode():
                return run_experiments(kt_kernel_ext, flash_attn_with_kvcache, args, log)
        except Exception as error:
            log(f"FAIL: {type(error).__name__}: {error}\n测试Fail！！")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
