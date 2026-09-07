#!/usr/bin/env python
# coding=utf-8
"""
Description  : Validate CPU Dense KVCache decode attention with CUDA FlashAttention.
Author       : Boxuan Yang
Date         : 2026-09-07
Version      : 1.0.0
Copyright (c) 2026 by KVCache.AI, All Rights Reserved.
"""

import argparse
import math
import statistics
import time
import torch


# Test configuration
layer_num = 2
kv_head_num = 4
head_dim = 128

q_head_num_list = [32, 64]
block_len_list = [32, 128]
thread_num_list = [1, 4]
batch_size_list = [1, 8, 32, 64]
sequence_length_list = [128, 1024, 4096]
warmup_iterations = 5
timing_iterations = 20
test_mode_list = ["random", "zero_lse", "extreme"]

decode_steps = 3
seed = 2026

device = torch.device("cuda:0")
numa_node = 0

output_atol = 2e-3
output_rtol = 2e-3
lse_atol = 4e-3


def make_sequence_lengths(batch_size, sequence_length, block_len, length_mode):
    """Uniform cases measure a specific length; ragged cases also cover empty/tail blocks."""
    if length_mode == "uniform":
        return [sequence_length] * batch_size
    boundaries = [0, 1, block_len - 1, block_len, block_len + 1,
                  2 * block_len - 1, 2 * block_len, sequence_length]
    lengths = [min(boundaries[i % len(boundaries)], sequence_length)
               for i in range(batch_size)]
    lengths[-1] = sequence_length
    return lengths


def benchmark_attention(cpu_call, gpu_call):
    """Median single-layer attention latency; KV writes and CPU/GPU copies are excluded."""
    for _ in range(warmup_iterations):
        cpu_call()
        gpu_call()
    torch.cuda.synchronize(device)

    cpu_times = []
    for _ in range(timing_iterations):
        start = time.perf_counter()
        cpu_call()  # The pybind call waits for CPU workers to finish.
        cpu_times.append((time.perf_counter() - start) * 1000)

    gpu_times = []
    with torch.cuda.device(device):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        for _ in range(timing_iterations):
            start_event.record()
            gpu_call()
            end_event.record()
            end_event.synchronize()
            gpu_times.append(start_event.elapsed_time(end_event))
    return statistics.median(cpu_times), statistics.median(gpu_times)


def measure_case(local_kvcache, backend, q, k_cache, v_cache, block_table, cache_seqlens):
    """Prepare both sides once, then time attention on the same unchanged history."""
    cpu_output = torch.empty_like(q)
    cpu_lse = torch.empty(q.shape[:3], dtype=torch.float32)
    gpu_q = q.to(device)
    gpu_k = k_cache.to(device)
    gpu_v = v_cache.to(device)
    gpu_lengths = cache_seqlens.to(device)

    def cpu_call():
        local_kvcache.attn(
            q_in=q.data_ptr(), output=cpu_output.data_ptr(), attn_lse=cpu_lse.data_ptr(),
            layer_idx=0, generate_token_idx=0, q_len=1, batch_size=q.shape[0],
            max_block_num=block_table.shape[1], block_table=block_table.data_ptr(),
            cache_seqlens=cache_seqlens.data_ptr(), backend=backend,
        )

    def gpu_call():
        return flash_attn_with_kvcache(
            q=gpu_q, k_cache=gpu_k, v_cache=gpu_v, cache_seqlens=gpu_lengths,
            softmax_scale=1.0 / math.sqrt(head_dim), causal=True, return_softmax_lse=True,
        )

    times = benchmark_attention(cpu_call, gpu_call)
    # Check the timed calls too, not just the separate decode validation path.
    gpu_output, gpu_lse = gpu_call()
    check_result(cpu_output, cpu_lse, gpu_output.float().cpu(),
                 gpu_lse.transpose(1, 2).float().cpu(), "timed attention")
    return times


def create_cpu_backend(thread_num):
    """Keep all requested workers in one subpool; dense attention uses subpool 0."""
    worker_config = kt_kernel_ext.WorkerPoolConfig()
    worker_config.subpool_count = 1
    worker_config.subpool_numa_map = [numa_node]
    worker_config.subpool_thread_count = [thread_num]
    return kt_kernel_ext.CPUInfer(worker_config)


def call_cpu_attention(cache, backend, q, output, lse, block_table, seqlens,
                       layer_idx=0, step=0, new_k=None, new_v=None):
    """All pointers refer to contiguous CPU tensors that stay alive during this synchronous call."""
    args = dict(
        q_in=q.data_ptr(), output=output.data_ptr(), attn_lse=lse.data_ptr(),
        layer_idx=layer_idx, generate_token_idx=step, q_len=1, batch_size=q.shape[0],
        max_block_num=block_table.shape[1], block_table=block_table.data_ptr(),
        cache_seqlens=seqlens.data_ptr(), backend=backend,
    )
    if new_k is None:
        cache.attn(**args)
    else:
        cache.attn_with_kvcache(k_in=new_k.data_ptr(), v_in=new_v.data_ptr(), **args)


@torch.no_grad()
def flash_attention_reference(q, k_cache, v_cache, cache_seqlens):
    """
    FlashAttention reference.

    Input:
        q           : [batch, 1, q_head_num, head_dim]
        k_cache     : [batch, max_seq_len, kv_head_num, head_dim]
        v_cache     : [batch, max_seq_len, kv_head_num, head_dim]
        cache_seqlens: [batch]

    Return:
        output      : [batch, 1, q_head_num, head_dim], FP32 CPU
        attn_lse    : [batch, 1, q_head_num], FP32 CPU
    """

    batch_size, _, q_head_num, _ = q.shape

    output = torch.zeros(q.shape, dtype=torch.float32)
    attn_lse = torch.full(
        (batch_size, 1, q_head_num),
        -torch.inf,
        dtype=torch.float32,
    )

    # FlashAttention is only called for non-empty KV sequences.
    active = torch.nonzero(cache_seqlens > 0, as_tuple=True)[0]

    if active.numel() == 0:
        return output, attn_lse

    gpu_output, gpu_lse = flash_attn_with_kvcache(
        q=q[active].to(device),
        k_cache=k_cache[active].to(device),
        v_cache=v_cache[active].to(device),
        cache_seqlens=cache_seqlens[active].to(device),
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
        return_softmax_lse=True,
    )

    output[active] = gpu_output.float().cpu()

    # FlashAttention LSE: [batch, q_head_num, 1]
    # CPU kernel LSE     : [batch, 1, q_head_num]
    attn_lse[active] = gpu_lse.transpose(1, 2).float().cpu()

    return output, attn_lse


def check_result(cpu_output, cpu_lse, gpu_output, gpu_lse, label):
    """Compare CPU result with FlashAttention."""

    torch.testing.assert_close(
        cpu_output.float(),
        gpu_output,
        atol=output_atol,
        rtol=output_rtol,
        msg=lambda detail: f"{label}\n{detail}",
    )

    torch.testing.assert_close(
        cpu_lse,
        gpu_lse,
        atol=lse_atol,
        rtol=0,
        msg=lambda detail: f"{label} LSE\n{detail}",
    )

    output_max_abs = (
        cpu_output.float() - gpu_output
    ).abs().max().item()

    finite = torch.isfinite(gpu_lse)

    if finite.any():
        lse_max_abs = (
            cpu_lse[finite] - gpu_lse[finite]
        ).abs().max().item()
    else:
        lse_max_abs = 0.0

    return output_max_abs, lse_max_abs


@torch.no_grad()
def create_test_data(batch_size, sequence_length, block_len, test_mode, length_mode):
    """Build logical FP16 inputs and a shuffled CPU page table, including room for appends."""
    cache_length_list = make_sequence_lengths(
        batch_size, sequence_length, block_len, length_mode)

    cache_seqlens_init = torch.tensor(
        cache_length_list,
        dtype=torch.int32,
        device="cpu",
    )

    batch_size = len(cache_length_list)
    max_seq_len = max(cache_length_list) + decode_steps

    generator = torch.Generator().manual_seed(seed)

    # Logical contiguous KV cache used by FlashAttention.
    k_cache = torch.randn(
        batch_size,
        max_seq_len,
        kv_head_num,
        head_dim,
        generator=generator,
        dtype=torch.float16,
        device="cpu",
    ).contiguous()

    v_cache = torch.randn(
        batch_size,
        max_seq_len,
        kv_head_num,
        head_dim,
        generator=generator,
        dtype=torch.float16,
        device="cpu",
    ).contiguous()

    # Numerical corner cases.
    if test_mode != "random":
        k_cache.zero_()

        for token_idx in range(max_seq_len):

            if test_mode == "zero_lse":
                score = (
                    0.0
                    if token_idx % block_len == 0
                    else -200.0
                )
            else:  # extreme
                score = (
                    -200.0
                    if token_idx < block_len
                    else 200.0
                )

            k_cache[:, token_idx, :, 0] = (
                score * math.sqrt(head_dim)
            )

            v_cache[:, token_idx] = (
                token_idx // block_len + 1
            )

    # Build randomized paged-KV block table.
    block_count_list = [
        (length + decode_steps + block_len - 1) // block_len
        for length in cache_length_list
    ]

    physical_block_num = sum(block_count_list)
    max_block_num = max(block_count_list) + 7

    block_table = torch.full(
        (batch_size, max_block_num),
        -1,
        dtype=torch.int32,
        device="cpu",
    )

    physical_blocks = torch.randperm(
        physical_block_num,
        generator=generator,
        dtype=torch.int32,
    )

    offset = 0

    for batch_idx, block_count in enumerate(block_count_list):
        block_table[batch_idx, :block_count] = (
            physical_blocks[offset:offset + block_count]
        )
        offset += block_count

    return k_cache, v_cache, block_table, physical_block_num, cache_seqlens_init, generator


def import_cpu_history(cache, backend, k_cache, v_cache, block_table, lengths):
    """Import each ragged sequence independently; this writes KV without prefill attention."""
    for layer_idx in range(layer_num):
        for batch_idx, length in enumerate(lengths.tolist()):
            if length == 0:
                continue
            past_length = torch.zeros(1, dtype=torch.int32)
            cache.update_kvcache_fp16(
                k_in=k_cache[batch_idx].data_ptr(), v_in=v_cache[batch_idx].data_ptr(),
                layer_id=layer_idx, block_table=block_table[batch_idx].data_ptr(),
                batch_size=1, max_block_num=block_table.shape[1],
                cache_seqlens=past_length.data_ptr(), q_len=length, backend=backend,
            )


@torch.no_grad()
def run_test(q_head_num, block_len, thread_num, test_mode,
             batch_size, sequence_length, length_mode):
    """Validate one workload through history import, repeated decode and empty-cache reset."""
    k_cache, v_cache, block_table, physical_block_num, cache_seqlens_init, generator = create_test_data(
        batch_size, sequence_length, block_len, test_mode, length_mode)
    # Create CPU worker pool.
    backend_owner = create_cpu_backend(thread_num)
    backend = backend_owner.backend_

    # Create CPU Dense KVCache.
    config = kt_kernel_ext.dense_kvcache.KVCacheConfig(
        layer_num=layer_num,
        kv_head_num=kv_head_num,
        q_head_num=q_head_num,
        head_dim=head_dim,
        block_len=block_len,
        kv_type=kt_kernel_ext.kvcache.ggml_type.FP16,
        max_block_num=physical_block_num,
        max_batch_size=batch_size,
        max_thread_num=thread_num,
    )

    local_kvcache = kt_kernel_ext.dense_kvcache.KVCache(config)

    # Each layer has independent sequence-length state.
    layer_cache_seqlens = [
        cache_seqlens_init.clone()
        for _ in range(layer_num)
    ]

    import_cpu_history(local_kvcache, backend, k_cache, v_cache, block_table, cache_seqlens_init)

    worst_output_error = 0.0
    worst_lse_error = 0.0
    timings = None

    # Decode validation.
    for step in range(decode_steps + 1):

        q = torch.randn(
            batch_size,
            1,
            q_head_num,
            head_dim,
            generator=generator,
            dtype=torch.float16,
            device="cpu",
        ).contiguous()

        if test_mode != "random":
            q.zero_()
            q[..., 0] = 1.0

        expected_cache_seqlens = cache_seqlens_init + step

        gpu_output, gpu_lse = flash_attention_reference(
            q,
            k_cache,
            v_cache,
            expected_cache_seqlens,
        )

        # Starting from step 1, append one new KV token.
        new_k = new_v = None
        if step > 0:

            token_position = (
                cache_seqlens_init + step - 1
            ).long()

            batch_index = torch.arange(batch_size)

            new_k = k_cache[
                batch_index,
                token_position,
            ].unsqueeze(1).contiguous()

            new_v = v_cache[
                batch_index,
                token_position,
            ].unsqueeze(1).contiguous()

        for layer_idx in range(layer_num):

            cpu_output = torch.full_like(q, torch.nan)

            cpu_lse = torch.full(
                (batch_size, 1, q_head_num),
                torch.nan,
                dtype=torch.float32,
                device="cpu",
            )

            call_cpu_attention(
                local_kvcache, backend, q, cpu_output, cpu_lse, block_table,
                layer_cache_seqlens[layer_idx], layer_idx, step, new_k, new_v,
            )

            # KV append must update sequence lengths correctly.
            torch.testing.assert_close(
                layer_cache_seqlens[layer_idx],
                expected_cache_seqlens,
                rtol=0,
                atol=0,
            )

            label = (
                f"Hq={q_head_num}, "
                f"block={block_len}, "
                f"threads={thread_num}, "
                f"{test_mode}, "
                f"step={step}, "
                f"layer={layer_idx}"
            )

            output_error, lse_error = check_result(
                cpu_output,
                cpu_lse,
                gpu_output,
                gpu_lse,
                label,
            )

            worst_output_error = max(
                worst_output_error,
                output_error,
            )

            worst_lse_error = max(
                worst_lse_error,
                lse_error,
            )

        # Only random, uniform cases are timed. Each row therefore has one exact B/S.
        # Run before append steps so sequence_length is the measured KV length.
        if step == 0 and test_mode == "random" and length_mode == "uniform":
            timings = measure_case(local_kvcache, backend, q, k_cache, v_cache,
                                  block_table, layer_cache_seqlens[0])

    # Empty-KV contract:
    # output = 0, LSE = -inf.
    empty_cache_seqlens = torch.zeros(
        batch_size,
        dtype=torch.int32,
        device="cpu",
    )

    empty_output = torch.full_like(q, torch.nan)

    empty_lse = torch.full(
        (batch_size, 1, q_head_num),
        torch.nan,
        dtype=torch.float32,
        device="cpu",
    )

    call_cpu_attention(local_kvcache, backend, q, empty_output, empty_lse,
                       block_table, empty_cache_seqlens)

    expected_empty_output = torch.zeros(
        q.shape,
        dtype=torch.float32,
    )

    expected_empty_lse = torch.full_like(
        empty_lse,
        -torch.inf,
    )

    check_result(
        empty_output,
        empty_lse,
        expected_empty_output,
        expected_empty_lse,
        "all-empty KV",
    )

    print(
        f"PASS "
        f"batch={batch_size} seq={sequence_length} {length_mode} "
        f"Hq={q_head_num:2d} "
        f"block={block_len:3d} "
        f"threads={thread_num:2d} "
        f"{test_mode:8s} "
        f"max_abs={worst_output_error:.6g} "
        f"lse_max_abs={worst_lse_error:.6g}"
    )

    return timings


def parse_arguments():
    parser = argparse.ArgumentParser(description="CPU decode correctness and FlashAttention timing")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=batch_size_list,
                        help="Concurrent sequences in one decode batch")
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=sequence_length_list,
                        help="KV lengths before the first append")
    parser.add_argument("--threads", type=int, nargs="+", default=thread_num_list)
    parser.add_argument("--block-lens", type=int, nargs="+", default=block_len_list)
    parser.add_argument("--q-heads", type=int, nargs="+", choices=[32, 64], default=q_head_num_list)
    parser.add_argument("--warmup", type=int, default=warmup_iterations)
    parser.add_argument("--iterations", type=int, default=timing_iterations)
    args = parser.parse_args()
    if any(value <= 0 for value in args.batch_sizes + args.sequence_lengths + args.threads):
        parser.error("batch sizes, sequence lengths and threads must be positive")
    if any(value <= 0 or value % 8 for value in args.block_lens):
        parser.error("block lengths must be positive multiples of 8")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be nonnegative and iterations must be positive")
    return args


def print_summary(results):
    """Keep every B/S/H/block/thread combination visible, including failures."""
    print("\nSingle-layer attention latency, median milliseconds (random uniform inputs)")
    print("CPU: synchronous pybind wall time; GPU: CUDA event elapsed time around FlashAttention.")
    print("Excluded: KV import/append, tensor copies, reference comparison. This is not end-to-end decode latency.")
    print(f"{'Batch':>6} {'Seq':>6} {'Hq':>4} {'Block':>6} {'Threads':>7} "
          f"{'CPU ms':>10} {'GPU ms':>10} {'CPU/GPU':>9} {'Checks':>8}")
    for result in results:
        batch, length, heads, block, threads = result["shape"]
        times = result["timings"]
        if times is None:
            timing_text = f"{'--':>10} {'--':>10} {'--':>9}"
        else:
            cpu_ms, gpu_ms = times
            timing_text = f"{cpu_ms:10.4f} {gpu_ms:10.4f} {cpu_ms / gpu_ms:9.2f}"
        status = "FAIL" if result["errors"] else "PASS"
        print(f"{batch:6} {length:6} {heads:4} {block:6} {threads:7} {timing_text} {status:>8}")
    for result in results:
        for error in result["errors"]:
            print(f"\nFAIL B/S/H/block/threads={result['shape']}: {error}")
    passed = bool(results) and all(not result["errors"] for result in results)
    print("恭喜！！正确性测试通过！" if passed else "测试Fail！！", flush=True)


def run_configuration(batch, length, heads, block, threads):
    result = {"shape": (batch, length, heads, block, threads), "timings": None, "errors": []}
    for length_mode in ["uniform", "ragged"]:
        for test_mode in test_mode_list:
            try:
                times = run_test(heads, block, threads, test_mode, batch, length, length_mode)
                if times is not None:
                    result["timings"] = times
            except (AssertionError, RuntimeError, ValueError) as error:
                # Continue with other workloads so the final report includes all concurrency levels.
                result["errors"].append(f"{length_mode}/{test_mode}: {type(error).__name__}: {error}")
                print(f"FAIL batch={batch} seq={length} Hq={heads} block={block} "
                      f"threads={threads} {length_mode}/{test_mode}: {error}", flush=True)
    return result


def main():
    global flash_attn_with_kvcache, kt_kernel_ext, warmup_iterations, timing_iterations
    args = parse_arguments()
    warmup_iterations, timing_iterations = args.warmup, args.iterations
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA PyTorch and a FlashAttention-compatible GPU are required")
        from flash_attn import flash_attn_with_kvcache
        from kt_kernel import kt_kernel_ext
        torch.set_num_threads(1)
        print(f"PyTorch: {torch.__version__}; GPU: {torch.cuda.get_device_name(device)}")
        print(f"pybind module: {kt_kernel_ext.__file__}")
        print(f"Timing: {warmup_iterations} warmups, {timing_iterations} samples; CPU NUMA node 0")

        results = []
        with torch.inference_mode():
            for batch in args.batch_sizes:
                for length in args.sequence_lengths:
                    for heads in args.q_heads:
                        for block in args.block_lens:
                            for threads in args.threads:
                                results.append(run_configuration(batch, length, heads, block, threads))
        print_summary(results)
    except Exception as error:
        print(f"Unable to complete test: {type(error).__name__}: {error}", flush=True)
        print("测试Fail！！", flush=True)
    return 0  # Required by this script's terminal-reporting contract, even on failure.


if __name__ == "__main__":
    raise SystemExit(main())
