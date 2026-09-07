"""通过现有 kt_kernel_ext pybind 接口，对比 CPU kernel 与 CUDA FlashAttention。

运行前需在同一 Python 环境安装 CUDA PyTorch、flash-attn，并编译本仓库 kt_kernel_ext：
  python kt-kernel/test/dense_kvcache/test_flash_attention.py
  python kt-kernel/test/dense_kvcache/test_flash_attention.py --threads 1 4 --block-lens 32 128

相同 FP16 输入分别交给 CPU pybind 和 GPU FlashAttention。Q/K 已完成 Norm/RoPE。
只检查 decode attention、LSE 和 KV 追加的正确性，不测量性能。
"""

import argparse
import importlib
import math

import torch


@torch.no_grad()
def flash_attention(q, k, v, seqlens, flash_attn_with_kvcache, device):
    """输入为 CPU FP16 tensor；调用 FlashAttention 后返回 CPU FP32 输出和 LSE。"""
    batch, _, q_heads, head_dim = q.shape
    output = torch.zeros(q.shape, dtype=torch.float32)
    lse = torch.full((batch, 1, q_heads), -torch.inf, dtype=torch.float32)
    # 空 KV 单独检查本实现的 output=0、LSE=-inf 约定，不依赖 FlashAttention 的空行行为。
    active = torch.nonzero(seqlens > 0, as_tuple=True)[0]
    if active.numel():
        # 参考端使用连续 KV，按真实长度屏蔽未来 token；CPU 端独立验证分页映射和追加。
        # FlashAttention 原生支持 GQA，无需复制 KV heads。
        gpu_output, gpu_lse = flash_attn_with_kvcache(
            q[active].to(device), k[active].to(device), v[active].to(device),
            cache_seqlens=seqlens[active].to(device),
            softmax_scale=1.0 / math.sqrt(head_dim), causal=True,
            return_softmax_lse=True,
        )
        output[active] = gpu_output.float().cpu()
        lse[active] = gpu_lse.transpose(1, 2).float().cpu()  # [B,Hq,1] -> [B,1,Hq]
    return output, lse


def check_output(label, actual, actual_lse, expected, expected_lse):
    # 两端使用 FP16 输入，但分块、归并和舍入路径不同；不要求逐位相同。
    torch.testing.assert_close(actual.float(), expected, atol=2e-3, rtol=2e-3, msg=label)
    torch.testing.assert_close(actual_lse, expected_lse, atol=4e-3, rtol=0, msg=label + " LSE")
    output_error = (actual.float() - expected).abs().max().item()
    finite = torch.isfinite(expected_lse)
    lse_error = (actual_lse[finite] - expected_lse[finite]).abs().max().item() if finite.any() else 0.0
    return output_error, lse_error


@torch.no_grad()
def run_case(ext, flash_attn_with_kvcache, device,
             q_heads, block_len, threads, steps, mode, seed, long_context):
    kv_heads, head_dim, layers = 4, 128, 2
    lengths_list = [0, 1, block_len - 1, block_len, block_len + 1,
                    2 * block_len - 1, 2 * block_len, 2 * block_len + 3]
    if long_context:
        lengths_list.append(long_context)
    initial_lengths = torch.tensor(lengths_list, dtype=torch.int32)
    batch = len(lengths_list)
    max_tokens = max(lengths_list) + steps
    generator = torch.Generator().manual_seed(seed)
    k = torch.randn(batch, max_tokens, kv_heads, head_dim, generator=generator).half()
    v = torch.randn(batch, max_tokens, kv_heads, head_dim, generator=generator).half()

    if mode != "random":
        # 每块只有第一个位置 score=0，其余 score=-200，可产生合法 LSE=0。
        # extreme 则让不同 block 的 score 从 -200 跳到 +200。
        k.zero_()
        for t in range(max_tokens):
            score = (0.0 if t % block_len == 0 else -200.0) if mode == "zero_lse" else (
                -200.0 if t < block_len else 200.0)
            k[:, t, :, 0] = score * math.sqrt(head_dim)
            v[:, t] = float(t // block_len + 1)

    # 物理 block 打乱。每行保留额外槽位并填 -1，避免误把行步长当成有效 block 数。
    final_blocks = [(length + steps + block_len - 1) // block_len for length in lengths_list]
    physical_capacity = sum(final_blocks)
    table_stride = max(final_blocks) + 7
    block_table = torch.full((batch, table_stride), -1, dtype=torch.int32)
    permutation = torch.randperm(physical_capacity, generator=generator, dtype=torch.int32)
    offset = 0
    for b, count in enumerate(final_blocks):
        block_table[b, :count] = permutation[offset:offset + count]
        offset += count

    # 以下对象和方法均来自现有 ext_bindings.cpp，直接传入同进程 CPU tensor 的指针。
    backend = ext.WorkerPool(threads)
    config = ext.dense_kvcache.KVCacheConfig(
        layer_num=layers, kv_head_num=kv_heads, q_head_num=q_heads, head_dim=head_dim,
        block_len=block_len, kv_type=ext.kvcache.ggml_type.FP16,
        max_block_num=physical_capacity, max_batch_size=batch, max_thread_num=threads,
    )
    cache = ext.dense_kvcache.KVCache(config)
    layer_lengths = [initial_lengths.clone() for _ in range(layers)]

    # 每个序列长度不同，逐序列导入已有历史；这里只写 KV，不计算 prefill attention。
    for layer in range(layers):
        for b, length in enumerate(lengths_list):
            if length == 0:
                continue
            past_length = torch.zeros(1, dtype=torch.int32)
            cache.update_kvcache_fp16(
                k_in=k[b].data_ptr(), v_in=v[b].data_ptr(), layer_id=layer,
                block_table=block_table[b].data_ptr(), batch_size=1, max_block_num=table_stride,
                cache_seqlens=past_length.data_ptr(), q_len=length, backend=backend,
            )

    worst_output, worst_lse, comparisons = 0.0, 0.0, 0
    for step in range(steps + 1):
        q = torch.randn(batch, 1, q_heads, head_dim, generator=generator).half()
        if mode != "random":
            q.zero_()
            q[..., 0] = 1.0
        expected_lengths = initial_lengths + step
        reference, reference_lse = flash_attention(
            q, k, v, expected_lengths, flash_attn_with_kvcache, device)
        if step:
            positions = (initial_lengths + step - 1).long()
            new_k = k[torch.arange(batch), positions].unsqueeze(1).contiguous()
            new_v = v[torch.arange(batch), positions].unsqueeze(1).contiguous()

        for layer in range(layers):
            output = torch.full_like(q, torch.nan)
            output_lse = torch.full((batch, 1, q_heads), torch.nan, dtype=torch.float32)
            args = dict(
                q_in=q.data_ptr(), output=output.data_ptr(), attn_lse=output_lse.data_ptr(),
                layer_idx=layer, generate_token_idx=step, q_len=1, batch_size=batch,
                max_block_num=table_stride, block_table=block_table.data_ptr(),
                cache_seqlens=layer_lengths[layer].data_ptr(), backend=backend,
            )
            if step == 0:
                cache.attn(**args)
            else:
                cache.attn_with_kvcache(k_in=new_k.data_ptr(), v_in=new_v.data_ptr(), **args)
            # 绑定内为同步调用，返回后即可读取结果，不需要 CPUInfer.submit/sync。
            torch.testing.assert_close(layer_lengths[layer], expected_lengths, rtol=0, atol=0)
            label = f"Hq={q_heads}, block={block_len}, threads={threads}, {mode}, step={step}, layer={layer}"
            err, lse_err = check_output(label, output, output_lse, reference, reference_lse)
            worst_output, worst_lse = max(worst_output, err), max(worst_lse, lse_err)
            comparisons += 1

    # 在同一缓存对象上重新使用全零长度，检查累计状态是否正确重置。
    empty_lengths = torch.zeros(batch, dtype=torch.int32)
    empty_output = torch.full_like(q, torch.nan)
    empty_lse = torch.full((batch, 1, q_heads), torch.nan, dtype=torch.float32)
    cache.attn(q.data_ptr(), empty_output.data_ptr(), empty_lse.data_ptr(),
               0, 0, 1, batch, table_stride, block_table.data_ptr(), empty_lengths.data_ptr(), backend)
    check_output("all-empty KV contract", empty_output, empty_lse,
                 torch.zeros(q.shape, dtype=torch.float32), torch.full_like(empty_lse, -torch.inf))
    print(f"PASS Hq={q_heads:2} block={block_len:3} threads={threads:2} {mode:8} "
          f"max_abs={worst_output:.6g} lse_max_abs={worst_lse:.6g}", flush=True)
    return comparisons + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, nargs="+", default=[4])
    parser.add_argument("--block-lens", type=int, nargs="+", default=[128])
    parser.add_argument("--q-heads", type=int, nargs="+", choices=[32, 64], default=[32, 64])
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--long-context", type=int, default=0, help="额外添加一个指定长度的序列")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0", help="FlashAttention 使用的 CUDA 设备")
    args = parser.parse_args()
    if (any(n <= 0 for n in args.threads) or any(n <= 0 or n % 8 for n in args.block_lens)
            or args.decode_steps < 0 or args.long_context < 0):
        parser.error("threads 必须为正，block_len 必须为正的 8 倍数，steps/context 不能为负")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("此对照测试需要 CUDA PyTorch 和支持 FlashAttention 的 GPU。")
    try:
        from flash_attn import flash_attn_with_kvcache
    except ImportError as error:
        raise SystemExit("无法导入 flash_attn_with_kvcache；请安装匹配当前 PyTorch/CUDA 的 flash-attn。\n"
                         f"原始错误：{error}") from error
    try:
        ext = importlib.import_module("kt_kernel_ext")
    except ImportError as error:
        raise SystemExit("无法导入 kt_kernel_ext；请在已编译本仓库扩展的 Linux Python 环境运行。\n"
                         f"原始错误：{error}") from error
    if not hasattr(ext, "dense_kvcache"):
        raise SystemExit("当前 kt_kernel_ext 没有 dense_kvcache，请重新编译本仓库版本。")
    torch.set_num_threads(1)
    print(f"PyTorch {torch.__version__}; GPU: {torch.cuda.get_device_name(device)}; "
          f"pybind module: {ext.__file__}", flush=True)
    count = 0
    for heads in args.q_heads:
        for block in args.block_lens:
            for threads in args.threads:
                for mode in ["random", "zero_lse", "extreme"]:
                    count += run_case(ext, flash_attn_with_kvcache, device,
                                      heads, block, threads, args.decode_steps,
                                      mode, args.seed, args.long_context)
    print(f"PASS: {count} 次检查（FlashAttention / CPU 输出与 LSE、追加长度及空 KV 约定）。", flush=True)


if __name__ == "__main__":
    main()
