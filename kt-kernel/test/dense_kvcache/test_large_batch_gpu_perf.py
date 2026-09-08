#!/usr/bin/env python
# coding=utf-8
"""Fill missing GPU timings in large_batch_cpuinfer.txt using ordinary FlashAttention.

No CUDA Graph, no CPU attention, and no kt_kernel_ext dependency. CPU timings and
the existing correctness status are preserved. GPU inputs are FP16, KV heads=4,
head_dim=128, query length=1, with the same scaling/GQA as test_cpu_perf.py.
"""

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import statistics
import tempfile


REPORT_PATH = Path(__file__).resolve().parents[3] / "large_batch_cpuinfer.txt"
HEADER = (f"{'Batch':>6} {'Seq':>6} {'Hq':>4} {'Block':>6} {'Threads':>7} "
          f"{'CPU ms':>10} {'GPU ms':>10} {'CPU/GPU':>9} {'Checks':>8}")


@dataclass
class ResultRow:
    batch: int
    length: int
    heads: int
    block: int
    threads: int
    cpu_ms: str
    gpu_ms: str
    ratio: str
    checks: str

    def format(self):
        return (f"{self.batch:6} {self.length:6} {self.heads:4} {self.block:6} {self.threads:7} "
                f"{self.cpu_ms:>10} {self.gpu_ms:>10} {self.ratio:>9} {self.checks:>8}")

    def update_ratio(self):
        if self.cpu_ms != "--" and self.gpu_ms != "--":
            self.ratio = f"{float(self.cpu_ms) / float(self.gpu_ms):.2f}"
        else:
            self.ratio = "--"


def parse_report(text):
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines or lines[0].split() != HEADER.split():
        raise ValueError("Unexpected report header; expected Batch/Seq/Hq/Block/Threads/CPU ms/GPU ms/CPU/GPU/Checks")
    rows = []
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split()
        if len(fields) != 9:
            raise ValueError(f"Report line {line_number}: expected 9 columns")
        row = ResultRow(*map(int, fields[:5]), *fields[5:])
        if (min(row.batch, row.length, row.block, row.threads) <= 0
                or row.heads not in (32, 64) or row.block % 8):
            raise ValueError(f"Report line {line_number}: invalid decode shape")
        for value in (row.cpu_ms, row.gpu_ms):
            if value != "--" and (not math.isfinite(float(value)) or float(value) <= 0):
                raise ValueError(f"Report line {line_number}: timing must be positive and finite")
        if row.checks not in ("PASS", "FAIL"):
            raise ValueError(f"Report line {line_number}: expected PASS or FAIL")
        rows.append(row)
    if not rows:
        raise ValueError("Report has no experiment rows")
    return rows


def save_report(path, rows, previous_text):
    """Replace atomically and refuse to overwrite edits made while benchmarking."""
    if path.read_text(encoding="utf-8") != previous_text:
        raise RuntimeError("Report changed during measurement; refusing to overwrite it")
    text = HEADER + "\n" + "\n".join(row.format() for row in rows) + "\n"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=path.name + ".", delete=False) as output:
            temporary_path = Path(output.name)
            output.write(text)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return text


def measure_gpu(torch, flash_attention, row, args):
    """Median CUDA-event latency; allocations/transfers for inputs precede warmup."""
    # GPU uses contiguous logical KV. CPU block length/worker count are not GPU parameters.
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    query = torch.randn(row.batch, 1, row.heads, 128, dtype=torch.float16,
                        device=args.device, generator=generator)
    keys = torch.randn(row.batch, row.length, 4, 128, dtype=torch.float16,
                       device=args.device, generator=generator)
    values = torch.randn(keys.shape, dtype=torch.float16, device=args.device, generator=generator)
    lengths = torch.full((row.batch,), row.length, dtype=torch.int32, device=args.device)

    def run():
        return flash_attention(q=query, k_cache=keys, v_cache=values,
                               cache_seqlens=lengths, softmax_scale=128 ** -0.5,
                               causal=True, return_softmax_lse=True)

    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(args.rounds):
        for _ in range(args.warmup):
            run()
        torch.cuda.synchronize(args.device)
        for _ in range(args.iterations):
            start.record()
            run()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))

    # Sanity check outside timing; this does not replace the original CPU/GPU correctness test.
    output, lse = run()
    if not torch.isfinite(output).all().item() or not torch.isfinite(lse).all().item():
        raise RuntimeError("FlashAttention returned nonfinite output or LSE")
    median = statistics.median(samples)
    if not math.isfinite(median) or median <= 0:
        raise RuntimeError(f"Invalid GPU median: {median}")
    return median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.rounds <= 0:
        parser.error("warmup must be nonnegative; iterations and rounds must be positive")
    previous_text = REPORT_PATH.read_text(encoding="utf-8")
    rows = parse_report(previous_text)
    missing = [row for row in rows if row.gpu_ms == "--"]
    if missing:
        import torch
        from flash_attn import flash_attn_with_kvcache
        if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA PyTorch and FlashAttention are required")
        torch.set_num_threads(1)
        print(f"GPU: {torch.cuda.get_device_name(args.device)}; {len(missing)} missing rows; no CUDA Graph", flush=True)
        print("Input setup excluded; median CUDA-event time. Existing Checks column is preserved.", flush=True)
        with torch.inference_mode(), torch.cuda.device(args.device):
            for row in missing:
                median = measure_gpu(torch, flash_attn_with_kvcache, row, args)
                row.gpu_ms = f"{median:.4f}" if median >= 0.0001 else f"{median:.8f}"
                row.update_ratio()
                previous_text = save_report(REPORT_PATH, rows, previous_text)
                print(row.format(), flush=True)
    for row in rows:
        row.update_ratio()
    save_report(REPORT_PATH, rows, previous_text)
    print(f"\nUpdated: {REPORT_PATH}\n{HEADER}")
    for row in rows:
        print(row.format())


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"GPU benchmark failed: {type(error).__name__}: {error}") from error
