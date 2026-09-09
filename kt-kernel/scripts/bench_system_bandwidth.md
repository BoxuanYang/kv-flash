# System bandwidth benchmark

Run from the repository root on the Linux inference server:

```bash
bash kt-kernel/scripts/bench_system_bandwidth.sh
bash kt-kernel/scripts/bench_system_bandwidth.sh --size-mib 1024 --threads 1,8,16,32,64 --remote
bash kt-kernel/scripts/bench_system_bandwidth.sh --gpu
```

Dependencies: Python 3 and a C++ compiler with OpenMP (default `g++`). Optional
`numactl` enables local, interleaved and remote NUMA memory placement. Optional
CUDA PyTorch enables `--gpu`. Nothing is downloaded or installed; root is not
required. NUMA policy restrictions in containers are reported as failures.

The default CPU test allocates **three 512 MiB arrays** and scans powers of two
up to the allowed physical core count. It honors the process CPU affinity and
uses one logical CPU per physical core. All-node tests distribute selected
cores across NUMA nodes. Per-node tests skip requested counts larger than the
available cores. Two warmup passes precede seven timed passes for each kernel.

Results are saved in a new `build/bandwidth/<timestamp>/` directory:

- `cpu.csv`: read, copy and triad bandwidth for each placement/thread count.
- `gpu.csv`: optional pinned host-to-device and device-to-host copy bandwidth.
- `machine.json`: topology, memory, compiler and GPU information.
- `run.log`, `summary.json`: results and explicit failures.
- Generated C++ source and executable, for reproducibility.

`--output PATH` selects a new result directory. The script refuses to overwrite
an existing directory and returns nonzero if any requested test fails.

## Interpretation

Bandwidth is **effective GB/s (10^9 bytes/s)**. Read counts N bytes, copy counts
2N bytes, and triad (`z = x + 3*y`) counts 3N bytes per pass. Stores may incur
extra write-allocate traffic, so these are not physical DRAM bus counters or
official STREAM results. CPU timing includes OpenMP scheduling/barrier overhead.
The output includes median and best bandwidth, not just the fastest sample.

Choose each array comfortably larger than the machine's LLC, especially for
the read test. Increase `--size-mib` on large-cache servers, while leaving memory
for the OS and other processes. The script checks available system memory and
visible cgroup-v2 limits but cannot guarantee free memory on each NUMA node.
Run on an otherwise idle server. CPU frequency, NUMA placement and background
traffic affect results. A single NUMA-node test allocates all three arrays on
that memory node.

GPU copies run sequentially after the CPU tests, one GPU/direction at a time.
They use CUDA event timing and pinned host memory with the process's default
NUMA policy; they do not measure GPU HBM bandwidth, bidirectional saturation,
or CPU-attention/MoE contention. CPU read/copy/triad similarly measure streaming
memory access, not actual model kernels. These results establish the machine's
bandwidth baseline before benchmarking concurrent attention and MoE.

Quick functionality check (cache-sized; **not** a DRAM performance result):

```bash
bash kt-kernel/scripts/bench_system_bandwidth.sh --size-mib 8 --threads 1,2 --iterations 3
```
