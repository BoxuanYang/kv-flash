#!/usr/bin/env bash
# Linux CPU/NUMA bandwidth benchmark; optional PCIe/NVLink host-device copies.
set -euo pipefail
command -v python3 >/dev/null || { echo 'python3 is required' >&2; exit 1; }
exec python3 - "$@" <<'PY'
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

p = argparse.ArgumentParser(prog='bench_system_bandwidth.sh', description='Measure effective CPU memory and optional CPU/GPU copy bandwidth. No downloads or root required.')
p.add_argument('--size-mib', type=int, default=512, help='MiB per CPU array; THREE arrays are allocated (default: 512)')
p.add_argument('--iterations', type=int, default=7, help='measured passes per kernel after 2 warmups (default: 7)')
p.add_argument('--threads', help='comma-separated thread counts; default: powers of two up to physical cores')
p.add_argument('--remote', action='store_true', help='also test every remote NUMA CPU/memory node pair')
p.add_argument('--gpu', action='store_true', help='also benchmark pinned H2D/D2H copies using installed CUDA PyTorch')
p.add_argument('--gpu-size-mib', type=int, default=256, help='buffer size for each GPU copy (default: 256)')
p.add_argument('--output', type=Path, help='new result directory; default: build/bandwidth/<timestamp>')
a = p.parse_args()
if min(a.size_mib, a.gpu_size_mib) < 1 or a.iterations < 3:
    p.error('sizes must be positive and --iterations must be at least 3')
try:
    requested = sorted(set(int(x) for x in a.threads.split(','))) if a.threads else None
    if requested is not None and (not requested or requested[0] < 1):
        raise ValueError()
except ValueError:
    p.error('--threads must contain positive comma-separated integers')
if not sys.platform.startswith('linux'):
    p.error('this benchmark requires Linux')
cxx = shutil.which(os.environ.get('CXX', 'g++'))
if not cxx:
    p.error('g++ with OpenMP support is required (or set CXX to a compiler executable)')

out = a.output or Path('build/bandwidth') / datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
out.mkdir(parents=True, exist_ok=False)
out = out.resolve()
print(f'Results: {out}', flush=True)
log = (out / 'run.log').open('w', buffering=1)
def note(s):
    print(s, flush=True)
    print(s, file=log)

def capture(cmd):
    r = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return r.stdout

metadata = {'arguments': vars(a) | {'output': str(out)}, 'uname': capture(['uname', '-a']),
            'compiler': capture([cxx, '--version'])}
for cmd in (['lscpu'], ['numactl', '--hardware'], ['nvidia-smi']):
    if shutil.which(cmd[0]):
        metadata[cmd[0]] = capture(cmd)
metadata['meminfo'] = Path('/proc/meminfo').read_text()

# Respect cpusets and use one hardware thread per physical core.
allowed = sorted(os.sched_getaffinity(0))
cores, seen = {}, set()
for cpu in allowed:
    base = Path(f'/sys/devices/system/cpu/cpu{cpu}')
    key = tuple((base / 'topology' / x).read_text().strip() for x in ('physical_package_id', 'core_id'))
    if key in seen:
        continue
    seen.add(key)
    nodes = list(base.glob('node[0-9]*'))
    node = int(nodes[0].name[4:]) if nodes else 0
    cores.setdefault(node, []).append(cpu)
all_cpus = sorted(c for group in cores.values() for c in group)
metadata['physical_cpu_ids_by_node'] = cores
(out / 'machine.json').write_text(json.dumps(metadata, indent=2, default=str))

available = int(next(x.split()[1] for x in metadata['meminfo'].splitlines() if x.startswith('MemAvailable:'))) * 1024
# Also account for a nested cgroup-v2 memory limit when available.
for line in Path('/proc/self/cgroup').read_text().splitlines():
    if line.startswith('0::'):
        group = Path('/sys/fs/cgroup') / line[3:].lstrip('/')
        while group == Path('/sys/fs/cgroup') or Path('/sys/fs/cgroup') in group.parents:
            try:
                limit = (group / 'memory.max').read_text().strip()
                if limit != 'max':
                    available = min(available, int(limit) - int((group / 'memory.current').read_text()))
            except OSError:
                pass
            if group == Path('/sys/fs/cgroup'):
                break
            group = group.parent
if 3 * a.size_mib * 2**20 > available * 0.6:
    p.error('three CPU arrays exceed 60% of available memory; reduce --size-mib')
note(f'Physical cores allowed: {len(all_cpus)}; arrays: 3 x {a.size_mib} MiB')
note('GB/s uses decimal bytes and useful array traffic, NOT memory-controller counters.')
note('Use an array size comfortably larger than LLC. Small smoke runs measure cache, not DRAM.')

source = r'''
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <omp.h>
using Clock = std::chrono::steady_clock;
volatile double checksum_sink = 0;
int main(int argc, char** argv) {
    if (argc != 4) return 2;
    const size_t n = std::stoull(argv[1]) * 1024 * 1024 / sizeof(double);
    const int repetitions = std::atoi(argv[2]), threads = std::atoi(argv[3]);
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    double *x=nullptr, *y=nullptr, *z=nullptr;
    if (posix_memalign((void**)&x, 64, n*8) || posix_memalign((void**)&y, 64, n*8) ||
        posix_memalign((void**)&z, 64, n*8)) { std::fprintf(stderr, "allocation failed\n"); return 3; }
    // Parallel first-touch follows the selected CPU/memory policy.
    #pragma omp parallel for schedule(static)
    for (size_t i=0; i<n; ++i) { x[i]=1.0; y[i]=2.0; z[i]=0.0; }
    const char* names[] = {"read", "copy", "triad"};
    const int traffic[] = {1, 2, 3};
    for (int kernel=0; kernel<3; ++kernel) {
        std::vector<double> times;
        for (int r=-2; r<repetitions; ++r) {
            double sum=0;
            auto start=Clock::now();
            if (kernel==0) {
                #pragma omp parallel for simd schedule(static) reduction(+:sum)
                for (size_t i=0; i<n; ++i) sum += x[i];
            } else if (kernel==1) {
                #pragma omp parallel for simd schedule(static)
                for (size_t i=0; i<n; ++i) z[i]=x[i];
            } else {
                #pragma omp parallel for simd schedule(static)
                for (size_t i=0; i<n; ++i) z[i]=x[i]+3.0*y[i];
            }
            // Keep all stores observable even under optimization.
            asm volatile("" ::: "memory");
            double dt=std::chrono::duration<double>(Clock::now()-start).count();
            checksum_sink = kernel==0 ? sum : z[n/2];
            if (kernel==0 && std::abs(sum-double(n)) > double(n)*1e-9) return 4;
            if (r>=0) times.push_back(dt);
        }
        if (kernel) {
            int bad=0;
            const double expected=kernel==1 ? 1.0 : 7.0;
            #pragma omp parallel for reduction(|:bad)
            for (size_t i=0; i<n; ++i) bad |= (z[i]!=expected);
            if (bad) { std::fprintf(stderr, "validation failed\n"); return 4; }
        }
        std::sort(times.begin(), times.end());
        size_t m=times.size()/2;
        double median=times.size()%2 ? times[m] : (times[m-1]+times[m])/2;
        double bytes=double(n)*8*traffic[kernel];
        std::printf("%s,%.6f,%.6f,%.6f\n", names[kernel], bytes/median/1e9,
                    bytes/times.front()/1e9, median*1e3);
    }
    std::free(x); std::free(y); std::free(z);
}
'''
src, exe = out / 'memory_bandwidth.cpp', out / 'memory_bandwidth'
src.write_text(source)
compile_cmd = [cxx, '-O3', '-march=native', '-fopenmp', '-std=c++17', str(src), '-o', str(exe)]
note('Compile: ' + ' '.join(compile_cmd))
subprocess.run(compile_cmd, check=True)

def counts(limit):
    if requested:
        return [x for x in requested if x <= limit]
    return sorted({limit} | {2**i for i in range(limit.bit_length()) if 2**i <= limit})

plans = [('all_first_touch', all_cpus, [])]
numactl = shutil.which('numactl')
if numactl:
    plans = [('all_interleave', all_cpus, [numactl, '--interleave=' + ','.join(map(str, sorted(cores)))])]
    for node, cpus in sorted(cores.items()):
        plans.append((f'local_cpu{node}_mem{node}', cpus, [numactl, f'--membind={node}']))
        if a.remote:
            for other in sorted(cores):
                if other != node:
                    plans.append((f'remote_cpu{node}_mem{other}', cpus, [numactl, f'--membind={other}']))
else:
    note('numactl absent: only parallel first-touch placement will be tested.')
if not counts(len(all_cpus)):
    p.error('no requested thread count fits the available physical cores')

cpu_runs = 0
failures = []
with (out / 'cpu.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['placement', 'threads', 'cpu_ids', 'array_mib', 'kernel', 'median_GBps', 'best_GBps', 'median_ms'])
    for label, cpus, prefix in plans:
        for threads in counts(len(cpus)):
            # Spread all-node tests across NUMA nodes rather than filling one socket first.
            order = [group[i] for i in range(max(map(len, cores.values())))
                     for group in cores.values() if i < len(group)] if cpus == all_cpus else cpus
            chosen = order[:threads]
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS=str(threads), OMP_DYNAMIC='FALSE', OMP_PROC_BIND='TRUE',
                       OMP_PLACES=','.join('{%d}' % c for c in chosen))
            # Avoid inherited affinity settings overriding the explicit placement.
            for key in ('GOMP_CPU_AFFINITY', 'KMP_AFFINITY'):
                env.pop(key, None)
            cmd = prefix + [str(exe), str(a.size_mib), str(a.iterations), str(threads)]
            r = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if r.returncode:
                failures.append(f'{label}/{threads}: {r.stderr.strip() or r.returncode}')
                note('FAILED: ' + failures[-1])
                continue
            if r.stderr:
                note(r.stderr.strip())
            for row in csv.reader(r.stdout.splitlines()):
                writer.writerow([label, threads, ','.join(map(str, chosen)), a.size_mib] + row)
                note(f'{label:26s} threads={threads:3d} {row[0]:5s} median={float(row[1]):8.2f} GB/s best={float(row[2]):8.2f} GB/s')
            f.flush()
            cpu_runs += 1

if a.gpu:
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA PyTorch or a visible CUDA GPU is unavailable')
        with (out / 'gpu.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['device', 'name', 'direction', 'buffer_mib', 'median_GBps', 'best_GBps'])
            for device in range(torch.cuda.device_count()):
                with torch.cuda.device(device):
                    nbytes = a.gpu_size_mib * 2**20
                    host = torch.ones(nbytes, dtype=torch.uint8, pin_memory=True)
                    gpu = torch.empty(nbytes, dtype=torch.uint8, device=f'cuda:{device}')
                    for direction in ('H2D', 'D2H'):
                        dst, origin = (gpu, host) if direction == 'H2D' else (host, gpu)
                        timings = []
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        for step in range(a.iterations + 2):
                            start.record()
                            dst.copy_(origin, non_blocking=True)
                            end.record()
                            end.synchronize()
                            if step >= 2:
                                timings.append(start.elapsed_time(end) / 1000)
                        median, best = nbytes/statistics.median(timings)/1e9, nbytes/min(timings)/1e9
                        writer.writerow([device, torch.cuda.get_device_name(device), direction, a.gpu_size_mib, median, best])
                        note(f'GPU {device} {direction}: median={median:.2f} GB/s best={best:.2f} GB/s')
                    del host, gpu
    except Exception as e:
        failures.append(f'GPU test: {e}')
        note('FAILED: ' + failures[-1])

(out / 'summary.json').write_text(json.dumps({'successful_cpu_runs': cpu_runs, 'failures': failures}, indent=2))
note(f'Finished: {out}')
if not cpu_runs or failures:
    sys.exit(1)
PY
