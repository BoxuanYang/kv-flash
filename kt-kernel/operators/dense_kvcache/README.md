# Dense Qwen3 CPU decode attention

支持 FP16 Qwen3-30B-A3B / Qwen3-235B-A22B 的完整 attention head 形状：

| 模型 | Q heads | KV heads | head_dim | GQA |
|---|---:|---:|---:|---:|
| 30B | 32 | 4 | 128 | 8 |
| 235B | 64 | 4 | 128 | 16 |

层数可按实际缓存的层数配置（完整模型为 48 / 94，也可只缓存部分层）。
当前入口不接受张量并行切分后的 head 形状。Q/K 必须由调用方完成 Norm 和 RoPE。
K/V/query/output 使用 FP16，score、单块输出、累计输出和 LSE 使用 FP32。
`block_len` 为正的 8 倍数，保留现有 GEMM 的完整 block 维度。

## 阅读顺序与修改位置

1. `attn_with_kvcache()`：追加一个 token，然后调用 `attn()`。
2. `attn()`：检查 decode 调用参数，初始化状态，再执行并行计算。
3. `attn_initialize_kvhead_()`：重置结果，构建有效任务前缀和。
4. `attention_kvhead_()`：领取四块任务、任务内归并，再执行两阶段 reduce（默认）或带锁提交。
5. `attn_with_kvcache_one_block_()`：QK、稳定 softmax、FP16 probability、PV。
6. `ThreadResize()` / `BatchResize()` / `BlockResize()`：预分配工作区和缓存。

未引入新的 Context / Workspace 类，也未更换 WorkerPool 或四层 vector 缓存布局。

## 数据布局与长度

- Q / attention output：`[batch, 1, q_heads, 128]`。
- 新 K/V：`[batch, q_len, 4, 128]`；批量导入允许 q_len > 1。
- 内部 K block：`[block_len, 128]`。
- 内部 V block：`[128, block_len]`。
- block_table：`[batch, max_block_num]`；这里的 `max_block_num` 是表的行步长。
- config.max_block_num：每层物理 block 的容量，与表的行步长不同。

`attn()` 的 cache_seqlens 表示参与 attention 的有效 KV 数，调用不修改它。
`update_kvcache_fp16()` 按写入前长度追加 K/V，但不更新长度。
`attn_with_kvcache()` 接受写入前长度，并原地将每个序列的长度加 1。
跨层调用不能把第一层递增过的数组直接作为第二层的写入前长度；每层需得到相同的写入前长度。

公开 attention 签名中的 generate_token_idx、q_len 保留以兼容已有绑定。
generate_token_idx 不参与计算；两个 attention 入口均在 Release 构建中拒绝 q_len != 1。
多 token KV 导入不等于 prefill attention，此能力用于接收上游产生的历史 KV。

## 有效任务编号：attention_kvhead_()

设序列 b 的有效 block 数为 `N[b] = ceil(cache_seqlens[b] / block_len)`。
每个 head 的任务数为 `C[b] = ceil(N[b] / 4)`，前缀数组为：

```text
task_offsets[0] = 0
task_offsets[b+1] = task_offsets[b] + kv_head_num * C[b]
```

任务按 `(batch, KV head, 四块分组)` 排列，任务数为 `kv_head_num * sum(C)`。
通过 upper_bound 查出 batch，再用该序列的 C 解出 head/分组。
每组处理 `[4 * group, min(4 * group + 4, N[b]))` 的逻辑 block；不会跨 sequence/head。
逻辑连续不要求物理页连续，每个 block 单独查表、检查物理页范围并调用原 GEMM。
访问 block_table 时仍使用原表行步长，不使用 N。
空序列没有任务；整个 batch 都为空时不调用线程池。

长序列拆成多个四块任务，由多个线程动态领取；尾任务允许只有 1～3 个 block。
当前 WorkerPool 的这条接口只使用第 0 个子池；容量检查也据此读取该子池配置。

## 单块输出与同步

单块 kernel 仍使用线程私有的 FP32 输出/LSE 和 FP16 probability 工作区。
同一 task 的最多 4 个 block 使用稳定 LSE 合并公式在本地累计，不改变物理 block 大小或 GEMM。

默认 `set_parallel_reduce(true)` 路径：

```text
最多四次单块 kernel -> 线程私有累计 O/LSE -> 每 task 一份独占暂存结果
                                               |
                                      所有计算 task 完成
                                               v
                           每个 (batch, query head) 独占输出的 reduce
                                               |
                                      最终 FP16 输出转换
```

只有完整成功的 task 才复制结果到暂存槽；任意 block 的物理页非法或 GEMM 失败时，
主线程等待计算结束后抛错，不启动 reduce。空序列输出为零、LSE 为负无穷。

`set_parallel_reduce(false)` 使用相同四块 task，保留同一线程连续领取同一 head 时的累计；
切换 batch/head 或线程退出时，使用该 head 的 mutex 提交结果。
两种模式都保留单块和累计结果的独立工作区，多线程结果不要求逐位一致。
同一个 KVCache / WorkerPool 不支持重叠调用；Resize 必须在没有计算时执行。

## 单独的正确性修复

- 单块 softmax 减去有效前缀最大值，尾部 probability 置零；满块和尾块共享一个调用。
- LSE 合并使用 max/min 形式，避免对正的大差值求 exp。
- 首次全局提交由独立字节标志记录；合法 LSE=0 不再被当作未初始化。
- 空序列输出约定为 0、LSE 为负无穷。
- GEMM 返回 false 或有效表项指向非法物理 block 时，工作线程记录失败，调用线程在同步结束后抛错。
- `clear_kvcache_all_layers()` 不再多清理整块边界后的下一个 block。
- 写入前检查目标表容量/物理 block；Resize 同步更新容量值。

仍按完整 block 执行 PV；尾部 V 存储应保持有限值（本实现分配时清零）。
不使用尾部 probability=0 来保证屏蔽 NaN/Inf 的 V。
辅助导出、快照函数保留；它们不是正常 decode 的必要路径。
用户此前删除了 qwen3_cpu_ops 源文件，本次同步移除 ext_bindings.cpp 中的悬空 include 和两个绑定。

## 独立验证

Linux x86-64 需要 g++、libnuma-dev、libhwloc-dev，以及 AVX2/FMA/F16C。
在仓库根目录运行：

```bash
bash kt-kernel/test/dense_kvcache/run.sh release
bash kt-kernel/test/dense_kvcache/run.sh sanitize
bash kt-kernel/test/dense_kvcache/run.sh release --bench
```

测试使用本仓库的 AVX2 GEMM（含 IQK 分派）和真实 WorkerPool，参考值为完整序列 FP64 attention。
检查 GQA=8/16、1/4 线程、block_len=8/32/128、异长/空序列、随机物理映射、四块分组边界、1～3 块尾任务、部分有效尾块、
跨两层连续追加、合法 LSE=0、极端 score、错误恢复和容量变化。
输出容差为 `0.002 + 0.002 * abs(reference)`，LSE 绝对容差 0.004；
包含原实现就存在的 probability FP16 舍入误差。

sanitize 检查 dense 实现和 GEMM 包装的 ASan/UBSan；依赖库未全部插桩。
关闭 LeakSanitizer，因为现有 WorkerPool 的 hwloc topology 生命周期不在本次修改范围内。
这不替代完整 Python/CUDA 扩展构建，也不替代目标服务器上的多 NUMA 吞吐测试。

## Python / FlashAttention 对照

测试文件：`kt-kernel/test/dense_kvcache/test_flash_attention.py`。
在同一 Linux Python 环境中准备 CUDA PyTorch、支持当前 GPU 的 `flash-attn`，
以及从本仓库编译、包含 `dense_kvcache` 绑定的 `kt_kernel_ext`。在仓库根目录运行：

```bash
python kt-kernel/test/dense_kvcache/test_flash_attention.py
python kt-kernel/test/dense_kvcache/test_flash_attention.py --threads 1 4 --block-lens 32 128 --long-context 8192
```

`flash_attention()` 调用 FlashAttention 的 `flash_attn_with_kvcache()`，使用 CUDA FP16
连续 KV、每条序列的有效长度、`causal=True` 和 `1/sqrt(head_dim)` 缩放，同时获取输出与 LSE。
接口依据：[FlashAttention 源码](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py)。
`run_case()` 将相同 FP16 数据通过现有 pybind 接口传给 CPU 的 `update_kvcache_fp16()`、
`attn()` 和 `attn_with_kvcache()`；CPU 使用随机物理 block 映射，参考端按连续逻辑序列计算。
不需要增加 C++ 测试绑定。Q/K 视为已完成 Norm/RoPE；历史导入只写 KV，不计算 prefill attention。

覆盖两种 Qwen3 head 形状、异长 batch、整块与尾块、两层连续 decode、合法 LSE=0 及极端 score。
`check_output()` 检查输出（绝对/相对容差均为 0.002）和 LSE（绝对容差 0.004），失败会抛出断言。
空 KV 行不调用 FlashAttention，而是单独检查 CPU 的输出为 0、LSE 为负无穷的约定。
脚本包含 CPU/GPU 数据传输，仅用于正确性检查，不用于性能对比。

本地仅完成该 Python 脚本的语法和命令行检查；当前可用 PyTorch 为 CPU 版本，
尚未运行 FlashAttention 与 pybind 的端到端对照。下文 C++ 验证结果不代表此项对照已通过。

## 四块任务验证（2026-09-09）

在 WSL Ubuntu 使用真实 AVX2 GEMM 和 WorkerPool 验证：two-phase / locked 两种模式
均通过 Release 和 ASan/UBSan 的 20 组场景及额外错误恢复检查。
覆盖完整四块任务、1～3 块尾任务、部分有效尾块、空序列、随机物理页映射、
任务中途遇到负数/越界物理页后的失败恢复，以及跨任务的极端 score 和 FP64 参考对照。
两种模式的 32/64 线程 profiling smoke test 均通过；每组 3 次调用共 6144 个计算 task，
two-phase 模式另有 6144 个 reduce task，flush 为 0。
这次未测目标服务器长上下文性能，以下旧版性能数据不代表四块任务版本的加速比。

## 历史验证结果（2026-09-07）

本机 WSL Ubuntu、Intel i9-13900HX，使用 AVX2/FMA/F16C，不依赖 BF16。
Release 使用 `-O3 -ffast-math`；独立 FP64 参考检查不启用 fast-math。
最终 Release 和 ASan/UBSan 两种构建均通过 16 组形状/线程/边界场景及额外错误恢复检查。
`git diff --check` 通过；尚未构建完整 Python/CUDA 扩展。

性能对照使用修改开始前的工作区副本，只修正了阻止编译的重复 bool 声明，
以及独立构建所需的头文件路径。旧版与新版使用相同的实际 GEMM 和 WorkerPool。
每种负载预热 20 次、测量 101 次；旧/新版交替顺序共 3 轮，下面取三轮中位耗时的中位数。
均为 batch=64、8 线程、block_len=128，测量单层 attention（不包含 KV 写入或缓存构造）。

| Q heads | 负载 | 旧版 ms | 新版 ms | 耗时下降 |
|---:|---|---:|---:|---:|
| 32 | 等长 512 token | 2.935 | 2.777 | 5.4% |
| 32 | 异长，表预留较大 | 2.231 | 1.716 | 23.1% |
| 64 | 等长 512 token | 5.252 | 5.077 | 3.3% |
| 64 | 异长，表预留较大 | 3.476 | 2.958 | 14.9% |

等长负载的表行步长为 5，有效 block 数为 4；异长负载长度为 `1 + (b*131)%512`，
表行步长为 64。测试计数包装给每次 GEMM 增加了相同的原子计数开销。
结果显示这些样本的中位数改善；单轮存在波动，不代表所有 batch/上下文/NUMA 场景都无退化。
尚未在目标服务器上验证长上下文、多 NUMA 和完整模型端到端性能。

本次保留公开 KVCache API、辅助导出/快照能力，未修改共享 WorkerPool、其他 attention 算子或 GEMM 实现。
