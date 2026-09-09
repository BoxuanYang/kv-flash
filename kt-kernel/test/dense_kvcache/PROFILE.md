# CPU task profiling

在实验室 Linux 机器上，进入已有依赖的 Python 环境，在仓库根目录重新编译并运行：

```bash
CPUINFER_FORCE_REBUILD=1 python -m pip install ./kt-kernel --no-build-isolation
numactl --cpunodebind=0 --membind=0 python kt-kernel/test/dense_kvcache/test_cpu_profile.py
```

沿用 `test_cpu_perf.py` 的输入准备、NUMA 0 线程池和 FlashAttention 正确性检查。
无需 sudo 或 perf。固定两组配置：B=64、Seq=4096、Block=128，线程数分别为 32 和 64。
默认 Hq=32、Hkv=4、D=128；可用 `--q-heads 64` 切换 Hq，两组测试保持相同模型形状。
默认每组 3 轮，每轮 20 次预热、200 次正式 attention；可用 `--rounds`、`--warmup`、`--iterations` 调整。

每组只有正式 attention 参与累计；一组完成且正确性检查通过后，C++ 计算平均值并写文件。
仓库根目录的 `cpu_profile.txt` 第一组覆盖、第二组追加。日志和错误只在终端输出。
正常情况下，每次 attention 完成 `64 * 4 * (4096 / 128) = 8192` 个 task；
默认每组总计 `600 * 8192 = 4915200` 个 task，文件写入前会检查总数。

文件开头解释全部指标，接着每组依次列出墙钟时间、阶段汇总、各线程累计值和各线程每 task 平均值。
所有时间单位为 ms。线程平均值的分母为该线程实际执行的 task 数；全线程汇总为加权平均。
先看汇总表的 `share_pct` 和 `max_avg`，再对照逐线程 `tasks` 和平均值判断瓶颈或任务分配差异。
`sync` 只计输出 mutex 的等待，`writeback` 计持锁合并及解锁；二者包含最后一次 flush，均按 task 数摊销。
`gap` 是回调之间的实测间隙，包含取任务、计时代码和系统调度，不能据此单独断言工作窃取或 NUMA 是根因。
`wall` 的平均分母单独使用 attention 调用次数；并行线程时间之和不能当作墙钟延迟。

默认关闭 profiling；原来的性能测试不自动开启计时。新增实现使用普通数组、循环和 `fprintf`，
chrono 仅封装在一个返回毫秒数的小函数里。每线程数组行留出间隔，避免统计变量之间的伪共享。
启停、清零、导出必须在同步 `attn` 返回后调用。
细粒度计时会扰动结果；评估真实加速比时继续使用关闭 profiling 的性能测试。

本地验证（真实 C++ GEMM 和线程池，无需 CUDA）：

```bash
bash kt-kernel/test/dense_kvcache/run.sh
bash kt-kernel/test/dense_kvcache/run.sh release --profile-smoke
```

第二条使用共享同一个物理 block 的小型数据，检查两组计数、重置和导出；
生成文件在 `build/dense_validation/`，只用于验证计时功能，不是实验室性能数据。
