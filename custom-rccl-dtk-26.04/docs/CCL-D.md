# RCCL 单节点 CCL-D 复现

本文记录基于 `paper/CCL-D.pdf` 和配套论文材料，在 DTK 26.04、`gfx936` RCCL 分支中的改造、运行方式和验证边界。当前目标是先完成单节点可验证链路，并保留通信域、节点编号和 Trace ID 字段供后续多节点汇聚。

## 当前实现

在 `ncclEnqueueCheck()` 完成参数检查后记录每次 collective 的 host enqueue 事件。记录包含 `comm_id`、`counter`、`trace_id`、rank 数、节点编号、操作、元素数、字节数、类型、root 和单调时钟时间。RCCL 的内部 `opCount` 在该插桩点尚未分配，因此这里使用按 communicator 地址维护的独立计数器；Trace ID 格式为 `<16位commHash十六进制>:<counter>`，例如 `f0da14f8a754631c:12`。communicator destroy 被 RCCL 接受后清除对应计数器，防止对象地址复用造成计数延续。

记录器默认关闭；开启后每个进程写独立 JSONL 文件。日志路径支持 `{rank}`、`${RANK}` 或 `%r` 占位符，例如 `rank-{rank}.jsonl`。路径没有占位符时，自动在扩展名前加 `.rank<N>`，避免多个 rank 进程同时追加同一个文件。日志只用于诊断，不参与 RCCL 的调度和通信参数。

分析器按 `(comm_id, counter)` 聚合各 rank 记录，可识别操作、元素数、字节数、类型或 root 不一致；也可在超时后报告该次操作缺少的 rank，并报告 `ENQUEUE_SKEW`（同机 host enqueue 时间偏斜）。`ENQUEUE_SKEW` 只是入口时间观察值，不是通信变慢结论。monitor 支持持续轮询或单次分析，并在 `--fail-on-anomaly` 下以退出码 `2` 表示检测到异常。

## 编译与运行

```bash
source /opt/dtk-26.04/env.sh
cd /workspace/CCL-D/custom-rccl-dtk-26.04-ccl-d
cmake -S . -B /workspace/CCL-D/ccld-artifacts/dtk-26.04-gfx936-try1/build \
  -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx936 -DBUILD_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX=/workspace/CCL-D/ccld-artifacts/dtk-26.04-gfx936-try1/install
cmake --build /workspace/CCL-D/ccld-artifacts/dtk-26.04-gfx936-try1/build -j 16
cmake --install /workspace/CCL-D/ccld-artifacts/dtk-26.04-gfx936-try1/build
```

开启记录并运行 RCCL 单测。关键是通过 `LD_LIBRARY_PATH` 让 UT 加载 CCL-D 安装目录里的 `librccl.so`：

```bash
export LD_LIBRARY_PATH=/workspace/CCL-D/ccld-artifacts/dtk-26.04-gfx936-try1/install/lib:$LD_LIBRARY_PATH
export RCCL_CCLD_ENABLE=1
mkdir -p /tmp/ccld
export RCCL_CCLD_LOG=/tmp/ccld/rank-{rank}.jsonl
/path/to/rccl-UnitTests --gtest_filter='AllReduce.OutOfPlace'
python3 tools/ccld_monitor.py --once --all /tmp/ccld/rank-*.jsonl
```

持续监控可省略 `--once`；`--timeout-ms` 控制缺 rank 的安静等待时间，`--slow-skew-ms` 控制同机 enqueue 时间差阈值。论文指出跨节点时钟不能作为核心判断依据；当前偏斜观察只限单节点，未来 slow 判断需增加各 channel 的 send/recv count 与 rate，而非把跨节点时间戳直接比较。

## 双节点验证

双节点验证使用 MPI 负责启动和广播 RCCL unique ID，RCCL communicator 使用全局 rank，日志写入共享 `/workspace`：

```bash
export LD_LIBRARY_PATH=/workspace/CCL-D/ccld-artifacts/dtk-26.04-gfx936-try1/install/lib:$LD_LIBRARY_PATH
export RCCL_CCLD_ENABLE=1
export RCCL_CCLD_LOG=/workspace/CCL-D/ccld-artifacts/multinode-logs/rank-{rank}.jsonl
/opt/mpi/bin/mpirun --allow-run-as-root --host HPDPS-6,HPDPS-7 -np 2 \
  --map-by ppr:1:node --mca plm_rsh_args '-p 2234' \
  -x LD_LIBRARY_PATH -x RCCL_CCLD_ENABLE -x RCCL_CCLD_LOG \
  /workspace/CCL-D/ccld-artifacts/multinode-test/multinode_allreduce_smoke
python3 tools/ccld_monitor.py --once --all \
  /workspace/CCL-D/ccld-artifacts/multinode-logs/rank-*.jsonl
```

已验证两节点各一个 rank 的 `ncclAllReduce` 返回 `PASS`，结果值为 `3`，MPI 退出码为 `0`，completion 聚合状态为 `OK`。跨节点记录的单调时钟差异不会触发 `ENQUEUE_SKEW`；monitor 仅在 `nnodes=1` 时进行该项判断。

## 手动注入与检测测试

故障注入仅改动 CCL-D 诊断记录，不改实际 collective 参数、communicator 或 RCCL 执行流程。支持：

| 环境变量 | 效果 |
|---|---|
| `RCCL_CCLD_FAULT=op_mismatch` | 将指定 rank 的记录操作名改为测试值，供多 rank 对照检测 |
| `RCCL_CCLD_FAULT=size_mismatch` | 仅将诊断记录中的 count/bytes 加一个元素 |
| `RCCL_CCLD_FAULT=skip_record` | 丢弃指定 rank 的诊断事件，供多 rank 缺失检测 |
| `RCCL_CCLD_FAULT=missing_rank` | 仅将指定事件记录里的 `nranks` 增加 1，单卡也可验证超时告警 |
| `RCCL_CCLD_FAULT_RANK=N` | 选择注入 rank，默认 `0` |
| `RCCL_CCLD_FAULT_COUNTER=N` | 只注入该 communicator 的指定计数；不设置时注入该 rank 的每个事件 |

单节点真实 RCCL 冒烟和单卡可触发告警测试：

```bash
RCCL_CCLD_LIBDIR=/workspace/CCL-D/ccld-artifacts/dtk-26.04-gfx936-try1/install/lib \
  tools/tests/run_ccld_runtime_smoke.sh \
  /workspace/CCL-D/rccl-artifacts/dtk-26.04-gfx936/build/test/rccl-UnitTests
```

脚本先运行未注入的 `AllReduce.OutOfPlace` 并要求日志可被 monitor 判为 `OK`；当该 UT 暴露至少两个 rank 时，还会分别注入操作名与规模差异，要求 monitor 输出 `MISMATCH`。所有环境都会注入 `missing_rank` 并要求输出 `MISSING_RANK`，退出码为 `2`。聚合逻辑也由 `python3 -m unittest discover -s tools/tests -v` 中的双 rank 记录测试覆盖。测试注入是诊断链路自检，不代表 RCCL 内部真的发生通信故障。

## 与论文设计的覆盖差异

本阶段已增加第一版 kernel 级诊断：复用 RCCL 的 `ENABLE_COLLTRACE` device→host buffer，在每个 channel 的 kernel 生命周期内维护 `sendCount/recvCount`，并在 Simple、LL、LL128 primitive 完成一个操作后按 peer 数累加字节数；kernel end 由 host 诊断线程输出 `phase=complete`。monitor 会按 rank 汇总 completion，并报告 `COUNT_MISMATCH` 或 `MISSING_COMPLETION`。它可以验证“rank 是否完成 kernel，以及各 rank 的逻辑收发字节数是否一致”。

仍需继续完善的部分是：将计数按 trace slot 而不是按当前 channel kernel 汇总；为每轮操作保存稳定的 start/end 和算法/协议字段；diagnostic thread 计算相邻采样的 sendRate/recvRate；tree 按同层 rank 比较；以及对 graph、CollNet、NVLS、MSCCL 路径补充计数。由于当前记录还没有 peer/edge 拓扑，monitor 对算法为 ring 或 tree 且各 rank 计数不同的 completion 返回 `TOPOLOGY_PENDING`，避免把合法的 ring 边数差异误报为故障；只有未带算法元数据的合成记录才会直接触发 `COUNT_MISMATCH`。当前计数注入仍然是诊断侧自检，不会让 RCCL 真正产生错误通信；故障脚本可模拟 kernel 停滞、慢速和缺失，并由单元测试覆盖 analyzer 分支。

## 8 rank 跨节点实测

HPDPS-6 和 HPDPS-7 各启动 4 个 rank 的 AllReduce 已通过，8 个 rank 均返回 `PASS`，结果为 36，MPI 退出码为 0。启动时需要 `--mca plm_rsh_args "-p 2234"`，并导出 `/opt/hyhal/lib`、`/opt/dtk`、MPI 及独立 CCL-D 安装目录的库路径。

使用 `RCCL_CCLD_FAULT=op_mismatch`、`RCCL_CCLD_FAULT_RANK=7`、`RCCL_CCLD_FAULT_COUNTER=0` 后，远端 rank 7 的 JSONL 记录出现 `CCLD_INJECTED_OP_MISMATCH`，`ccld_monitor.py --fail-on-anomaly` 返回退出码 2 和 `MISMATCH`。注入只改诊断记录，通信数据面仍完成，适合验证 monitor 链路。

## UCX 升级

两节点当前均为 UCX 1.15.0，Open MPI 5.0.3 的 `libmpi.so` 从 `/opt/ucx` 加载 UCX，并提示组件要求 API >= 1.16。建议使用同一源码和配置在两节点并行安装到独立前缀 `/opt/ucx-1.16.x`，保留 `/opt/ucx` 回滚。仅通过 `LD_LIBRARY_PATH=/opt/ucx-1.16.x/lib:/opt/mpi/lib:$LD_LIBRARY_PATH` 验证 `ucx_info`、MPI 的 `ldd` 和 `mpirun --mca pml ucx`，确认通过后再切换默认环境；本次没有覆盖现有 UCX。
