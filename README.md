# CCL-D：RCCL 集合通信诊断扩展

CCL-D 将 RCCL 集合通信中的 **hang、slow、参数不一致和拓扑推进异常** 转成可追踪证据：`comm_id + counter/trace_id + rank + 状态 + 证据`。它适用于当前 DTK 26.04 / `gfx936` 环境，支持单节点和已验证的双节点 8 rank 运行。

## Requirements

当前版本按以下环境构建和验证。复用已编译 RCCL 时，目标服务器应尽量保持这些版本和路径一致。

| 项目 | 要求/已验证配置 |
|---|---|
| 容器工作目录 | `/workspace/CCL-D`，源码、独立 build、install 和实验产物均位于该目录下 |
| GPU/DCU 架构 | `gfx936`；CMake 使用 `-DGPU_TARGETS=gfx936` |
| DTK/HIP | `/opt/dtk-26.04`；`hipcc` 使用 dcc 25.10.0、Clang 17.0.0 |
| RCCL 安装 | 独立前缀 `/workspace/CCL-D/ccld-artifacts/sample-probe-install`，不覆盖系统 RCCL |
| MPI | Open MPI 5.0.3，入口 `/opt/mpi/bin/mpirun`、编译器 `/opt/mpi/bin/mpicxx` |
| UCX | `/opt/ucx-1.16.0`，运行时必须优先使用 `/opt/ucx-1.16.0/lib`；系统 `/opt/ucx` 保留作旧版本 |
| 操作系统/内核 | Linux x86_64；已验证节点为 HPDPS-6/7，内核 5.15.0-25-generic |
| 双节点布局 | HPDPS-6 (`10.18.91.6`) 与 HPDPS-7 (`10.18.91.7`)，SSH 端口 `2234`，每节点 4 rank，共 8 rank |
| 共享存储 | 两节点共享 `/workspace`，无需复制源码、RCCL 安装和测试程序 |

初始化环境时确保自定义 UCX 和 RCCL 在动态库搜索路径前面：

```bash
cd /workspace/CCL-D
source /opt/dtk-26.04/env.sh
export RCCL_PREFIX=$PWD/ccld-artifacts/sample-probe-install
export UCX_PREFIX=/opt/ucx-1.16.0
export LD_LIBRARY_PATH=$RCCL_PREFIX/lib:$UCX_PREFIX/lib:/opt/hyhal/lib:/opt/dtk/lib:/opt/dtk/hip/lib:/opt/mpi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

如果目标服务器同样是 `gfx936`，并且使用相同容器、DTK/HIP、RCCL ABI、驱动和通信栈，可以复用 `sample-probe-install`，不必重新编译 RCCL。首次使用仍应运行多算子 slot 回归，确认设备拓扑、UCX 和 MPI 配置可用。若 GPU 架构、DTK/ROCm、编译器 ABI 或运行时路径发生变化，应重新构建并安装。

## 工作闭环

```text
RCCL host enqueue
      │ 记录 op/count/bytes/datatype/root、rank、counter
      ▼
统一 Trace ID：<comm_hash>:<opCount>
      │
RCCL device channel 诊断计数
      │ 记录 algorithm/protocol、send/recv、start/end、Ring/Tree peer
      ▼
每 rank JSONL ──► ccld_analyze.py ──► ccld_monitor.py
                                      │
                                      └─ 状态、首错 rank、根因、证据
```

当前分析结果除了兼容的 `status` 字段，还提供：

- `root_cause`：`PARAMETER_MISMATCH`、`RANK_DID_NOT_ENQUEUE`、`COLLECTIVE_DID_NOT_COMPLETE`、`COMMUNICATION_PROGRESS_SLOW`、`TOPOLOGY_PROGRESS_ASYMMETRY` 等。
- `first_error_rank`：可以优先检查的 rank；无法确定时为 `null`。
- `confidence` 和 `evidence`：给出计数、速率、缺失 rank、拓扑等判定依据。

详细状态、完成度和后续缺口记录在 [docs/CCL-D_STATUS.md](docs/CCL-D_STATUS.md)，原理与改造说明见 [docs/CCL-D.md](docs/CCL-D.md)。

独立安装前缀不会覆盖系统 RCCL，后续源码修改也不会改变已生成的安装目录。

```bash
cd /workspace/CCL-D
source /opt/dtk-26.04/env.sh
cmake -S . -B ccld-artifacts/root-build-check \
  -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx936 -DBUILD_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX=$PWD/ccld-artifacts/root-install
cmake --build ccld-artifacts/root-build-check -j16
cmake --install ccld-artifacts/root-build-check
export CCLD_PREFIX=$PWD/ccld-artifacts/root-install
export LD_LIBRARY_PATH=$CCLD_PREFIX/lib:$LD_LIBRARY_PATH
```

## 开启记录并分析

```bash
mkdir -p ccld-artifacts/run
export RCCL_CCLD_ENABLE=1
export RCCL_CCLD_LOG=/workspace/CCL-D/ccld-artifacts/run/rank-{rank}.jsonl
# 运行中每 10ms 采集一次 send/recv 累计计数；可按需调整
export RCCL_CCLD_SAMPLE_MS=10

python3 tools/ccld_monitor.py --once --all \
  --slow-rate-ratio 0.5 \
  /workspace/CCL-D/ccld-artifacts/run/rank-*.jsonl
```

发现异常时让 monitor 返回退出码 2：

```bash
python3 tools/ccld_monitor.py --once --fail-on-anomaly \
  /workspace/CCL-D/ccld-artifacts/run/rank-*.jsonl
```

持续多轮偏慢检查：`--slow-window 5`。窗口模式只用于过滤单轮抖动，仍然是相对速率证据。
运行中采样停滞检查默认启用，连续 3 个采样间隔无进展且同节点对照 rank 仍有进展时输出 `COMM_STALLED`；可用 `--stagnant-intervals N` 调整间隔数。 completion 事件还包含 `send_count_by_protocol` 和 `recv_count_by_protocol`，数组顺序为 LL、LL128、Simple。

## 多算子 slot 回归

同一份测试程序验证 AllGather、ReduceScatter、Broadcast、AllToAll 与环形 P2P Send/Recv 的数据正确性，并在每轮操作后同步，便于将数据面失败定位到具体算子。RCCL 只需构建一次，测试程序单独编译：

```bash
tools/tests/build_multinode_collective_slot_regression.sh
CCLD_OUT_DIR=$PWD/ccld-artifacts/slot-regression-final \
CCLD_ITERS=20 CCLD_COUNT=4096 \
tools/tests/run_multinode_collective_slot_regression.sh
```

实验脚本保存每个 rank 的 JSONL、monitor 输出和运行日志。测试设计与 slot 生命周期经验见 [docs/CCL-D_TEST_PLAN.md](docs/CCL-D_TEST_PLAN.md)。

## 双节点验证

已验证节点为 `HPDPS-6 (10.18.91.6)` 与 `HPDPS-7 (10.18.91.7)`，SSH 端口 `2234`，每节点 4 rank，共 8 rank。共享 `/workspace`，不需要复制安装和日志。运行脚本：

```bash
tools/tests/run_multinode_ccld_validation.sh
tools/tests/run_multinode_fault_matrix.sh
```

正常实验记录在 `ccld-artifacts/validation-20iter2/`：8 rank、20 轮 AllReduce、count 4096，结果校验通过。

## 故障注入

注入只改变诊断记录，不破坏 RCCL 数据面：

```bash
export RCCL_CCLD_FAULT=op_mismatch       # 或 size_mismatch / skip_record / missing_rank
export RCCL_CCLD_FAULT_RANK=7
export RCCL_CCLD_FAULT_COUNTER=10        # 可选；不设置则作用于该 rank 的所有轮次
```

对应结果包括 `MISMATCH`、`MISSING_RANK`；合成 device 记录还可验证 `MISSING_COMPLETION`、`COMM_SLOW`。完整矩阵：

```bash
tools/tests/run_fault_simulation_matrix.sh
```

## 测试和产物

```bash
python3 -m unittest discover -s tools/tests -v
tools/tests/run_ccld_runtime_smoke.sh \
  /workspace/CCL-D/rccl-artifacts/dtk-26.04-gfx936/build/test/rccl-UnitTests
```

当前 Python 单元测试以实际运行结果为准（最近一次 34 项通过）。构建、安装和实验结果统一保存在 `ccld-artifacts/`；源码目录中的修改不会覆盖已有独立安装产物。
