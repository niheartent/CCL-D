# CCL-D

CCL-D 是面向 RCCL 集合通信的高精度诊断实现。它在通信库关键路径记录一次操作的 TraceID、算法/协议、channel 拓扑和收发计数，由 host 侧诊断线程持续采样，再由分析器聚合各 rank 的证据，判断通信是否正常、停滞或未完成，并给出疑似 rank 和根因。

## 功能

- **RCCL 插桩**：记录 enqueue、kernel 完成、TraceID、算法/协议、操作规模及每个 channel 的 `sendCount/recvCount`。
- **生命周期管理**：pinned frame、运行时 slot 容量、完成回收和通信继续策略；容量不足时输出 `OBSERVATION_DROPPED`，不阻断通信。
- **拓扑分析**：Ring 邻居对称性比较；Tree 从 `tree_up/tree_down*` 重建层级并进行同层比较。
- **故障分析**：识别 rank 未进入 collective、collective 未完成、rank 进度停滞和通信速率异常，输出 `root_cause` 与受影响 rank。
- **多节点验证工具**：提供五类 collective 回归、slot 压力测试、slow/missing/stall 故障注入和 monitor 实时分析脚本。

实现规格见 [CCL-D 设计说明](docs/CCL-D.md)，验收结果见 [Evaluation](docs/CCL-D_EVALUATION.md)，详细状态见 [CCL-D_STATUS](docs/CCL-D_STATUS.md)。

## 环境要求

已验证配置：

- gfx936 DCU；DTK 26.04（HIP 6.3.26113，Clang 17）
- Open MPI 5.0.3
- UCX 1.16（默认路径 `/opt/ucx-1.16.0`）
- Linux 容器，源码、构建目录和安装目录位于同一共享文件系统
- 多节点运行需要节点间 SSH 互通，并确保 hostname 能解析到正确地址

复用已安装 RCCL 时，GPU 目标、RCCL ABI、DTK/HIP、MPI 和 UCX 必须兼容。当前 gfx936 构建不提供 LL128 kernel；双节点 Ring+LL、Tree+Simple 等受支持组合可正常运行。

## 构建和安装 RCCL

在仓库根目录执行：

```bash
source /opt/dtk-26.04/env.sh

cmake -S . -B rccl-artifacts/ccld-frame-gfx936/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_TARGETS=gfx936 \
  -DBUILD_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX="$PWD/rccl-artifacts/ccld-frame-gfx936/install"

cmake --build rccl-artifacts/ccld-frame-gfx936/build -j8
cmake --install rccl-artifacts/ccld-frame-gfx936/build
```

构建对象和安装库放在 `rccl-artifacts/` 下，与源码修改隔离。`RCCL_CCLD_SLOTS` 是运行时参数，改变容量不需要重新编译：

```bash
export RCCL_CCLD_ENABLE=1
export RCCL_CCLD_SLOTS=512
export RCCL_CCLD_SAMPLE_US=1000
# Optional: initial host capacity for one plan's counter list (it grows as needed).
export RCCL_CCLD_PLAN_COUNTER_INITIAL=8
export RCCL_CCLD_LOG=/path/to/results/rank-{rank}.jsonl
```

## 运行分析器

对已生成的 rank JSONL 做一次分析：

```bash
python3 tools/ccld_monitor.py --once --all results/rank-*.jsonl
```

持续读取正在追加的日志：

```bash
python3 tools/ccld_monitor.py --poll-ms 50 --timeout-ms 0 --all \
  results/rank-*.jsonl
```

`status` 表示检测状态，`root_cause` 表示分析器判定的根因，`slow_ranks`、`missing_ranks` 和 `incomplete_ranks` 列出受影响 rank。

## 双节点测试

默认使用 HPDPS-6/7，每节点 4 rank，共 8 rank。设置本地 RCCL、UCX 路径和测试规模：

```bash
source /opt/dtk-26.04/env.sh
export RCCL_PREFIX="$PWD/rccl-artifacts/ccld-frame-gfx936/install"
export UCX_PREFIX=/opt/ucx-1.16.0
export CCLD_ITERS=100
export CCLD_COUNT=4096
export RCCL_CCLD_SLOTS=512

tools/tests/run_multinode_collective_slot_regression.sh
```

运行受控故障注入：

```bash
tools/tests/run_multinode_ccld_faults.sh
```

节点和 rank 映射可通过环境变量调整：

```bash
export CCLD_HOSTS='HPDPS-6:4,HPDPS-7:4'
export CCLD_NP=8
```

脚本会在 `rccl-artifacts/` 下创建带时间戳的结果目录，保存运行日志、每 rank JSONL 和 monitor 输出。

## 单元测试

```bash
python3 -m unittest discover -s tools/tests -q
```

测试设计见 [测试计划](docs/CCL-D_TEST_PLAN.md)，当前实验记录见 [Evaluation](docs/CCL-D_EVALUATION.md)。

## 目录结构

| 路径 | 内容 |
|---|---|
| `src/` | RCCL 通信库及 CCL-D 插桩实现 |
| `tools/ccld_analyze.py` | host 侧聚合与决策分析器 |
| `tools/ccld_monitor.py` | JSONL 实时/离线 monitor |
| `tools/tests/` | 单测、双节点回归和故障注入工具 |
| `docs/` | 设计说明、测试计划、状态和评估结果 |
| `rccl-artifacts/` | 本地构建、安装和实验产物 |
