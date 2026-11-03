# CCL-D：集合通信诊断实现

CCL-D 是基于 RCCL 的集合通信诊断实现，目标是把训练中的 `hang` / `slow` 从一个模糊的任务现象，转换成可以按通信域、轮次和 rank 追踪的证据。

系统完成度、实验边界和后续工作记录在 [docs/CCL-D_STATUS.md](docs/CCL-D_STATUS.md)；原理与详细设计说明见 [docs/CCL-D.md](docs/CCL-D.md)。

## 1. 工作原理

一次集合通信需要通信域内所有 rank 共同参与。同一轮 collective 应满足：

1. 所有 rank 都进入同一轮操作。
2. 操作名、规模和关键参数一致。
3. kernel 最终完成，逻辑收发状态满足对应算法和拓扑约束。

CCL-D 在 Host 侧记录 collective enqueue，在 Device/Host 诊断通道记录 kernel 状态和收发计数，再按 `(comm_id, counter)` 聚合不同 rank 的记录。Trace ID 的格式为：

```text
<comm_id>:<counter>
```

Monitor 根据聚合结果输出状态：

| 状态 | 含义 |
|---|---|
| `OK` | 当前轮次的基础检查通过 |
| `MISMATCH` | 操作、规模、字节数、datatype 或 root 不一致 |
| `MISSING_RANK` | 预期 rank 没有产生 enqueue 记录 |
| `MISSING_COMPLETION` | 预期 rank 没有产生 kernel completion |
| `COUNT_MISMATCH` | 诊断记录中的逻辑 send/recv 计数不一致 |
| `COMM_SLOW` | 同一 collective 内某 rank 的 send/recv 推进速率低于基线比例 |
| `TOPOLOGY_PENDING` | 计数存在差异，但当前还缺少 peer/edge 拓扑，暂不下故障结论 |

`ENQUEUE_SKEW` 只表示同一节点内 Host enqueue 时间偏斜，不代表通信变慢。跨节点不会直接比较不同机器的绝对 monotonic timestamp。

## 2. 目录结构

```text
/workspace/CCL-D/
├── src/                         RCCL 源码和 CCL-D 插桩
├── tools/
│   ├── ccld_analyze.py          JSONL 聚合和判定逻辑
│   ├── ccld_monitor.py          持续/单次 monitor
│   └── tests/                   冒烟、故障矩阵和 Python 单元测试
├── docs/                        设计记录、状态和路线图
├── paper/                       CCL-D 论文和参考材料
├── ccld-artifacts/              独立构建、安装和实验产物
└── README.md                    当前入口文档
```

重要产物：

```text
ccld-artifacts/dtk-26.04-gfx936-try1/
├── build/                       gfx936 构建目录
├── install/                     独立 RCCL 安装目录
└── rccl-ccld-dtk-26.04-gfx936-install.tar.gz

ccld-artifacts/validation-20iter2/
└── 8 rank 双节点 20 轮正常运行记录

ccld-artifacts/fault-matrix-latest3/
└── normal、op_mismatch、size_mismatch、missing_rank、missing_record

ccld-artifacts/fault-simulation-latest/
└── stall、slow、missing completion 合成诊断记录
```

## 3. 编译和安装

编译环境是 DTK 26.04，目标 GPU 为 `gfx936`。构建产物安装在独立前缀，不覆盖系统 RCCL，也不会被后续源码修改覆盖。

```bash
cd /workspace/CCL-D
source /opt/dtk-26.04/env.sh

cmake -S . \
  -B ccld-artifacts/dtk-26.04-gfx936-try1/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_TARGETS=gfx936 \
  -DBUILD_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX=$PWD/ccld-artifacts/dtk-26.04-gfx936-try1/install

cmake --build ccld-artifacts/dtk-26.04-gfx936-try1/build -j16
cmake --install ccld-artifacts/dtk-26.04-gfx936-try1/build
```

使用独立安装：

```bash
export CCLD_PREFIX=/workspace/CCL-D/ccld-artifacts/dtk-26.04-gfx936-try1/install
export LD_LIBRARY_PATH=$CCLD_PREFIX/lib:$LD_LIBRARY_PATH
```

## 4. 开启诊断记录

诊断默认关闭。开启后，每个 rank 写自己的 JSONL 文件：

```bash
export RCCL_CCLD_ENABLE=1
export RCCL_CCLD_LOG=/workspace/CCL-D/ccld-artifacts/run/rank-{rank}.jsonl
mkdir -p /workspace/CCL-D/ccld-artifacts/run
```

`RCCL_CCLD_LOG` 支持 `{rank}`、`${RANK}` 和 `%r` 占位符。没有占位符时，程序会自动追加 `.rank<N>`，避免多个 rank 同时写同一个文件。

## 5. Monitor 使用

单次分析：

```bash
python3 tools/ccld_monitor.py --once --all \
  /workspace/CCL-D/ccld-artifacts/run/rank-*.jsonl
```

带 kernel 速率离群检查（默认低于中位数 50% 判定为 `COMM_SLOW`）：

```bash
python3 tools/ccld_monitor.py --once --all --slow-rate-ratio 0.5 \
  /workspace/CCL-D/ccld-artifacts/run/rank-*.jsonl
```

连续多轮检查（例如连续 5 轮中至少 2 轮偏慢）可启用窗口模式：

```bash
python3 tools/ccld_monitor.py --once --all --slow-window 5 \
  /workspace/CCL-D/ccld-artifacts/run/rank-*.jsonl
```

窗口模式输出 `COMM_SLOW_WINDOW`，用于过滤单轮抖动；它仍然只比较同一 collective 内各 rank 的设备时钟速率。

当 completion 记录包含 `send_rank`、`recv_rank`、`channel_base` 时，分析器还会输出 `topology_groups`，提供每条 peer/channel 边的收发量、持续时间和速率；`topology_ready=true` 表示边字段齐全，可继续进行 Ring/Tree 分层分析。

拓扑判定结果：`RING_SYMMETRIC` 表示每个 channel 的环边互为对端且边速率没有明显失衡；`RING_ASYMMETRY` 表示发现缺边、重复边或边速率至少相差 2 倍。Tree 会从 RCCL channel 的 `tree_up/tree_down/tree_depth` 字段构造父子边，并按显式 depth 分组比较；缺少这些字段时返回 `TREE_LEVEL_PENDING`，同层明显失衡时返回 `TREE_ASYMMETRY`。

检测到异常时返回退出码 `2`：

```bash
python3 tools/ccld_monitor.py --once --fail-on-anomaly \
  /workspace/CCL-D/ccld-artifacts/run/rank-*.jsonl
echo $?
```

持续轮询：

```bash
python3 tools/ccld_monitor.py \
  --poll-ms 250 \
  --timeout-ms 2000 \
  /workspace/CCL-D/ccld-artifacts/run/rank-*.jsonl
```

## 6. 双节点 8 rank 验证

两台节点：

```text
HPDPS-6：10.18.91.6
HPDPS-7：10.18.91.7
SSH：2234
每节点 4 rank，总计 8 rank
```

共享存储已经用于日志和产物，不需要在节点之间复制文件。运行时需要使用相同的 RCCL、MPI、UCX 和 CCL-D 环境。典型启动方式：

```bash
export LD_LIBRARY_PATH=/workspace/CCL-D/ccld-artifacts/dtk-26.04-gfx936-try1/install/lib:$LD_LIBRARY_PATH
export RCCL_CCLD_ENABLE=1
export RCCL_CCLD_LOG=/workspace/CCL-D/ccld-artifacts/multinode-run/rank-{rank}.jsonl

/opt/mpi/bin/mpirun --allow-run-as-root \
  --host HPDPS-6,HPDPS-7 \
  -np 8 \
  --map-by ppr:4:node \
  --mca plm_rsh_args '-p 2234' \
  -x LD_LIBRARY_PATH \
  -x RCCL_CCLD_ENABLE \
  -x RCCL_CCLD_LOG \
  <RCCL-8-rank-test>
```

已验证的正常实验：8 rank、2 节点、20 轮、4096 elements/轮，AllReduce 结果为 `36`，所有 rank 返回 `PASS`。

## 7. 故障注入

这些注入用于验证诊断链路和 monitor，不改变 RCCL 实际通信数据面。

```bash
# 指定 rank 7 在指定 counter 注入操作名不一致
export RCCL_CCLD_FAULT=op_mismatch
export RCCL_CCLD_FAULT_RANK=7
export RCCL_CCLD_FAULT_COUNTER=10

# 规模/字节数不一致
export RCCL_CCLD_FAULT=size_mismatch

# 丢弃指定 rank 的诊断记录
export RCCL_CCLD_FAULT=skip_record

# 增大事件中的预期 rank 数，用于验证超时缺失检测
export RCCL_CCLD_FAULT=missing_rank
```

已经验证：

| 注入 | Monitor 结果 |
|---|---|
| operation mismatch | `MISMATCH`，退出码 2 |
| size/count mismatch | `MISMATCH`，退出码 2 |
| skip record | `MISSING_RANK`，退出码 2 |
| missing rank | `MISSING_RANK`，退出码 2 |
| 合成 stall | `MISSING_COMPLETION`（stall rank） |
| 合成 slow | `COMM_SLOW`，输出 `slow_ranks=[2]` |
| 合成 completion 缺失 | `MISSING_COMPLETION` |

## 8. 测试

Python analyzer 和合成故障测试：

```bash
python3 -m unittest discover -s tools/tests -v
```

当前结果：19 项通过。

故障矩阵脚本：

```bash
tools/tests/run_fault_simulation_matrix.sh

tools/tests/run_multinode_fault_matrix.sh
```

双节点运行验证：

```bash
tools/tests/run_multinode_ccld_validation.sh
```

RCCL runtime smoke：

```bash
tools/tests/run_ccld_runtime_smoke.sh \
  /workspace/CCL-D/rccl-artifacts/dtk-26.04-gfx936/build/test/rccl-UnitTests
```

当前所有构建、测试和实验产物都保存在 `/workspace/CCL-D/ccld-artifacts/`，源码修改不会覆盖独立安装结果。
