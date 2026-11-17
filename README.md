# CCL-D

CCL-D 是 RCCL 集合通信诊断实现。它在通信库中记录集合操作的统一 TraceID、kernel 元数据、channel 收发计数和采样增量，由独立 diagnose thread 读取 pinned memory，再由 host analyzer 按 Ring/Tree 拓扑和 slow/hang 决策树给出 rank 级证据。

## 当前设计

- TraceID：`commId(8B) + counter(8B) + reserved(44B)`；counter 使用通信域 RCCL `opCount`，grouped P2P 调用共享父 counter。
- Header：`kernelIndex`、`mode`、`counter`、`detectChannel`、`traceId`。
- Body：每个 slot、每个 channel 只有 `sendCount` 和 `recvCount`。
- mode：`0` 关闭；`1` 仅 tid 0 计数；`2` 所有 tid 计数。
- 生命周期：kernel callback 在所有 plan 完成后回收 slot。
- 资源策略：slot 不足时记录 `OBSERVATION_DROPPED`，通信继续。
- 采样：diagnose thread 输出采样周期内 `send_rate`、`recv_rate` 增量。
- 分析：Ring 比较同 channel 全 rank；Tree 比较同 channel、同层级、同父子角色 rank。

## 环境要求

- gfx936 DCU、DTK 26.04、HIP 6.3、Clang 17
- Open MPI 5.0.3、UCX 1.16
- 多节点需要共享文件系统、hostname 解析和 SSH 互通

## 运行参数

```bash
export RCCL_CCLD_ENABLE=1
export RCCL_CCLD_MODE=1
export RCCL_CCLD_SLOTS=512
export RCCL_CCLD_SAMPLE_US=1000
export RCCL_CCLD_PLAN_COUNTER_INITIAL=8
export RCCL_CCLD_LOG=/path/to/results/rank-{rank}.jsonl
```

修改 CCL-D 运行参数不需要重新编译。

## 构建

```bash
source /opt/dtk-26.04/env.sh
cmake -S . -B rccl-artifacts/ccld-frame-gfx936/build \
  -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx936 -DBUILD_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX="$PWD/rccl-artifacts/ccld-frame-gfx936/install"
cmake --build rccl-artifacts/ccld-frame-gfx936/build -j8
cmake --install rccl-artifacts/ccld-frame-gfx936/build
```

当前源码已完成静态检查，新的 gfx936 编译尚未执行。

## 分析器

```bash
python3 tools/ccld_monitor.py --once --all results/rank-*.jsonl
```

主要状态包括 `COUNT_MISMATCH`、`COMM_SLOW`、`COMPUTATION_SLOW`、`MIXED_SLOW`、`HANG_MISSING_RANK`、`HANG_PARTIAL_COMPLETION`、`HANG_ALL_RANKS` 和 `NON_COMMUNICATION_HANG_UNLOCALIZED`。

## 静态检查

```bash
python3 -m unittest discover -s tools/tests -q
```

设计细节见 [docs/CCL-D.md](docs/CCL-D.md)，验证场景见 [docs/CCL-D_TEST_PLAN.md](docs/CCL-D_TEST_PLAN.md)。
