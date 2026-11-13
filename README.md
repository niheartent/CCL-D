# CCL-D：集合通信诊断扩展

本项目以 AntCCL 的诊断系统为复现目标，项目名称统一为 CCL-D。目标链路是：RCCL 为一次用户操作采集跨 rank、channel、work 的通信证据，monitor 按操作身份聚合并判断参数异常、slow、hang 和拓扑问题。

当前已有 gfx936 RCCL 插桩原型、五类算子双节点数据面回归和分析器测试。跨 plan 操作归属、完成通知与最终回收、Simple/LL/LL128 精确计数尚未完成；详细状态只维护在 [状态文档](docs/CCL-D_STATUS.md)，目标模型见 [复现说明](docs/CCL-D.md)。

## 已验证环境

| 项目 | 配置 |
|---|---|
| GPU/DCU | gfx936 |
| DTK/HIP | `/opt/dtk-26.04`，dcc 25.10.0，Clang 17.0.0 |
| RCCL 安装 | `ccld-artifacts/sample-probe-install`，独立于系统库 |
| MPI | Open MPI 5.0.3，`/opt/mpi` |
| 双节点 | HPDPS-6/7，10.18.91.6/7，SSH 2234，每节点 4 rank |
| 共享存储 | 两节点共享 `/workspace` |

同一 gfx936、容器、DTK/HIP、RCCL ABI 和运行时环境可复用构建产物；环境或 ABI 不同则重新构建。双节点运行时以实际加载的 UCX/MPI 库为准。

```bash
cd /workspace/CCL-D
source /opt/dtk-26.04/env.sh
export RCCL_PREFIX=$PWD/ccld-artifacts/sample-probe-install
export LD_LIBRARY_PATH=$RCCL_PREFIX/lib:/opt/hyhal/lib:/opt/dtk/lib:/opt/dtk/hip/lib:/opt/mpi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

## 构建独立 RCCL

```bash
cmake -S . -B ccld-artifacts/my-build \
  -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx936 -DBUILD_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX=$PWD/ccld-artifacts/my-install
cmake --build ccld-artifacts/my-build -j16
cmake --install ccld-artifacts/my-build
```

## 运行双节点多算子回归

覆盖 AllGather、ReduceScatter、Broadcast、AllToAll 和 P2P；每轮校验通信数据并保存 rank 日志。

```bash
tools/tests/build_multinode_collective_slot_regression.sh
CCLD_OUT_DIR=$PWD/ccld-artifacts/slot-regression \
CCLD_ITERS=20 CCLD_COUNT=4096 \
tools/tests/run_multinode_collective_slot_regression.sh
```

双节点验证和故障矩阵：

```bash
tools/tests/run_multinode_ccld_validation.sh
tools/tests/run_multinode_fault_matrix.sh
```

故障注入只修改诊断证据，不模拟真实 GPU、驱动或网络故障。Python analyzer/monitor 测试：

```bash
python3 -m unittest discover -s tools/tests -v
```

运行日志、独立构建和安装产物存放在 `ccld-artifacts/`。核心剩余工作和验收条件见 [状态文档](docs/CCL-D_STATUS.md)、[路线图](docs/ROADMAP.md) 与 [测试计划](docs/CCL-D_TEST_PLAN.md)。
