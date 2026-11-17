# CCL-D：集合通信诊断复现

本仓库按脱敏参考材料的图文设计复现集合通信诊断：RCCL 记录 TraceID、channel 收发计数及通信状态，诊断线程周期计算速率，kernel 完成事件通知线程输出操作指标，分析器按 Ring 全 rank、Tree 同层 rank 诊断 slow/hang。数据结构与诊断流程见 [复现规格](docs/CCL-D.md)。

## 环境

已验证环境：gfx936、DTK 26.04（dcc 25.10.0 / Clang 17）、Open MPI 5.0.3；双节点 HPDPS-6/7，每节点 4 rank，共享 `/workspace`。复用构建产物要求 GPU 目标、RCCL ABI、DTK/HIP 和通信运行环境兼容。

## 构建

```bash
source /opt/dtk-26.04/env.sh
cmake -S . -B rccl-artifacts/my-build \
  -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx936 -DBUILD_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX="$PWD/rccl-artifacts/my-install"
cmake --build rccl-artifacts/my-build -j8
cmake --install rccl-artifacts/my-build
```

Build directories and installed libraries should stay under `rccl-artifacts/` so generated objects and libraries remain separate from source changes. `RCCL_CCLD_SLOTS` sets the per-communicator frame capacity at runtime; the default is 16 and can be changed without recompiling.

Python 单测：`python3 -m unittest discover -s tools/tests -v`。实现评估见 [状态](docs/CCL-D_STATUS.md)，验收步骤见 [测试计划](docs/CCL-D_TEST_PLAN.md)。
