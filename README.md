# CCL-D：集合通信诊断复现

本仓库围绕 脱敏参考材料中明确的诊断链路开展复现：RCCL 在通信期间记录操作和 channel 级 send/recv 证据，诊断线程计算通信速率，kernel 完成后形成操作级记录，再按 Ring 跨 rank、Tree 同层 rank 比较。材料未公开完整 slow/hang 决策树，因此代码中的具体判定规则仍需通过实验单独验证。

## 环境要求

已验证环境为 gfx936、DTK 26.04（dcc 25.10.0 / Clang 17）、Open MPI 5.0.3；双节点验证使用 HPDPS-6/7、每节点 4 rank、共享 `/workspace`。复用 RCCL 构建产物要求设备目标、RCCL ABI、DTK/HIP 运行时及通信环境兼容；更换其中关键项时应重新构建并记录实际加载的库。

## 构建和运行

```bash
source /opt/dtk-26.04/env.sh
cmake -S . -B ccld-artifacts/my-build \
  -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx936 -DBUILD_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX="$PWD/ccld-artifacts/my-install"
cmake --build ccld-artifacts/my-build -j16
cmake --install ccld-artifacts/my-build
```

双节点回归脚本及环境变量见 `tools/tests/`。Python 分析器单测：

```bash
python3 -m unittest discover -s tools/tests -v
```

当前代码和证据边界见 [项目评估](docs/CCL-D_STATUS.md)；材料与实现的区别见 [复现依据](docs/CCL-D.md)，验证步骤见 [测试计划](docs/CCL-D_TEST_PLAN.md)。
