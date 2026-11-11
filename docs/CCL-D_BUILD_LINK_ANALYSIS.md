# gfx936 构建与链接耗时核对

## 已确认事实

最近一次完整 `gfx936` 构建记录在 `ccld-artifacts/dtk-26.04-gfx936-try1/build.log`：

- 总耗时：`1434.49 s`，重复构建记录为 `1445.35 s`；
- RCCL 目标包含约 `193` 个对象文件；
- 其中自动生成的 `hipify/gensrc/*.cpp.o` 约占 `48 MiB`；
- 使用 `/opt/dtk-26.04/bin/hipcc`、`gfx936`、`-O3`、`-fgpu-rdc`；
- 编译参数包含 `-parallel-jobs=12`；
- device link 参数使用 `-parallel-jobs=16`；
- 最终通过一个 HIP shared-library device link 生成 `librccl.so.1.0`。

## 结论

“链接很慢”不能简单归因于最后一步 `ld.lld`。当前时间主要由三部分组成：

1. 大量自动生成的 collective/reduction kernel 变体编译；
2. `-fgpu-rdc` 使每个 HIP 对象保留可重定位 device code，并要求最终统一 device link；
3. 修改 `src/device/common.h` 等公共 device 头文件时，约 193 个对象会重新进入 HIP 编译，再执行一次整体 device link。

当前容器有较多 CPU 和内存，但工程只给 hipcc 编译设置了 12 个内部并行 job，链接设置为 16 个 job。提高这两个数可能缩短时间，但不能消除 193 个对象和 `-fgpu-rdc` 的固定工作量；过度提高并行度还可能造成内存压力。

## 本轮不采用的做法

- 不关闭 `-fgpu-rdc`：RCCL 当前 device 代码依赖该能力，不能为了缩短链接时间破坏正确性。
- 不裁剪生成的 kernel 变体：这会改变支持范围，不能作为诊断重写的默认构建优化。
- 不覆盖现有安装目录：诊断核心重写期间继续使用独立 build/install 前缀。

## 可执行优化

1. 诊断核心先在 host 侧和合成状态机完成，避免每个设计迭代都触发 HIP 全量重编译。
2. 所有 device work/body/计数修改合并成一次变更后再编译。
3. 每轮使用独立 build 目录，并保留上一版安装目录用于回归对照。
4. 仅在确认可接受内存余量后，把 `-parallel-jobs=12` 和 device-link `16` 做成一次对照实验；这属于构建参数实验，不改变诊断语义。
5. 记录“首次编译”和“只改 host C++”两种耗时，不能把 host 增量结果外推到 device 头文件修改。

## 开发构建配置

现在可以显式控制 HIP 编译和 device-link 并行度，也可以限制自动生成的 device function 集合。完整验收仍使用默认全量函数；下面的配置只用于调试诊断核心和 host/work 元数据：

```bash
source /opt/dtk-26.04/env.sh
cmake -S . -B ccld-artifacts/dev-gfx936/build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGPU_TARGETS=gfx936 \
  -DBUILD_LOCAL_GPU_TARGET_ONLY=ON \
  -DONLY_FUNCS='AllReduce RING LL/SIMPLE Sum f32|AllGather RING LL/SIMPLE Sum f32|ReduceScatter RING LL/SIMPLE Sum f32|Broadcast RING LL/SIMPLE Sum f32|AllToAllPivot RING SIMPLE Sum i8|SendRecv RING SIMPLE Sum i8' \
  -DRCCL_HIP_PARALLEL_JOBS=24 \
  -DRCCL_DEVICE_LINK_JOBS=16 \
  -DCMAKE_INSTALL_PREFIX=$PWD/ccld-artifacts/dev-gfx936/install
cmake --build ccld-artifacts/dev-gfx936/build --parallel 24
```

`ONLY_FUNCS` 会改变生成的 kernel 覆盖范围，开发构建不能用于最终能力声明或完整回归。修改 `src/device/common.h` 时仍会触发选定函数的 HIP 重编译；只修改 host C++ 时，现有 build 目录可以保留大部分对象。

建议固定两套构建目录：

- `dev-gfx936`：窄函数集、`RelWithDebInfo`，用于快速迭代；
- `full-gfx936`：全函数集、Release，用于最终安装和验收。
