# CCL-D Evaluation

本文按参考材料第 5 节 Evaluation 的验收口径记录当前 gfx936 双节点结果。材料中的功能验收关注 slow/hang 的发现、定位和分析；性能验收关注训练耗时、模型精度和资源开销。没有实测的数据不填入推断值。待补实现与场景矩阵见 [Evaluation 补充实施清单](CCL-D_EVALUATION_GAPS.md)。

## 验收结论（逐项对照）

“有代码”或“通信通过”不等于达到 AntCCL 验收标准。按材料要求逐项判断：

| AntCCL 验收项 | 当前判定 | 判定依据 |
|---|---|---|
| rank 未进入 collective | 未通过 | 有近似 missing 注入，但没有按材料口径完成发现时间、首错 rank、根因和原始证据闭环 |
| kernel 内异常 hang | 未通过 | 现有 stall 在通信开始前阻塞，不是 kernel 内 send/recv 阶段异常 |
| collective count 不一致 | 未通过 | 尚无 rank 少 1024 B 的多节点注入和定位结果 |
| collective 后非通信 hang | 未通过 | 尚无通信完成后单 rank host 阻塞场景 |
| 计算侧 slow | 未通过 | 尚无 collective 调用前 rank sleep 3 s 的分析结果 |
| 通信侧 slow | 未通过 | 尚无每次 send/recv 后逐步延迟 2 us 的设备侧注入结果 |
| 训练耗时、吞吐、loss、精度 ≤1% | 未通过 | 按用户要求尚未启动训练验收 |
| 诊断线程 CPU 约 1% | 未通过 | 当前是共享 RCCL 服务线程；只有采样区 CPU 记录，没有独立 diagnose thread 的 AntCCL 结构和正式对照 |
| pinned memory 资源验收 | 部分完成 | frame 大小和 slots 可记录，VmPin 已有探针，但还没有按 slots 矩阵完成基线/峰值报告 |
| GPU 无额外开销 | 未通过 | HIP event 已显示 off/on 有差异，尚未拆分纯 kernel、callback 和后台线程开销 |
| 同一 group 多 work 生命周期 | 已通过核心回归 | 256 work × 2、512 slots、8 rank，通信正确、516 counter/rank、dropped=0 |

因此，当前不能说“已经达到 AntCCL 标准”：功能六项是 **0/6 完整通过**；grouped-work 生命周期是已通过的工程回归项；资源和性能项仍未完成正式验收。

## 1. 测试环境与方法

| 项目 | 当前测试 |
|---|---|
| 节点 | HPDPS-6、HPDPS-7；每节点 4 rank，共 8 rank |
| GPU/软件 | gfx936；DTK 26.04；独立 RCCL 安装前缀；UCX `/opt/ucx-1.16.0` |
| 诊断容量 | `RCCL_CCLD_ENABLE=1`、`RCCL_CCLD_SLOTS=512`、`RCCL_CCLD_SAMPLE_US=1000`（采样周期可运行时调整） |
| 正常回归 | AllGather、ReduceScatter、Broadcast、AllToAll、P2P；`CCLD_ITERS=100`、`CCLD_COUNT=4096` |
| 故障回归 | rank 2 注入 slow（stream 延迟 2 s）、missing（不进入 collective）、stall（stream 等待 30 s） |
| 原始证据 | 每 rank RCCL JSONL、monitor 输出、设备结果校验；所有运行目录保存在 `rccl-artifacts/ccld-frame-gfx936/` |

## 2. 功能验收

### 2.1 正常通信与观测完整性

压力回归目录：`rccl-artifacts/ccld-frame-gfx936/multinode-stress-slots512-20261117T055324Z/`。

| 指标 | 结果 | 判定 |
|---|---:|---|
| rank 数 | 8 | 通过 |
| 每 rank channel complete | 3,500 | 通过 |
| complete 总数 | 28,000 | 通过 |
| sample 总数 | 4,858 | 通过 |
| `OBSERVATION_DROPPED` | 0 | 通过 |
| rank 数据校验 | 全部通过 | 通过 |
| monitor 结果 | 2,900 条，全部 `OK` | 通过 |

这轮测试按每个算子完成后同步，证明 100 轮长迭代下通信、TraceID、channel 计数、完成回收和 monitor 聚合稳定；它没有覆盖同一 group 内多个 work 并发 enqueue。

### 2.2 故障发现、定位与分析

故障矩阵目录：`rccl-artifacts/ccld-frame-gfx936/multinode-faults-current-20261117T055417Z/`。

| 场景 | 注入方式 | monitor 结果 | 通信/进程结果 |
|---|---|---|---|
| slow | rank 2 的 collective 前插入 2 s stream 延迟 | `COMM_STALLED`；`root_cause=RANK_PROGRESS_STALLED`；`slow_ranks=[2]`；progressing peers 为 `[0,1,3]` | 8 rank 最终完成，数据校验通过 |
| missing | rank 2 不进入首个 collective | `MISSING_RANK`，`missing_ranks=[2]`；随后报告 collective 未完成 | 按注入预期以退出码 97 终止 |
| stall/hang | rank 2 的 stream 等待 30 s 后才允许 collective 继续 | `COMM_STALLED`；`root_cause=RANK_PROGRESS_STALLED`；`slow_ranks=[2]` | 外层 8 s 超时退出码 124 |

当前 monitor 轮询周期为 50 ms；slow 记录中的停滞窗口约 61 ms（3 个采样间隔），因此达到了“秒级发现、rank 级定位”的功能目标。材料中的“5 s 发现、1 min 定位”是上限式验收口径，本测试没有需要等待到该上限。

## 3. 与材料第 5 节的对照

| 材料验收项 | 当前状态 | 证据或缺口 |
|---|---|---|
| slow/hang 发现 | 已完成 | 双节点 live 注入；monitor 实时输出异常状态 |
| slow/hang 精确定位 | 已完成 | 三类故障均定位到 rank 2，并给出 root cause |
| 故障分析流程 | 已完成 | missing、progress stalled、collective incomplete 分支均有输出 |
| 训练耗时影响 ≤1% | 未测 | 当前是通信回归程序，没有 LLM 训练 baseline/enable 对照 |
| 模型精度影响 ≤1% | 未测 | 尚未接入真实训练任务和 loss 曲线对照 |
| CPU/pinned memory/GPU 资源开销 | 部分已测 | 采样区 CPU、共享线程 CPU、frame 字节和 HIP event 探针已工作；短消息微基准显示端到端有开销，训练级开销与纯 kernel 开销仍未测 |
| 同一 group 多 work 并发 | 已完成核心路径回归 | 最终 callback 快照版本下 256 works × 2、512 slots、8 rank 全部完成，dropped=0；覆盖 AllReduce 与混合 P2P |

## 4. 下一步验收

在多节点环境进行真实训练或等价的长时间 grouped-work 负载，至少保存 baseline 与开启 CCL-D 两组结果：总训练时间、step 平均耗时、吞吐、loss 曲线、诊断线程 CPU、pinned memory 峰值、GPU kernel/显存开销，以及 `OBSERVATION_DROPPED` 和 rank 完成比例。只有这些数据齐全后，才能对材料第 5.2 节的“影响在 1% 内”作出结论。

## 5. 非训练 grouped-work 与资源探针结果

测试程序 `tools/tests/multinode_grouped_work.cpp` 在 HPDPS-6/7 的 8 rank 上，在同一个 `ncclGroupStart/End` 中提交 AllReduce 与 P2P：

| 配置 | 通信校验 | 诊断结果 |
|---|---|---|
| 64 works × 2，512 slots | PASS，errors=0 | 每 rank 132 个 operation counter，dropped=0 |
| 128 works × 2，512 slots | PASS，errors=0 | 每 rank 260 个 operation counter，dropped=0 |
| 256 works × 2，512 slots（最终 callback 快照版本） | PASS，errors=0 | 4,704 条 complete、16 条 resource、每 rank 516 个 counter，dropped=0 |
| 256 works × 2，1024 slots | PASS，errors=0 | 4,704 条 complete，16 条 resource，dropped=0，每 rank 516 个 counter |

早期实现中 callback 只排队完成事件，后台轮询前 slot 未释放，导致第二轮出现假性容量耗尽；最终实现由 callback 快照并释放 slot 后，这个 512-slot 回归恢复为无丢失。该测试不是训练性能结论。

程序同时输出 `/proc/self/status` 的 `VmPin` 前后读数，作为 pinned memory 交叉探针；容器和 UCX 初始化本身会改变 VmPin，因此必须结合 JSONL 中 `frame_pinned_bytes = 2144 × slots` 解读，不能把单次 VmPin 差值直接归因于 CCL-D。

## 6. 设备时间微基准

HIP event 微基准最终版本的 off/on 配对结果（9 次非 warmup 样本）为：4096 元素时均值 1.180651/2.144755 ms；65536 元素时 7.368763/8.821697 ms。两组通信校验均通过。诊断开销随消息变大而相对下降，但仍不能宣称 GPU 零开销；还需用 profiler 拆分 kernel 计数、host callback 和后台线程部分。

该计时覆盖 grouped operation 的设备 stream，不等同于纯 GPU kernel 插桩开销，也不能外推到训练任务。后续应使用更多重复和 profiler 拆分 kernel 与 host callback 时间。
