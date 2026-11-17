# CCL-D Evaluation

本文按参考材料第 5 节 Evaluation 的验收口径记录当前 gfx936 双节点结果。材料中的功能验收关注 slow/hang 的发现、定位和分析；性能验收关注训练耗时、模型精度和资源开销。没有实测的数据不填入推断值。待补实现与场景矩阵见 [Evaluation 补充实施清单](CCL-D_EVALUATION_GAPS.md)。

## 1. 测试环境与方法

| 项目 | 当前测试 |
|---|---|
| 节点 | HPDPS-6、HPDPS-7；每节点 4 rank，共 8 rank |
| GPU/软件 | gfx936；DTK 26.04；独立 RCCL 安装前缀；UCX `/opt/ucx-1.16.0` |
| 诊断容量 | `RCCL_CCLD_ENABLE=1`、`RCCL_CCLD_SLOTS=512`、采样周期 20 ms |
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
| CPU/pinned memory/GPU 资源开销 | 部分已测 | RCCL 观测数据已验证容量和丢失策略；尚未采集训练级 CPU、pinned memory 峰值和 GPU kernel 开销 |
| 同一 group 多 work 并发 | 待测 | 当前压力程序每个算子后同步，需增加 grouped work 场景 |

## 4. 下一步验收

在多节点环境进行真实训练或等价的长时间 grouped-work 负载，至少保存 baseline 与开启 CCL-D 两组结果：总训练时间、step 平均耗时、吞吐、loss 曲线、诊断线程 CPU、pinned memory 峰值、GPU kernel/显存开销，以及 `OBSERVATION_DROPPED` 和 rank 完成比例。只有这些数据齐全后，才能对材料第 5.2 节的“影响在 1% 内”作出结论。

## 5. 非训练 grouped-work 与资源探针结果

测试程序 `tools/tests/multinode_grouped_work.cpp` 在 HPDPS-6/7 的 8 rank 上，在同一个 `ncclGroupStart/End` 中提交 AllReduce 与 P2P：

| 配置 | 通信校验 | 诊断结果 |
|---|---|---|
| 64 works × 2，512 slots | PASS，errors=0 | 每 rank 132 个 operation counter，dropped=0 |
| 128 works × 2，512 slots | PASS，errors=0 | 每 rank 260 个 operation counter，dropped=0 |
| 256 works × 2，512 slots | PASS，errors=0 | 产生 516 个 operation；部分 rank 各丢 4 条，原因是同时在用的记录超过 512 slots |
| 256 works × 2，1024 slots | PASS，errors=0 | 4,704 条 complete，16 条 resource，dropped=0，每 rank 516 个 counter |

512-slot 容量边界证明了资源保护语义：观测记录不足时显式标记丢失，通信继续完成；1024 slots 回归证明多 plan、混合 P2P 和连续 group 在容量足够时可以完成并回收。该测试不是训练性能结论。

程序同时输出 `/proc/self/status` 的 `VmPin` 前后读数，作为 pinned memory 交叉探针；容器和 UCX 初始化本身会改变 VmPin，因此必须结合 JSONL 中 `frame_pinned_bytes = 2144 × slots` 解读，不能把单次 VmPin 差值直接归因于 CCL-D。
