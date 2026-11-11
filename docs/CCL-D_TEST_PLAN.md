# CCL-D 实验设计与验收方案

本文定义 CCL-D 完成开发后的实验顺序、故障场景、观测指标和验收标准。当前只固化方案，待新版通信库和核心功能完成后再执行真实实验。每次实验保存 Git commit、RCCL 安装前缀、节点和 rank 映射、环境变量、原始 JSONL、monitor 输出和退出码。

## 1. 测试目标

实验需要回答四个问题：是否能关联 host enqueue、kernel completion、send/recv、算法、协议、rank、channel 和拓扑；是否能发现参数不一致、rank 缺失、kernel 未完成、速率异常和持续停滞；是否能定位异常 rank、counter、channel、peer 和协议；开启诊断后训练吞吐和通信时延增加多少。

测试分三层：单元测试与合成日志 -> RCCL 双节点多算子压力 -> 真实训练任务与可控故障。训练任务用于验证实际价值，不能替代确定性的通信库测试。

## 2. 基线环境

- 节点：HPDPS-6、HPDPS-7，SSH 端口 2234
- 共享目录：/workspace/CCL-D，GPU 目标：gfx936
- 默认规模：每节点 4 rank，共 8 rank
- 默认通信：RCCL，MPI 负责启动和 unique ID 广播

每次实验保存 collective 类型、数据量、轮数、rank 布局、RCCL/DTK/UCX/MPI 版本、CCL-D 环境变量、原始日志、结构化结果和 monitor 退出码。

## 3. 故障与覆盖场景

### 3.1 正常基线

8 rank 双节点 AllReduce，覆盖 1K、4K、1M、64M 元素，100～1000 轮，并分别测试 Ring 和 Tree。诊断关闭和开启各执行一组。

检查 collective 次数、completion 完整率、rank 间 send/recv 差异、拓扑一致性、P50/P95/P99 时延、monitor 误报数和诊断开销。验收要求：正常场景误报为 0，连续 100 轮完成。

### 3.2 参数不一致

注入 count、datatype、collective 操作、root 或 counter 不一致。预期结论为 MISMATCH，同时给出异常 rank、counter、trace_id 和具体差异字段。测量检测延迟、定位准确率、证据完整性和实际通信结果。

### 3.3 rank 缺失或进程退出

覆盖进入 collective 前退出、完成一个 counter 后停止、以及仅丢弃诊断记录三种情况。预期结论为 MISSING_RANK 或 MISSING_COMPLETION，报告缺失 rank、最后正常 counter 和等待超时。monitor 必须按配置退出，不能无限等待。

### 3.4 通信推进慢

保留连续采样窗口，分别测试短暂抖动、单 rank 持续低速、所有 rank 同时低速、单 channel 或单 peer 低速。每个窗口保存 rank、counter、channel、send/recv delta、窗口时长、sendRate/recvRate 和协议。短暂抖动不报警，持续异常报告 COMM_SLOW，无进展升级为 COMM_STALLED。

### 3.5 kernel 停滞

在诊断侧模拟某 rank 停止更新 completion、停止更新 send/recv、某 channel 计数保持不变而其他 rank 继续推进。预期结论为 COMM_STALLED 或 MISSING_COMPLETION，必须包含最后进展时间、最后 counter 和对照 rank。

### 3.6 Ring 对称性

使用实际 topology_groups 比较每个 channel 的前驱、后继、send/recv 和边速率，定位异常 peer/channel。缺少拓扑字段时报告 RING_TOPOLOGY_PENDING，不能根据 rank 编号猜拓扑。

### 3.7 Tree 层级比较

按 tree_depth 比较 tree_up、tree_down 和同层中位数，区分整层变慢与单点变慢。缺少层级字段时报告 TREE_LEVEL_PENDING。

### 3.8 协议级比较

分别统计 Simple、LL、LL128 的 send/recv count、bytes、速率、rank 差异及 channel/peer 归属，重点检查单一协议异常而总量仍接近的情况。结果必须明确显示协议名称，不能只展示总 send/recv。

## 4. RCCL 多算子和长期覆盖

在合成故障通过后，执行 AllReduce、AllGather、ReduceScatter、Broadcast、AllToAll、Ring、Tree、多 communicator、多 stream 和 20～1000 轮压力测试。检查每条路径的 enqueue、sample、completion、算法、协议、channel、拓扑字段，以及计数器回退、slot 泄漏、日志截断和 monitor 超时治理。

## 5. 真实训练任务

通信库和故障矩阵稳定后再执行可控的 8 rank 数据并行训练，固定 batch 和迭代次数，每轮输出 training step，并在指定轮次注入故障：某 rank 提前退出、跳过一次梯度同步、同步前 sleep、停止更新通信计数。预期分别得到 MISSING_RANK、MISMATCH/MISSING_COMPLETION、COMM_SLOW/COMM_STALLED 和 COMM_STALLED。训练 step、collective counter、rank、通信速率和诊断结论必须能对齐。

## 6. 结果展示指标

探针侧展示 rank、counter、channel、采样窗口、sendRate/recvRate、Simple/LL/LL128 计数、peer、Ring 前后继、Tree depth 和 completion 状态。

分析器侧使用下表：

| 场景 | 结论 | 定位 | 检测延迟 |
|---|---|---|---:|
| 正常 8 rank | OK | 无异常 | — |
| 参数不一致 | MISMATCH | rank/counter/字段 | 实测 |
| rank 缺失 | MISSING_RANK | rank/最后 counter | 实测 |
| 通信停滞 | COMM_STALLED | rank/channel | 实测 |
| Ring 边异常 | EDGE_SLOW | peer/channel | 实测 |
| Tree 层异常 | TREE_LEVEL_SLOW | depth/rank | 实测 |

系统代价至少报告诊断开关两组的单步时间、P50/P95/P99 通信时延、吞吐、CPU、pinned memory、每 rank 日志量和长稳资源增长。

## 7. 验收门槛

| 验收项 | 目标 |
|---|---:|
| 正常场景误报率 | 0 |
| 参数不一致检测率 | 100% |
| 缺失 rank 检测率 | 100% |
| 合成停滞检测率 | 100% |
| 异常 rank 定位准确率 | 100% |
| Ring/Tree 拓扑字段完整率 | 100% |
| 双节点 8 rank 连续回归 | 至少 100 轮 |
| 长稳测试 | 至少 30 分钟 |
| 诊断开销和日志量 | 实测并报告 |

## 8. 执行顺序

单元测试与合成故障 -> 新版独立 RCCL -> 双节点正常基线 -> 多算子/多协议/Ring/Tree -> 参数/缺失/slow/stall 注入 -> 多 communicator/stream/长稳 -> 真实训练基线 -> 训练故障注入 -> 性能开销和结果汇总。

每完成一个阶段，保存原始日志、结构化结果、测试命令和 Git commit，并更新 docs/CCL-D_STATUS.md.

