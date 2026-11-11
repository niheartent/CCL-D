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

peer 采集使用每 channel 的有限槽位；记录中必须包含 peer 槽溢出状态。出现溢出时结果标为 `PEER_DATA_TRUNCATED`，该轮不能作为拓扑和流量完整性通过样本，不能将缺失的 peer 计数当作 0。

诊断槽按 `channel × opCount % CCLD_DIAG_SLOTS` 选择，同一 CTA 内每个 work batch 独立开始和提交。压力测试需要检查 slot 是否被后续 batch 覆盖，以及每个 batch 的协议和 peer 明细是否与总计数一致。

## 4. RCCL 多算子和长期覆盖

在合成故障通过后，执行 AllReduce、AllGather、ReduceScatter、Broadcast、AllToAll、Ring、Tree、多 communicator、多 stream 和 20～1000 轮压力测试。检查每条路径的 enqueue、sample、completion、算法、协议、channel、拓扑字段，以及计数器回退、slot 泄漏、日志截断和 monitor 超时治理。

### Monitor 语义分层与实验设计

monitor 采用“通用完整性检查 + 算子契约检查 + 拓扑检查”三层结构。通用层只负责 `comm_id/counter` 分组、rank 是否齐全、completion 是否完成、计数器是否回退；算子层按通信契约解释同一批记录，避免用一条规则误判不同 collective。

| 分析器 | 适用操作 | 核心契约 | 故障注入/验收 |
|---|---|---|---|
| P2P edge analyzer | Send/Recv | Send(A→B) 必须有 Recv(B←A)，count/bytes/datatype 对齐 | 删除对端 Recv → `P2P_PEER_MISSING`；改变 count → `P2P_TRANSFER_MISMATCH` |
| Broadcast role analyzer | Broadcast | 所有 rank 的 root 相同且在 `[0,nranks)` | 改一个 root → `BROADCAST_ROLE_MISMATCH` |
| AllToAll shape/edge analyzer | AllToAll | 所有 rank 的 count/datatype 一致；completion 的 peer 矩阵应互为转置 | 改一个 rank 的 count → `ALLTOALL_SHAPE_MISMATCH`；删一条边 → edge mismatch |
| Gather/scatter shape analyzer | AllGather/ReduceScatter | 输入/输出规模按 nranks 成比例，datatype 一致；ReduceScatter 的归约参数一致 | 改 count/datatype → `COLLECTIVE_SHAPE_MISMATCH` |
| Reduction analyzer | AllReduce/ReduceScatter | reduction op、datatype、count 一致；不比较 root | 改 reduction 参数 → 参数不一致 |
| Topology analyzer | Ring/Tree | Ring 前后继互反；Tree 按 depth 分层比较 | 改 topology peer/depth → `RING_ASYMMETRY`/`TREE_ASYMMETRY` |

实验顺序固定为：正常样本 -> 单字段参数错误 -> 缺失 rank/缺失边 -> completion 未完成 -> 单 rank slow/stall -> peer/channel 计数错误 -> 长窗口回归。每个故障只改一个证据字段，并检查状态、`root_cause`、首错 rank、证据字段和 monitor 退出码；正常样本要求无误报。

当前 Python analyzer 已加入上述 P2P、Broadcast、AllToAll 和基础集合形状规则，使用合成 JSONL 做确定性单测；这些规则不改变 RCCL ABI，因此不需要等待本轮 RCCL 编译完成后再开发。真实五算子双节点回归仍需使用新安装库验证。

### Slot 回归实现经验

- **以 batch 为粒度记录**：一个 channel/kernel batch 内可能有多个 work item；slot 不能只在 kernel 入口初始化一次。每个 work batch 都应从自己的 work 元数据取 `opCount`、算法和协议，并独立提交 completion。
- **先分辨 work 类型再解释字段**：collective work 与 `ncclDevWorkP2p` 的布局和计数语义不同。P2P 需要从 send/recv 的 opCount、协议标记和 peer 字段取证，不能把其内存强转成 collective 结构。
- **slot 只负责暂存，不是事件身份**：跨 rank 聚合使用 communicator 与 opCount/trace_id；channel、peer、protocol 是该事件的观测维度。循环复用 slot 时必须验证记录绑定仍指向正确 batch。
- **将数据正确性与插桩正确性分开验**：每类 collective 用 rank 编码的输入验证输出；另检查每个 completion 的计数、协议、peer 明细和 overflow。零填充只能证明调用返回，不能证明数据分发正确。
- **P2P 使用对端配对语义**：Send 和 Recv 的 rank 预期不同，不能拿它们套 collective 的 root/参数一致性比较。数据面按环形邻居验证，日志按 send-peer/recv-peer 成对核对。
- **一次改完再编译**：先合并 RCCL 共享 batch/slot 修复和统一回归程序，再执行一次 gfx936 RCCL 构建、安装和五算子双节点矩阵。路径失败后才对对应路径增量修复，避免每个算子触发一次完整 hipcc 链接。

当前已增加统一多算子 slot 回归程序源码 `tools/tests/multinode_collective_slot_regression.cpp`，编译产物为 `ccld-artifacts/multinode-test/multinode_collective_slot_regression`，一次运行覆盖 AllGather、ReduceScatter、Broadcast、AllToAll 和 P2P Send/Recv。脚本为 `tools/tests/run_multinode_collective_slot_regression.sh`，使用当前独立 RCCL 安装，不为每个 collective 单独编译 RCCL。

该回归统一编译一次、顺序覆盖五类数据面操作。经验是：先在 RCCL 层统一修复 batch/slot 生命周期，再复用同一安装前缀编译一个回归程序；不要为每个 collective 单独编译 RCCL。普通 collective 使用统一 monitor 判定；P2P 采用独立 Send/Recv 对端语义，不能套用普通 collective 的 root/rank 对照规则。

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

