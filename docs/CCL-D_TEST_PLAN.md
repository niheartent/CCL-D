# CCL-D 实验设计与验收方案

本文按设计材料定义实验，不把数据面 PASS 当成诊断系统完成。每次实验必须保存 commit、RCCL build/install 前缀、节点/rank 映射、环境变量、原始 JSONL、结构化分析、退出码和未解决问题。

## 1. 先验证诊断生命周期

在重新编译 RCCL 之前，用合成状态机验证以下状态：

```text
FREE → RESERVED → RUNNING → COMPLETE → CONSUMED → FREE
```

异常状态必须分开：

```text
RESOURCE_EXHAUSTED
OBSERVATION_DROPPED
KERNEL_ABORTED
MISSING_COMPLETION
```

验收要求：诊断记录未分配不能被报告为通信 hang；slot/record 忙不能静默覆盖；同一操作的多 channel、多 work 必须能回到同一个 TraceID。

## 2. 设计材料中的记录模型

一次用户集合通信操作使用：

```text
TraceID = commId + counter + 扩展位
```

`kernelIndex` 是 pinned memory body 的 LRU 物理索引，不是 TraceID。body 按 channel 保存 sendCount/recvCount。diagnostic thread 周期读取 body 计算 sendRate/recvRate，kernel host callback 通知操作结束并触发回收。

实验必须分别验证：

- 同一通信域 counter 在所有 rank 对齐；
- 多 communicator 的 TraceID 不串；
- 多 channel/multi-work 归属同一个操作 body；
- callback 到达后 body 才能回收；
- 记录未分配、记录丢失、kernel 未完成是三个不同结论。

## 3. 正常基线

在单节点和双节点分别执行 AllReduce、AllGather、ReduceScatter、Broadcast、AllToAll 和显式 P2P。覆盖 1K、4K、1M、64M 元素以及 20、100、1000 轮。

检查：

- 数据结果；
- enqueue/TraceID 数量；
- body/channel completion 数量；
- `dropped/RESOURCE_EXHAUSTED`；
- Simple、LL、LL128 计数；
- Ring/Tree 分组；
- 日志完整性和资源增长。

正常基线要求数据正确、无观测丢失、无误报。若诊断记录丢失，只能标记该样本不可用于诊断验收，不能把它作为正常通过样本。

## 4. 操作级契约实验

| 操作 | 检查契约 | 故障结果 |
|---|---|---|
| AllReduce | count、datatype、reduction op、TraceID 对齐；不比较 root | 参数不一致 |
| ReduceScatter | 输入分片、datatype、reduction op 和 TraceID 对齐 | shape/reduction mismatch |
| AllGather | 每 rank sendcount、datatype、接收规模与 nranks 关系 | shape mismatch |
| Broadcast | 所有 rank root 相同且有效；root/非 root 角色完整 | root mismatch |
| AllToAll | 父操作 TraceID 保留；展开后的 rank→peer 矩阵完整且收发互为转置 | edge/parent mismatch |
| P2P | Send(A→B) 与 Recv(B←A) 成对，方向、count、datatype、bytes 对齐 | peer/transfer mismatch |

AllToAll 展开成 P2P 时不能只根据 `op=Send/Recv` 解释，必须保留 parent operation。

## 5. Ring/Tree/协议实验

Ring：同一 collective 内比较所有 rank 的 send/recv count 和 sendRate/recvRate，并验证实际前驱/后继互反。

Tree：只比较相同 `tree_depth`、相同 channel 的 rank；不能把不同层级直接比较。

协议：严格按设计材料验证计数点：

- Simple：send/recv 在 generic 操作末尾；
- LL/LL128：send 在 waitSend 后，recv 在 generic 操作末尾；
- mode 1：channel tid 0 低开销统计；
- mode 2：所有 tid 原子统计，用于精确对照实验。

采样周期至少比较 200 μs、1 ms、10 ms，分别报告检测延迟、CPU、pinned memory 和误报情况。

## 6. 故障实验

依次执行：

1. rank 未进入 collective；
2. 参数/操作类型不一致；
3. kernel 未完成；
4. 单 rank 通信推进慢；
5. 单 channel 停滞；
6. AllToAll/P2P 缺失边；
7. diagnostic record 未分配或被丢弃；
8. 多 communicator 或 stream 串扰。

每个故障只改变一个因素，记录状态、root cause、首错 rank、TraceID、kernelIndex、channel、peer、protocol 和检测延迟。

当前 `RCCL_CCLD_FAULT=*` 主要改变诊断记录，不能宣称为真实 GPU hang、驱动错误或网络中断。真实故障必须单独设计超时、清理和日志隔离。

## 7. 训练和性能实验

诊断稳定后再运行实际训练任务，固定 batch、迭代次数和通信规模，在指定 step 注入：rank 退出、同步前 sleep、通信计数停止和参数不一致。

性能实验至少报告：单步时间、P50/P95/P99 通信延迟、吞吐、CPU、pinned memory、GPU 影响、日志量、检测延迟和长稳资源变化。没有这些对照数据，不下“开销可控”的结论。

## 8. 当前不允许默认通过的项目

- `opCount % N` slot 方案不能作为材料级 LRU 实现；
- slot 数量扩大不能替代 LRU/callback/回收状态机；
- batch 内第一个 work 的元数据不能代表整个 batch；
- AllToAll 展开的 P2P 不能默认等于显式 P2P；
- 合成 slow/hang 不能等价于真实 GPU/网络故障；
- 单次五算子数据 PASS 不能等价于诊断系统完成。
