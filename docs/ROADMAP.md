# CCL-D 实施路线

本路线按提供的设计材料和当前 RCCL 源码实际生命周期重新整理。优先修复记录身份和生命周期，再扩展指标、分析器和故障实验。没有源码、单测和运行证据的项目保持“待验证”。

## 当前方向纠正

当前 device kernel 在 `src/device/common.h` 中按：

```text
channel × (opCount % CCLD_DIAG_SLOTS)
```

选择诊断槽，并在 batch 入口初始化、batch 结束置为 complete。这个方案不能作为材料级实现，因为物理槽位被当成了操作身份，且一个用户操作的多个 channel/work 不一定共享同一 body。

后续不再沿着“扩大 slot 数量、继续堆字段、先完善 analyzer”的方向推进。必须先把操作级记录和生命周期改正确；否则任何速率、拓扑或 slow/hang 结论都可能建立在错误记录上。

## P0：建立材料级操作记录

### P0.1 host 操作记录

在 collective/P2P 被加入 planner 时建立诊断操作上下文，至少保存：

```text
commId, counter, traceId, op, count, datatype, root,
algorithm/protocol（确定后补齐）, detectChannel, kernelIndex, state
```

`counter` 使用通信域的统一操作计数；显式 P2P 的方向和 parent operation 另存字段。device 不再自己通过 `opCount % N` 推导物理槽。

### P0.2 kernelIndex LRU

host 侧实现固定数量 body 的 LRU 分配器：

```text
FREE → RESERVED → RUNNING → COMPLETE → CONSUMED → FREE
```

分配失败必须输出 `RESOURCE_EXHAUSTED` 或 `OBSERVATION_DROPPED`，不能静默复用或把它报告成通信 hang。`kernelIndex` 只是 pinned body 的物理索引，不能写入 TraceID 语义。

### P0.3 把索引和 TraceID 传入 work

为 collective work、P2P work 和 AllToAll 展开的子 work 增加诊断元数据传递路径。一个用户操作的多个 channel、多个 work/batch 必须回到同一个操作记录；AllToAll 子边必须保留 parent TraceID。

这一步需要先确认 work 结构的 ABI/大小预算和 FIFO/graph/persistent 三种存储路径，确认后再修改结构并统一编译。

## P1：按材料实现 device 计数

1. body 按 channel 保存同一操作的 `sendCount`/`recvCount`。
2. mode 1 只统计每 channel 的 tid 0；mode 2 才启用所有 tid 原子统计。
3. Simple：send/recv 在 generic 操作末尾。
4. LL/LL128：send 在 `waitSend` 后，recv 在 generic 操作末尾。
5. 协议计数与 peer/channel 计数都写入同一 operation body，不能继续写入由 batch 自行选择的槽。

先用 host/device 结构和合成状态机单测验证字段映射，再编译 RCCL。一次改完共享 work/slot 逻辑后，再跑所有算子，不为每个算子单独编译 RCCL。

## P2：结束通知与回收

RCCL 已有 plan、strong stream 和 callback 队列，但现有 callback 主要负责 plan/proxy 资源回收，尚未证明它已通知 CCL-D 诊断记录。因此需要：

1. 确认一个 plan 的所有 channel/work 已完成的边界；
2. 将该边界连接到诊断操作的结束通知；
3. 由诊断线程读取 pinned body、输出最终记录，再进入 `CONSUMED`；
4. 由 host 回收 `kernelIndex`；
5. communicator 销毁时停止诊断线程并回收所有未完成记录。

没有完成这条链路前，不能把轮询看到 `state=2` 当成材料中的 callback 回收实现。

## P3：分析器和拓扑规则

生命周期稳定后再完善集中式分析：

- 通用完整性：TraceID、rank、counter、completion、记录丢失；
- Ring：同一操作比较所有 rank；
- Tree：只比较同一 channel、同一 depth 的 rank；
- P2P：方向、peer、count/bytes 和 Send/Recv 配对；
- AllToAll：父操作完整性和展开后的 rank→peer 矩阵；
- slow：按采样窗口的 sendRate/recvRate；
- hang：未进入、未完成、无进展和观测资源耗尽分开归类。

分析器可以先用合成记录开发，但合成记录只能验证规则，不得作为 RCCL 插桩完成的证据。

## P4：验证和编译顺序

1. host LRU、状态机、TraceID 继承和资源耗尽单测；
2. 单节点最小 collective，确认一个操作跨 channel 只有一个 TraceID/body；
3. 单节点五类 collective + 显式 P2P；
4. 双节点 8 rank，检查所有 rank 的 enqueue/body/completion；
5. 20/100/1000 轮和多 communicator/stream；
6. Simple、LL、LL128 的真实协议矩阵；
7. Ring/Tree 拓扑比较；
8. 诊断侧 slow/stall/missing completion 注入；
9. 真实进程退出、GPU 超时、网络异常和训练任务；
10. CPU、pinned memory、GPU、吞吐和长稳开销。

采样周期按材料的约 200 微秒默认值、示例中的 1 毫秒精度以及当前 10 毫秒实现做对照实验；在对照完成前不把任何一个周期写成唯一验收值。

## 暂缓项目

Graph、CollNet、NVLS、MSCCL 以及真实 UCX/网络故障属于扩展验证，暂不阻塞 P0～P3 的基本材料级诊断链路。但它们不能在文档中被写成已覆盖。
