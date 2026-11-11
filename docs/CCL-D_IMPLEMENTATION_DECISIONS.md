# CCL-D 未定义语义的实现决策

论文材料明确了 LRU、pinned memory、`kernelIndex`、按 channel 记录收发计数和 callback 通知，但没有规定 LRU 资源耗尽策略，也没有规定多 channel 操作的整体完成聚合方式。以下两项是本工程的实现决策，不宣称是论文原文要求。

## 1. LRU body 用尽

采用：**记录丢失但通信继续**。

规则：

- 活动 body 不允许覆盖；
- 不阻塞 RCCL collective，不让诊断资源耗尽导致通信死锁；
- 不自动扩容 pinned body，避免运行期间改变内存布局和诊断开销；
- 如果操作没有成功分配 body，输出 `RESOURCE_EXHAUSTED`；
- 如果已识别操作但某个 channel/work 无法挂接，输出 `OBSERVATION_DROPPED`；
- 这两种状态不能解释成 `MISSING_COMPLETION` 或 GPU/network hang；
- 数据面结果仍然有效，但该样本不能作为诊断完整性验收样本。

选择理由：诊断是旁路能力，不能反过来阻塞通信；静默覆盖会破坏 TraceID 和计数证据，自动扩容会使 pinned memory 和开销不可控。显式标记丢失可以保留通信可用性，同时避免误报。

## 2. 多 channel 操作的结束边界

采用：**以同一 TraceID 对应的所有 plan/work/channel 完成作为用户操作的整体结束**。

规则：

- 一个用户操作只有一个 TraceID 和一个 operation record；
- 每个 channel 在同一个 operation body 下保存自己的 `sendCount`/`recvCount`；
- 诊断线程可以在运行期间按 channel 周期采样；
- 单个 kernel/channel 结束只更新 channel 完成状态，不回收 operation record；
- RCCL 可能因 work/FIFO 预算把一个操作拆到多个 kernel plan；不能把任意单个 plan callback 当成操作结束；
- host 记录该 TraceID 的预期 channel/work 片段，所有相关 plan 的完成通知到齐后，才输出最终操作记录；
- 若 plan 级结束通知未到达，记录进入 `MISSING_COMPLETION`；
- 若 operation record 本身未分配或无法挂接，优先报告 `RESOURCE_EXHAUSTED`/`OBSERVATION_DROPPED`；
- communicator 销毁时，未完成 operation record 必须显式标记并回收。

选择理由：TraceID 在材料中表示“一轮集合通信操作”，不是一个 channel 或一个 kernel。RCCL plan 是 work 提交边界，但一个操作可能跨多个 plan；因此按 TraceID 聚合全部相关 work/channel 完成事件，避免把一个操作拆成多个诊断事件。channel 级数据仍保留，满足 Ring/Tree 和速率分析。

## 3. 实现约束

这两项决策不保留旧的 `channel × (opCount % CCLD_DIAG_SLOTS)` 兼容路径。新的实现必须：

1. host 侧先创建 operation record 并分配 `kernelIndex`；
2. work 元数据携带 `kernelIndex` 和 operation identity；
3. device 按 operation body/channel 写入计数；
4. 所有相关 plan/work 的完成通知驱动最终输出和回收；
5. slot 忙和 callback 缺失分别进入不同状态；
6. 单元测试先覆盖资源耗尽、跨 channel 聚合、重复回收和销毁清理，再进行 HIP 编译。

## 4. 运行中采样与最终收口

采用“运行中按 channel 采样、结束时按 TraceID 收口”的两阶段处理：

- body 正在运行时，diagnostic thread 按 channel 读取 pinned 计数，输出 sendRate/recvRate 和局部进展；
- channel/kernel 完成只更新该 TraceID 的完成集合，不立即回收 body；
- host 记录该 TraceID 涉及的全部 plan/work/channel；
- 所有相关完成通知到齐后，输出一次 operation-level 最终结果；
- 诊断线程消费最终 body 后立即进入 `CONSUMED → FREE`；
- body 不足时只产生显式观测缺口，不能阻塞通信，也不能把该样本当作正常诊断通过。

这样既保留材料要求的高频 channel 指标，又避免多 plan/channel 被拆成多个操作身份。
