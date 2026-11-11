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

采用：**以一次 plan 的所有 channel/work 完成作为用户操作的整体结束**。

规则：

- 一个用户操作只有一个 TraceID 和一个 operation record；
- 每个 channel 在同一个 operation body 下保存自己的 `sendCount`/`recvCount`；
- 诊断线程可以在运行期间按 channel 周期采样；
- 单个 kernel/channel 结束只更新 channel 完成状态，不回收 operation record；
- plan 级结束通知到达且所有预期 channel/work 都完成后，才输出最终操作记录；
- 若 plan 级结束通知未到达，记录进入 `MISSING_COMPLETION`；
- 若 operation record 本身未分配或无法挂接，优先报告 `RESOURCE_EXHAUSTED`/`OBSERVATION_DROPPED`；
- communicator 销毁时，未完成 operation record 必须显式标记并回收。

选择理由：TraceID 在材料中表示“一轮集合通信操作”，不是一个 channel 或一个 kernel。plan 是当前 RCCL host 侧组织多 channel work 的边界，按 plan 聚合才能避免同一操作被拆成多个诊断事件；channel 级数据仍保留，满足 Ring/Tree 和速率分析。

## 3. 实现约束

这两项决策不保留旧的 `channel × (opCount % CCLD_DIAG_SLOTS)` 兼容路径。新的实现必须：

1. host 侧先创建 operation record 并分配 `kernelIndex`；
2. work 元数据携带 `kernelIndex` 和 operation identity；
3. device 按 operation body/channel 写入计数；
4. plan 完成通知驱动最终输出和回收；
5. slot 忙和 callback 缺失分别进入不同状态；
6. 单元测试先覆盖资源耗尽、跨 channel 聚合、重复回收和销毁清理，再进行 HIP 编译。
