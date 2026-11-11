# CCL-D 诊断记录生命周期复核

## 结论

提供的设计材料采用的是“通信操作级 TraceID + `kernelIndex` LRU body + pinned memory + 诊断线程/结束 callback”模型。当前工程实现的“`channel × (opCount % CCLD_DIAG_SLOTS)` 固定槽”只能作为临时缓存，不能作为最终的事件身份和生命周期模型。

在继续修改或重新编译 RCCL 前，必须先完成以下语义对齐：

1. 一个用户集合通信操作只分配一个 TraceID：`commId + counter + 扩展位`。
2. `counter` 在 host collective 入口按通信域递增；内部 work、channel、P2P 子任务不能重新生成独立的用户 counter。
3. `kernelIndex` 是诊断 body 的 LRU 物理索引，不是 TraceID，也不应通过 `opCount % N` 推导。
4. 一个操作的 body 按 channel 保存 sendCount/recvCount；同一操作内的多个内部 work 必须归入同一个 TraceID/body，而不是把第一个 work 的 opCount 当成整个 batch 身份。
5. kernel 完成后必须有明确的结束通知和 host 回收动作。诊断线程只负责周期采样和计算 sendRate/recvRate，不能单独承担 slot 回收确认。
6. slot 忙时不能静默跳过。必须输出显式的资源不足/观测丢失证据，并区分“通信未完成”和“诊断记录未采集”。

## 材料中明确的模型

设计材料描述了以下字段和流程：

- header 中的 `kernelIndex`：全局复用，用于 LRU 诊断 body；
- header 中的 `mode`：关闭、每 channel 的 tid 0 计数、或所有 tid 原子计数；
- header 中的 `counter`：集合通信操作总迭代次数；
- header 中的 `detectChannel`：本次操作的 channel 数；
- 每轮操作独享 `traceid = commId + counter + 扩展位`；
- body 分成 N 块，由 `kernelIndex` 选择；
- 每个 body/channel 保存 sendCount、recvCount；
- 诊断线程周期读取 pinned memory，计算 sendRate/recvRate；
- kernel host callback 通知诊断线程某次集合通信结束；
- 通信域销毁时回收诊断线程资源。

材料还强调 Ring 和 Tree 的比较口径不同：Ring 可以比较同一 collective 的所有 rank；Tree 只能比较同一层级的 rank。LL/LL128/Simple 的 send/recv 计数位置也不同，不能只在统一的 generic 末尾粗粒度累加。

## 当前实现与设计的差异

| 项目 | 当前实现 | 设计要求 | 影响 |
|---|---|---|---|
| 事件身份 | `opCount % CCLD_DIAG_SLOTS` | 独立 TraceID + LRU `kernelIndex` | 物理槽位和逻辑事件混淆 |
| 分配粒度 | 在 batch/kernel 侧选槽 | host 操作级分配 body | 多 work batch 归属不稳定 |
| 生命周期 | state 0/1/2，诊断线程轮询清理 | kernel 结束 callback + 诊断线程回收 | host 消费慢时可能丢记录 |
| 多 work batch | 入口读取一个 work 的字段 | 一个用户操作统一 TraceID，body 按 channel 累加 | P2P/AllToAll 容易出现错误关联 |
| AllToAll | API 展开为多组 P2P，日志难区分父操作 | 内部 work 继承父操作 TraceID | 只能看到 P2P，难还原 AllToAll 语义 |
| 采样周期 | 当前默认 10 ms | 材料默认约 200 μs 级周期 | slow 检测灵敏度和开销口径不同 |
| 协议计数 | 统一宏路径累加 | LL send/recv 与 Simple 的计数点不同 | 可能产生协议级偏差 |
| 丢失记录 | 已加入 dropped_batches | 应有显式资源/观测丢失状态 | 不能把诊断丢失当通信 hang |

## 正确的实现顺序

### 第一阶段：先建立操作级记录表

host 侧在 collective enqueue 时分配一个诊断记录：

```text
(comm_id, counter, trace_id, kernel_index, op, count, datatype,
 algorithm, protocol, detect_channel, state)
```

`kernel_index` 由 host 侧 LRU 分配，并在该操作结束 callback 到达后回收。device 只接收已分配的 `kernel_index`/trace metadata，不再通过 `opCount % N` 自己推导物理槽。

### 第二阶段：按 channel 累加同一操作

一个操作的多个 kernel work/batch 只更新同一操作对应的 body/channel 计数。若同一 channel 需要区分多个 work，增加 work/phase 字段；不要用第一个 work 的 opCount 覆盖整个 batch。

### 第三阶段：补齐结束通知和回收协议

定义明确的状态机：

```text
FREE → RESERVED → RUNNING → COMPLETE → CONSUMED → FREE
```

异常路径单独记录：

```text
RESOURCE_EXHAUSTED
OBSERVATION_DROPPED
KERNEL_ABORTED
```

`MISSING_COMPLETION` 只能在记录已经成功 RESERVED/RUNNING 后、预期结束通知未到达时产生；如果根本没有拿到记录，应报告观测资源问题，不能报告通信 hang。

### 第四阶段：协议级计数点复核

按材料重新核对：

- Simple：send/recv 在 generic 操作末尾；
- LL/LL128：send 在 waitSend 后，recv 在 generic 操作末尾；
- mode 1：优先 tid 0 低开销统计；
- mode 2：所有 tid 原子统计，仅用于精确实验。

## 在重新编译前必须完成的验证

不需要 GPU 编译即可先验证：

1. 一个 collective 是否只生成一个 TraceID；
2. 多 channel/multi-work 是否都能映射到同一个 operation record；
3. AllToAll 展开的 P2P 是否保留父操作类型；
4. LRU 回收和 slot 忙时的状态转移；
5. `MISSING_COMPLETION`、`OBSERVATION_DROPPED`、`RESOURCE_EXHAUSTED` 的 monitor 优先级；
6. Ring/Tree/协议计数的比较分组；
7. 200 μs、1 ms、10 ms 采样周期下的检测延迟和开销。

只有上述状态机和字段设计通过合成日志单测，才重新编译 RCCL。扩大 `CCLD_DIAG_SLOTS` 只能作为容量参数实验，不能替代 LRU 分配、TraceID 继承和 callback 回收。
