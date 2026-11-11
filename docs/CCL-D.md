# CCL-D 复现说明

本文只把提供的设计材料、当前源码和已经运行过的测试分开记录。材料中没有明确、源码中没有实现或没有实测的数据，不写成“已完成”。

## 1. 材料明确的诊断模型

提供的材料把系统分成通信库侧高精度指标采集和集中式决策分析两层。目标是将一次通信域内的同一集合通信关联起来，并用用户状态、kernel 状态、每个 channel 的收发计数和变化速率区分 slow/hang。

材料明确给出的操作标识是：

```text
TraceID = commId + counter + 扩展位
```

其中 `commId` 标识通信域，`counter` 在每次集合通信开始时递增，扩展位预留。材料明确要求同一通信域同一轮在各 rank 上保持一致。

材料描述的诊断内存模型是连续 pinned memory：header 含 `kernelIndex`、`mode`、`counter`、`detectChannel` 和本轮 `traceid`；body 按 `kernelIndex` 选择，按 channel 保存 `sendCount`/`recvCount`。`kernelIndex` 是全局复用的诊断操作 LRU 次数，不是 TraceID。

材料描述的线程关系是：kernel 侧记录收发成功次数；diagnose thread 定期读取 pinned memory 计算 `sendRate`/`recvRate`；每个 kernel 结束触发 host callback，由 callback 通知 diagnose thread 输出本次集合通信结果；通信域销毁时回收诊断线程资源。

材料给出的采样口径有两个需要区分的表述：正文写“默认 200 微秒”周期；示例说明使用 1 毫秒统计精度，并提醒统计精度会影响异常 rank 定位。因此本工程把采样周期作为待实测参数，不能直接把某一个周期写成唯一验收标准。

## 2. Ring、Tree 和协议比较口径

材料明确要求协议计数点不同：LL/LL128 的 send 在 `waitSend` 后统计，recv 在 `genericOp` 末尾统计；Simple 的 send/recv 都在 `genericOp` 末尾统计。

Ring 比较同一通信域同一次集合通信的所有 rank；Tree 只比较同一 tree 层级的 rank。比较指标包括集合通信开始/结束时间、操作耗时、算法带宽、操作类型、每 channel 的 `sendRate`/`recvRate` 及最终 `sendCount`/`recvCount`。跨节点绝对时间戳不能直接作为核心比较依据。

## 3. 当前源码实际情况

当前源码已经存在以下诊断侧代码，但它们不能被描述成材料级完整实现：

- host enqueue、RCCL `opCount`、操作类型、rank、count/bytes 等 JSONL 记录；
- device 侧 Simple、LL、LL128 原语中的统一 `CCLD_DIAG_ADD` 计数宏；
- pinned host 诊断区域、channel 维度计数、协议计数、peer 字段和周期轮询；
- completion/sample 输出，以及 Python analyzer/monitor 的参数、缺失、速率和基础拓扑规则。

源码当前仍在 device 侧使用：

```text
channel × (opCount % CCLD_DIAG_SLOTS)
```

来选择诊断槽。该实现与材料描述的 host 操作级 LRU `kernelIndex` 不同。当前代码还没有证实以下材料级语义：

1. host 在集合通信入口分配并保存可复用的 `kernelIndex` LRU body；
2. 一个用户操作的多个 channel、多个 work/batch 都继承同一 TraceID/body；
3. kernel host callback 驱动结束通知和 body 回收；
4. AllToAll 展开的内部 P2P work 保留父操作 TraceID；
5. Simple、LL、LL128 已按材料规定的原语位置分别完成精确计数。

因此，当前 `dropped_batches`、扩大 slot 数量、一次数据面回归通过，都不能证明诊断生命周期已经复现。

## 4. 已有测试证据

- Python analyzer/monitor 单元测试：最近一次 `39` 项通过。
- 五类数据面回归：AllGather、ReduceScatter、Broadcast、AllToAll、P2P 在已有 8 rank 双节点程序中通过结果校验；这只证明通信数据面样本通过。
- 参数不一致、缺失 rank、合成 slow/stall/missing completion：已验证 analyzer/monitor 分支；注入只改变诊断记录，不等价于真实 GPU kernel hang、驱动错误或网络故障。
- 已有独立 gfx936 构建和安装目录；最近 slot 生命周期修正后没有重新安装并完成双节点诊断验收。

## 5. 未确认项目

以下内容必须在源码审计、状态机单测和重新编译后再下结论：

- `kernelIndex` LRU 分配、占用、回收和资源耗尽语义；
- `FREE → RESERVED → RUNNING → COMPLETE → CONSUMED → FREE` 生命周期；
- 多 work/batch 到一个操作 TraceID 的继承；
- callback 与诊断线程的真实连接；
- 协议级精确计数、Ring 全局比较、Tree 同层比较的真实多节点覆盖；
- 多 communicator、stream/graph、CollNet、NVLS、MSCCL 路径；
- 训练任务中的真实 slow/hang、网络异常和性能开销。

在上述项目明确前，本仓库只能称为“CCL-D 诊断链路的部分复现”，不能称为完整材料复现。

## 6. 复核顺序

1. 只读核对 host enqueue、plan、work batch、kernel、callback、diagnostic thread 的真实生命周期。
2. 用合成状态机单测验证 TraceID、`kernelIndex`、body/channel 映射和资源耗尽优先级。
3. 明确 AllToAll 父操作和 P2P 子 work 的身份继承。
4. 按材料逐个核对 Simple、LL、LL128 计数插入点。
5. 通过上述单测后，重新构建并安装 gfx936 RCCL。
6. 用干净日志目录重新执行单节点、双节点、多 collective 和故障矩阵。

相关状态与实验定义见 [`CCL-D_STATUS.md`](CCL-D_STATUS.md)、[`CCL-D_TEST_PLAN.md`](CCL-D_TEST_PLAN.md) 和 [`CCL-D_SLOT_LIFECYCLE_REVIEW.md`](CCL-D_SLOT_LIFECYCLE_REVIEW.md)。
