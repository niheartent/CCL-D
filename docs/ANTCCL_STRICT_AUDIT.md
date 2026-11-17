# AntCCL 严格实现审计

本审计只接受材料中定义的结构、线程模型、决策树和评测场景。通信能运行、近似注入能报警、单元测试通过，都不能替代对应验收项。

## 结构逐项核对

| 材料要求 | 当前代码 | 严格结论 |
|---|---|---|
| TraceID = commId + counter + 扩展位 | `src/include/ccld_data.h` 定义 8+8+44 字节；44-byte 是保留存储区，当前不赋予额外语义 | 布局符合；counter 语义未符合 |
| header: kernelIndex/mode/counter/detectChannel/traceid | `ncclCcldDiag` 使用对应字段 | 已符合 |
| body: 每个 KID、每个 channel 的 sendCount/recvCount | `ChannelBody` 为两个 64-bit 计数 | 已符合 |
| mode=0/1/2 语义 | host/device 使用 0、1、2；mode=1 为 tid==0 计数，mode=2 为全部线程计数 | 已符合代码语义，尚缺三协议逐路径实测 |
| 连续 pinned memory | communicator 初始化时分配 pinned frame，slots 可由环境变量设置 | 结构已实现，严格资源验收未完成 |
| 与 RCCL collTrace 解耦 | CCL-D frame、communicator/device 字段、分配和计数宏已移出 `ENABLE_COLLTRACE`；CCL-D 不再启动 legacy collTrace 线程，也不再读取其 diag 计数 | 源码已符合；最终 gfx936 构建和运行验收待完成 |
| LRU 按 kernelIndex 复用 body | host `RecordPool` 选择空闲 block，device 通过 slot 索引；同一 TraceID 记录 pending plan 数，最后一个 kernel callback 才释放 | counter 已改为通信域 `opCount`；grouped P2P 共用父 counter 的代码已加入，待后续专门静态回归与构建后运行验收 |

## 线程模型核对

材料明确要求：

1. diagnose thread 定时读取 pinned buffer，计算 sendRate/recvRate；
2. kernel thread 更新 sendCount/recvCount；
3. kernel hostcallback 在 kernel 完成后触发；
4. hostcallback 通知 diagnose thread 当前操作完成；
5. xccl main thread 回收 diagnose thread 资源。

当前代码已创建独立 diagnose thread：它周期读取 pinned frame、调用采样/输出逻辑，并在 communicator 销毁时停止并 join；kernel hostcallback 通过条件变量通知该线程。回调粒度为每个 kernel plan；同一 TraceID 的多个 plan 只在最后一个 callback 回收 frame 并输出 complete。该改动正在进行 gfx936 增量构建，构建和多节点回归完成前仍标为待验收。

算法带宽字段已加入输出；材料要求该字段但未给出公式，当前实现采用 RCCL replayer 的 AllReduce、AllGather、ReduceScatter bus-bandwidth 倍率，必须经过运行数据核对后才能验收。

## 决策树核对

### slow

材料要求的顺序是：任务 trace 是否超过基线 → 获取 rank trace → 判断步调是否一致 → 判断 sendRate/recvRate 是否一致 → 区分计算慢、通信慢和混合场景。

当前 analyzer 已按该顺序保留告警、步调偏差、收发速率比较，并输出 `COMPUTATION_SLOW`、`COMM_SLOW`、`MIXED_SLOW` 及证据字段；跨节点不使用未同步时钟比较。仍需用材料 5.1 的真实注入场景进行端到端验收。

### hang

材料要求的顺序是：是否存在未进入 collective 的 rank → 是否存在进入但未 hang 的 rank → 是否所有 rank 都进入且都 hang → 对同算法同位置 rank 比较收发次数定位首错。

当前 analyzer 能报告 missing、incomplete、部分 count mismatch，但没有完成按算法拓扑位置比较收发次数的统一根因路径，也没有非通信 hang 的明确结果分支。因此 hang 决策树未完成。

## 功能评测核对

材料第 5.1 节的六个场景必须逐项通过：

| 场景 | 当前严格结论 |
|---|---|
| rank 未进入 collective | 未完成；现有测试会 abort，未完成材料要求的完整诊断闭环 |
| kernel 内异常 hang | 未完成；现有 stall 在通信开始前 |
| rank 2 少 1024 B | 未完成；只有 analyzer 单元数据，没有双节点运行注入 |
| collective 后非通信 hang | 未完成 |
| collective 前计算慢 3 s | 未完成 |
| 每次 send/recv 后通信慢 2 us | 未完成 |

## 性能评测核对

材料第 5.2 节要求训练耗时、吞吐、loss、CPU、pinned memory、GPU 开销均在 1% 口径内。当前没有训练实验；HIP event 微基准不能替代训练评测，采样区 CPU 不能替代独立 diagnose thread CPU，frame 字节记录不能替代 pinned memory 峰值对照。因此性能评测整体未完成。

## 当前唯一可直接宣称的结果

最终 callback 快照版本已通过一项工程回归：双节点 8 rank、同一 group 内 256 个 AllReduce 加 P2P、运行 2 轮，通信结果正确，slot 回收后 516 个 counter/rank 均有 complete，观测丢失为 0。这只证明该 grouped-work 生命周期路径，不能代表 AntCCL 评测完成。

## 本次图片与正文复核新增结论

- TraceID 的 44-byte 区域只是预留存储区；当前布局正确。counter 已改为通信域内集合操作的统一 `opCount`，RecordPool 不再生成身份；grouped P2P 的多个 Send/Recv 复用同一父 counter。该模块仍需在编译后用多 rank grouped P2P 验证跨 rank 一致性。
- 采样输出已删除不符合材料语义的 `send_rate_Bps/recv_rate_Bps` 字段，改为固定采样周期内的 `send_rate/recv_rate` 原始计数增量，并将按时间换算的值单独命名为 `send_frequency_hz/recv_frequency_hz`。分析器现已只使用这些采样增量做 slow 判定；没有采样证据时不再用 complete duration 伪造通信速率。
- slow 决策树的顺序已部分体现，但 hang 决策树尚未完整实现“未进入、部分完成、全部进入且都 hang、同算法同位置计数比较、非通信 hang”五个分支。
- `algorithm_bandwidth_Bps` 的计算公式是实现自行选择的 RCCL bus-bandwidth 倍率，材料没有给出该公式；在没有运行数据核对前只能标为待验收。
