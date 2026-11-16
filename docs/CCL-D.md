# CCL-D 复现模型

CCL-D 复现材料定义的通信诊断链路：RCCL 在通信路径采集证据，pinned-memory body 保存 channel 状态，诊断线程输出指标，monitor 按通信域和操作身份做跨 rank 分析。

一次用户 collective 使用 `TraceID = commId + counter + 扩展位` 标识。`kernelIndex` 指向 LRU body；同一 TraceID 的 work、channel 和 plan 必须保持身份隔离，body 完成后才能回收。资源不足只影响观测，不得改变通信行为或借用其他操作的诊断身份。

材料明确的指标包括集合通信开始/结束时间、算法、协议、操作规模、sendCount/recvCount 和 sendRate/recvRate。Ring 按同一通信域横向比较；Tree 按同层 rank 比较。LL/LL128 的 send/recv 计数点与 Simple 不同，必须分别验证。

当前已实现 host/device 记录、channel/peer 计数、协议字段、P2P/AllToAll 父子关联、Ring/Tree 基础分析和故障证据注入。尚未完成的工作只保留在 [状态文档](CCL-D_STATUS.md) 和 [测试计划](CCL-D_TEST_PLAN.md) 中。
