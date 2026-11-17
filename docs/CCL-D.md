# CCL-D 当前设计

## 数据结构

`TraceId` 固定为 60 字节：`commId` 8 字节、`counter` 8 字节、预留区 44 字节。预留区没有字段定义，也不参与分析。

每个 pinned frame 由多个 slot 组成。slot header 包含 `kernelIndex`、`mode`、`counter`、`detectChannel` 和 `traceId`。每个 channel 只有两个 64 位字段：`sendCount` 和 `recvCount`。

## 身份和生命周期

TraceID counter 直接使用 RCCL 通信域 `opCount`，RecordPool 只选择空闲 slot。grouped P2P 的多个 Send/Recv 共享父 counter。首次 enqueue 初始化 slot，后续同 counter 调用复用 slot，不清零已有计数。所有 plan callback 完成后才关闭 header、释放 slot并输出 complete。

slot 不足时输出 `OBSERVATION_DROPPED`，不等待、不覆盖活动 slot、不影响通信。

## 线程模型

每个 communicator 启动独立 diagnose thread，周期读取 pinned frame、输出 sample，并在 kernel callback 通知后 drain 完成事件；communicator 销毁时停止并 join。legacy collTrace 使用 `RCCL_COLLTRACE_SAMPLE_US`，CCL-D 使用 `RCCL_CCLD_SAMPLE_US`。

## 采样指标

sample 记录包含当前 count、采样周期内的 `send_delta`/`recv_delta`，以及同义的 `send_rate`/`recv_rate` 增量。`send_frequency_hz` 和 `recv_frequency_hz` 是按实际间隔换算的派生字段。slow 判断只使用采样增量。

## 拓扑和决策

Ring 按 channel 比较全 rank；Tree 从 `tree_up`、`tree_down*` 和深度重建层级，按 channel、层级、父子角色分组。hang 状态为 `HANG_MISSING_RANK`、`HANG_PARTIAL_COMPLETION`、`HANG_ALL_RANKS`；计数异常为 `COUNT_MISMATCH`；slow 状态为 `COMM_SLOW`、`COMPUTATION_SLOW`、`MIXED_SLOW`。集合通信完成后的外部 hang 输出 `NON_COMMUNICATION_HANG_UNLOCALIZED`。

## 当前边界

源码和 host analyzer 已完成静态检查。新的 gfx936 编译、协议逐路径运行、双节点故障注入和训练级资源开销仍需验证。
