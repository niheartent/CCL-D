# CCL-D 实施路线

本路线以提供的设计材料为准。每个目标都必须有源码、单测、双节点实测和原始日志；没有证据的项目保持“待验证”。

## P0：材料级生命周期复现

1. 明确 `commId + counter + 扩展位` TraceID。
2. 实现 host 侧 `kernelIndex` LRU 分配和回收。
3. 用 pinned memory body 保存每个操作的 channel send/recv。
4. 建立 `FREE → RESERVED → RUNNING → COMPLETE → CONSUMED → FREE` 状态机。
5. 用 callback/结束通知驱动回收，诊断线程负责周期采样和输出。
6. 明确 multi-work/batch 到 operation TraceID 的继承关系。

## P1：指标和路径

1. 按材料核对 Simple、LL、LL128 的真实计数点。
2. 保留 AllToAll 的 parent operation，同时记录子 P2P 边。
3. 对显式 P2P 使用方向性 Send/Recv TraceID。
4. Ring 全局比较，Tree 同层比较。
5. 覆盖 AllReduce、AllGather、ReduceScatter、Broadcast、AllToAll 和 P2P。

## P1：验证矩阵

- 单节点/双节点；
- 2/4/8 rank；
- 1K/4K/1M/64M；
- 20/100/1000 轮；
- 200 μs/1 ms/10 ms 采样周期；
- 参数不一致、未进入、未完成、slow、stalled、观测丢失；
- 多 communicator、多 stream 和资源回收。

## P2：扩展路径和真实故障

- graph、CollNet、NVLS、MSCCL；
- UCX/网络异常；
- 进程退出、GPU kernel 超时；
- 实际训练任务；
- CPU/pinned/GPU/吞吐开销和 30 分钟长稳。

在 P0 生命周期和合成状态机没有通过前，不重新编译 RCCL，也不把扩大 slot 数量作为解决方案。
