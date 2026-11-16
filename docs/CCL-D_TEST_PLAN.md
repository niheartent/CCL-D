# CCL-D 核心测试计划

每项实验保存 commit、RCCL build/install 前缀、rank 映射、环境变量、原始 JSONL、monitor 输出和退出码。

## TraceID 与生命周期

覆盖多 plan/work/channel、AllToAll 父子 P2P、重复完成、缺失完成和 communicator 销毁。

通过条件：TraceID 不串操作；所有 body 只收口一次；无 body 的 work 不产生 completion；通信不因观测资源不足阻塞。

## 协议计数

分别触发 Simple、LL、LL128，核对材料规定的计数位置、方向、channel 和总字节数；mode 1 只由 tid0 统计，mode 2 验证原子计数结果。

通过条件：协议归属正确，计数与已知传输规模相符，不能用其他协议的结果推断当前协议。

## 双节点回归

HPDPS-6/7 运行 AllGather、ReduceScatter、Broadcast、AllToAll 和 P2P，先 20 轮，再 100/1000 轮；默认 `RCCL_CCLD_SLOTS=16`，另测容量参数变化。

通过条件：数据正确；TraceID 与 completion 归属一致；无伪 completion、错误回收或跨 rank 身份混合。另用极小 slot 容量主动触发观测丢失，确认 monitor 报告 `OBSERVATION_DROPPED`，无 body 的操作没有 completion，且通信成功。

## 故障与分析

合成注入覆盖参数错误、缺 rank、slow、stall、missing completion 和 P2P/AllToAll 缺边。合成注入只验证 monitor 决策，不等价于真实 GPU、驱动或网络故障。

真实训练、驱动/网络故障和性能对照作为扩展实验，必须单独记录，不能用合成注入结果代替。
