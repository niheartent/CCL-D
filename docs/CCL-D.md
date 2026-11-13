# CCL-D 复现说明

## 目标模型

系统由 RCCL 数据采集和集中分析两部分组成。每次用户 collective 以 `TraceID = commId + counter + 扩展位` 标识；`kernelIndex` 只表示 pinned-memory body 的物理槽位。同一操作的 channel、work 和 plan 需要归入同一 TraceID，分析器据此判断参数一致性、通信进度和拓扑关系。

材料描述的采集字段包括操作身份、算法/协议、channel 收发计数和完成状态。诊断线程负责周期采样；操作结束通知负责最终收口和回收。Ring 在同一 collective 内按 rank 检查邻居关系；Tree 按 channel 和层级比较。Simple、LL、LL128 的收发计数位置不同，应按各自 primitive 验证。

## 当前实现

仓库已有 host enqueue 记录、pinned-memory 诊断 body、Simple/LL/LL128 计数宏、周期采样、completion 输出、P2P/AllToAll 关联和 Python monitor。gfx936 RCCL 独立构建、双节点五类算子数据面回归以及 47 项 analyzer 单测已通过。

这证明基础链路可运行，不等于材料级语义已经完成。当前仍不能证明一个用户操作跨多个 plan/work/channel 的计数完整且只在整体结束后回收；当前结束路径依赖诊断线程观察 body 状态，尚未接入完整的 RCCL plan/work 完成通知；协议计数点也未逐路径对照材料和真实规模验收。

## 尚需完成

1. **操作归属**：为同一操作的全部 channel/work/plan 保持统一 TraceID 和 body 归属，包含 AllToAll 子 work。
2. **结束与回收**：接入真实完成通知，所有预期 body 完成后才收口回收；资源耗尽、观测丢失和未完成通信分别报告。
3. **协议语义**：分别验证 Simple、LL、LL128 的 send/recv 插入点、计数规模和统计 mode。
4. **运行验收**：重新构建后运行多节点长轮次及诊断故障测试；合成 slow/hang 不能替代真实故障证据。

状态和顺序见 [`CCL-D_STATUS.md`](CCL-D_STATUS.md) 与 [`ROADMAP.md`](ROADMAP.md)，验收条件见 [`CCL-D_TEST_PLAN.md`](CCL-D_TEST_PLAN.md)。
