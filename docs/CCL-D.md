# CCL-D 复现目标

本项目以 AntCCL 的通信诊断系统为复现目标，项目对外名称为 CCL-D。复现重点是 RCCL 内的高精度通信观测与集中式分析，不把单纯的数据面正确性或 monitor 能运行视为完成。

## 目标语义

一次用户操作以 `TraceID = commId + counter + 扩展位` 标识。`kernelIndex` 是 pinned-memory body 的物理槽位。该操作涉及的所有 rank、channel、work 和 plan 都必须归入同一身份；body 保留 channel 维度的收发计数。分析器使用这些证据识别参数不一致、通信停滞/变慢和拓扑异常。

Ring 按一次 collective 的 rank/邻居关系比较；Tree 按 channel 和层级比较。Simple、LL、LL128 的计数点按协议分别核对。诊断资源不足只能造成显式观测缺失，不能阻塞通信或伪装成通信故障。

## 当前边界

已有 RCCL 插桩原型、host/device 诊断记录、采样与 completion 输出、P2P/AllToAll 关联、基础 analyzer/monitor，以及 gfx936 双节点五类算子数据面回归。操作跨 plan 的统一归属、RCCL 完成通知与最终回收、协议计数实测仍未闭环。

准确状态与验收条件分别见 [CCL-D_STATUS.md](CCL-D_STATUS.md)、[ROADMAP.md](ROADMAP.md) 和 [CCL-D_TEST_PLAN.md](CCL-D_TEST_PLAN.md)。
