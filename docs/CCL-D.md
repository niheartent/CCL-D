# CCL-D 复现依据

本文以仓库内脱敏参考材料为依据，只把材料明确写出的内容称为“材料设计”。材料没有定义的实现选择单独标为“本项目实现”或“未确定”，不据此反推材料原文。

## 材料明确的诊断链路

材料目标是从集合通信中发现并定位 slow/hang。每条记录关联通信域和操作身份，材料给出的 TraceID 形式为 `commId + counter + 扩展位`。RCCL 在 pinned memory 中维护诊断 buffer：header 包含 `kernelIndex`、`mode`、`counter`、`detectChannel`、`traceid`；body 按 channel 保存 send/recv 计数。

device 侧按协议记录通信计数：Simple 的 send/recv 计数点均在 generic operation 末尾；LL/LL128 的 send 计数点在 waitSend 之后，recv 计数点在 generic operation 末尾。诊断线程周期性读取 buffer 并计算 sendRate/recvRate；kernel host callback 在 kernel 结束时通知诊断线程，输出操作级结果。材料描述 mode 0 为关闭、mode 1 为每 channel 的 tid0 计数、mode 2 为每 channel 所有 tid 原子计数。通信域创建/销毁与诊断线程资源绑定。

分析比较方式：Ring 在同一轮次比较各 rank；Tree 比较同一层 rank。材料提到 slow/hang 诊断，但当前文档稿没有给出完整案例清单或可直接复刻的决策树规则。材料对采样精度也有 200 微秒和 1 毫秒两处不同描述，因此本项目不能据此声称复刻了唯一的精度值。

## 本仓库实现状态

代码已经具备 RCCL host/device 记录路径、Simple/LL/LL128 计数插点、按 channel 的 pinned-memory 诊断区、诊断线程采样/完成快照，以及 JSONL 输出和 Python 分析器。它证明了数据采集链路已接通；协议计数数值与材料语义仍需逐协议验收，分析器也不能据此称为完整复刻材料中的决策树。

历史双节点回归曾验证 AllReduce 与 AllGather、ReduceScatter、Broadcast、AllToAll、P2P 的通信数据正确性及记录输出。具体环境、提交和原始结果应以 [状态记录](CCL-D_STATUS.md) 指向的实际产物为准；这些回归不等于 slow/hang 定位能力已被证明。

## 材料未规定的内容

材料没有规定 slot 数量及耗尽时的行为、RecordPool 回收算法、多 plan/work 聚合方法、AllToAll 父子身份、peer 字节分摊、故障注入接口，或完整的 slow/hang 决策规则。它们如继续保留，均属于本项目工程实现或扩展，必须有独立代码与实验依据，不得作为复现材料要求。当前状态与最小验证目标见 [CCL-D_STATUS.md](CCL-D_STATUS.md) 和 [CCL-D_TEST_PLAN.md](CCL-D_TEST_PLAN.md)。
