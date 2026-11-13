# CCL-D 后续实现路线

本文只保留当前核心复现必须完成的工作。已经通过的数据面、P2P 关联和基础分析器测试不再列为待办；没有运行证据的项目不标记为完成。

## 当前核心缺口

### 1. 操作级归属闭环

把一次用户 collective 的所有 channel、work、batch 和 plan 绑定到同一个 `TraceID` 与 `kernelIndex`。验收条件：跨多个 plan 的操作仍只有一个操作记录；AllToAll 展开的 P2P 保留父操作；多个 communicator 不串记录。

### 2. 完成通知与回收闭环

把 RCCL 的 plan/work 完成边界接到诊断记录。诊断线程负责采样和输出，不能单独推断整个操作已经结束。验收条件：所有预期 body 消费后才回收；callback 缺失进入 `MISSING_COMPLETION`；资源不足进入 `RESOURCE_EXHAUSTED`/`OBSERVATION_DROPPED`；通信不因诊断资源不足阻塞。

### 3. 协议计数语义

按材料逐项核对并实测 Simple、LL、LL128 的 send/recv 插入点、channel 归属和计数结果。验收条件：协议计数与已知传输规模一致，且 mode 1/2 的统计差异可解释。

### 4. 真实诊断验收

在完成上述两条运行时闭环后，重新编译 RCCL，执行单节点和双节点的 20/100/1000 轮测试，覆盖五类 collective 与 P2P，并验证 slow、stall、missing completion、资源耗尽和参数错误。每项必须保存原始 JSONL、monitor 输出、退出码和构建前缀。

## 暂不纳入核心验收

Graph、CollNet、NVLS、MSCCL、真实网络设备故障以及训练性能对照属于扩展实验；在核心闭环完成前不把它们写成完成度指标，也不阻塞基本复现。

## 顺序

1. 先用状态机和合成记录补齐操作归属、跨 plan 聚合和异常优先级测试。
2. 修改 RCCL work/plan 结束通知及回收路径。
3. 按协议逐路径核对计数并重新构建 gfx936 RCCL。
4. 运行单节点、双节点、长轮次和故障注入验收。
