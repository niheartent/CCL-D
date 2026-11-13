# CCL-D 状态总览

本文按“代码存在”和“运行语义已验收”区分状态。

## 已验收

| 项目 | 证据 |
|---|---|
| gfx936 独立 RCCL 构建与安装 | `ccld-artifacts/full-gfx936/` |
| 双节点 8 rank 启动与共享存储 | HPDPS-6/7，SSH 2234，共享 `/workspace` |
| AllGather、ReduceScatter、Broadcast、AllToAll、P2P 数据正确性 | 多算子 slot 回归通过 |
| 参数、缺失 rank、P2P 配对和合成 slow/stall 分析 | analyzer/monitor 单测与故障矩阵通过 |
| grouped P2P 与 AllToAll 父操作关联 | 双节点回放验证，无 P2P 未关联告警 |
| Ring 结构检查 | 邻居闭合、partial channel pending 和误报回归已覆盖 |
| Python analyzer/monitor | 47 项单测通过 |

## 已接入但尚未完成运行时验收

| 项目 | 当前状态 | 验收缺口 |
|---|---|---|
| 操作级 TraceID 与 `kernelIndex` | host 已分配并传入 collective/P2P work | 跨多个 plan/work/channel 的唯一归属尚未证明 |
| pinned body 与 slot 回收 | planner 登记 body，诊断线程消费后释放 | callback 缺失、销毁清理和长轮次容量尚未验收 |
| AllToAll 父操作 | 已接入单轮路径 | 多 plan、长轮次仍需验证 |
| Ring/Tree 分析 | Ring 基础规则已验收，Tree 有规则 | 真实 Tree 多节点长期回归未完成 |

## 尚未完成的核心实现

1. **完成通知**：将 RCCL plan/work 的真实完成边界连接到诊断记录，区分 channel 完成和整个 TraceID 完成。
2. **跨 plan 聚合**：同一用户操作的所有 body 必须在最终收口后再回收，不能由单个 kernel 完成事件提前释放。
3. **协议计数**：按材料逐条核对 Simple、LL、LL128 的计数位置和 mode 语义，并用实际传输规模验证。
4. **真实 slow/hang 验收**：当前故障注入主要改变诊断证据，不等价于 GPU kernel、驱动或网络故障。
5. **长期资源验收**：100/1000 轮、多 communicator/stream 下确认无资源泄漏、观测丢失和错误回收。

## 已确定的实现语义

- LRU body 用尽：记录显式丢失，通信继续；输出 `RESOURCE_EXHAUSTED` 或 `OBSERVATION_DROPPED`。
- 多 channel/work/plan：以同一 TraceID 的全部预期 body 完成为整体结束；单个 kernel 不得提前回收。
- `MISSING_COMPLETION` 只用于已经成功分配记录但未收到预期完成通知的情况。

## 当前证据边界

最近双节点 5 轮回放为 65 条 `OK`、5 条 `P2P_DEVICE_EVIDENCE`、5 条 `TOPOLOGY_PENDING`，无 Ring/Tree 结构告警。该结果证明当前样本没有立即退化，不证明材料级生命周期和协议计数已经完成。
