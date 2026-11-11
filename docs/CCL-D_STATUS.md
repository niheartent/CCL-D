# CCL-D 状态总览

本文只记录已验证事实、材料明确要求和当前未验证缺口。没有实测或没有在设计材料中明确的内容，不标记为完成。

## 已验证事实

| 项目 | 状态 | 证据 |
|---|---|---|
| gfx936 独立 RCCL 构建目录 | 已完成 | `ccld-artifacts/sample-probe-build` |
| gfx936 独立 RCCL 安装前缀 | 已完成上一版安装 | `ccld-artifacts/sample-probe-install`；最近增量构建曾完成但未在 slot 生命周期修正后重新安装 |
| 双节点 SSH/共享存储 | 已完成 | HPDPS-6/7、SSH 2234、共享 `/workspace` |
| UCX/MPI 双节点启动 | 已验证基础路径 | Open MPI 5.0.3，双节点 8 rank 启动成功 |
| 五类数据面回归 | 已验证 | AllGather、ReduceScatter、Broadcast、AllToAll、P2P 数据结果通过 |
| 8 rank、20 轮、count 4096 | 数据面通过 | `slot_regression PASS` |
| Python analyzer/monitor 单测 | 已完成 | 最近一次 39 项通过 |
| 参数/缺失 rank 合成注入 | 已验证 | `MISMATCH`、`MISSING_RANK` |
| slow/stall/missing completion 合成日志 | 已验证分析器路径 | 不等价于真实 GPU/网络故障 |
| host 分配 `kernelIndex` 并传入 collective/P2P work | 原型已接入并通过窄函数集编译 | 目前尚未完成跨 plan 聚合、最终回收和完整运行验收 |

## 设计材料明确要求但当前没有完成的部分

| 设计要求 | 当前状态 | 不能默认完成的原因 |
|---|---|---|
| `commId + counter + 扩展位` 作为操作级 TraceID | 部分实现 | host 已分配独立 `ccldCounter`，并贯穿 collective/P2P work 与 device diag；跨 plan 聚合和最终回收仍在接入 |
| `kernelIndex` LRU 物理索引 | 部分实现 | host RecordPool 已实现不覆盖活动 body 的状态机并分配物理索引；尚未完成 planner 级多 plan 注册和最终回收接线 |
| pinned memory body 按操作/channel 保存计数 | 部分实现 | pinned 区域和 channel 计数已存在，device 不再使用 `opCount % slot`；body 生命周期仍需由 plan/work 完成事件收口 |
| diagnostic thread 周期采样 | 部分实现 | 有轮询 sample，当前默认 10 ms；材料正文给出默认约 200 微秒，示例又使用 1 ms 精度，必须通过实测确定本实现的有效口径 |
| kernel host callback 通知结束 | 部分实现 | 诊断线程已能消费 complete body；planner/reclaimer 的 operation-level callback 聚合尚未完成 |
| 多 work/batch 归属同一操作 TraceID | 未完成 | 当前 kernel 侧不能证明 batch 内所有 work 都正确归入同一操作记录 |
| AllToAll 保留父操作身份 | 未完成 | RCCL 当前路径展开为 P2P Send/Recv，尚未记录 parent operation |
| Simple/LL/LL128 精确计数点 | 未完成 | 材料要求不同 primitive 位置，当前统一计数宏尚未逐路径对照验证 |
| Ring 全局比较 | 部分实现 | 规则已写入 analyzer，但真实数据完整性和长期回归未完成 |
| Tree 同层比较 | 部分实现 | 规则已写入 analyzer，但真实多节点 Tree 回归未完成 |
| 决策树根因归类 | 部分实现 | 目前是规则集合，不是材料中的完整 slow/hang 决策树 |
| 真实训练 slow/hang | 未完成 | 当前主要是诊断记录注入和合成日志 |
| 性能、CPU、pinned memory、GPU 开销 | 未完成 | 没有按材料口径完成对照实验 |

## 已确定的实现决策

- LRU body 用尽时不阻塞通信、不覆盖活动记录、不自动扩容；输出 `RESOURCE_EXHAUSTED` 或 `OBSERVATION_DROPPED`。
- 多 channel 操作以同一 TraceID 的全部 plan/work/channel 完成为整体结束；一个 RCCL plan callback 不能单独结束跨 plan 操作。
- 运行中按 channel 输出 sample，最终按 TraceID 收口；body 消费后立即回收，资源耗尽仅形成显式观测缺口。
- 这两项是本工程的实现决策，不是材料中明确给出的原文规则。详细理由见 [`CCL-D_IMPLEMENTATION_DECISIONS.md`](CCL-D_IMPLEMENTATION_DECISIONS.md)。host 侧独立状态机原型已通过单测，尚未接入 RCCL planner/work。

## 当前明确问题

## 2026-11-11 完整 gfx936 构建与双节点回归

- 完整 Release 构建成功：`ccld-artifacts/full-gfx936/build`。
- 独立安装成功：`ccld-artifacts/full-gfx936/install/lib/librccl.so.1.0`。
- 全量 device LTO/link 用时约 1809 秒（约 30 分钟）；窄函数构建仍用于快速编译检查。
- 8 rank、HPDPS-6/HPDPS-7、AllGather/ReduceScatter/Broadcast/AllToAll/P2P 数据面测试通过。
- host record pool 单测通过，Python analyzer 39 项通过。
- 新鲜双节点诊断日志仍出现 `RESOURCE_EXHAUSTED`：2 轮测试记录 208 条资源耗尽、1892 条 complete。该结果符合“观测资源耗尽时通信继续”的安全策略，但说明 operation-level body 回收尚未接入，不能作为诊断完整验收通过。
- 后续必须接入 plan/work/channel 的完成聚合和最终回收，再重复同一回归确认资源耗尽归零或仅在明确超载场景出现。

上一轮五算子数据面回归中观察到诊断记录丢失。随后加入了 `dropped_batches` 记录，但这只能把观测丢失显式化，不能证明生命周期正确。问题根因尚未完成材料级修复：当前实现仍把物理 slot 和操作身份混用，且没有完成 `kernelIndex` LRU、callback 回收和多 work 归属。

源码审计还确认：`rcclCcldRecordEnqueue()` 目前在 `ncclEnqueueCheck()` 的 `taskAppend()` 之前执行，而 `comm->opCount` 的推进发生在 proxy 提交路径。因而不能直接把当前 enqueue 日志中的 `comm->opCount` 当作已经稳定分配的材料级操作 counter；分组调用和 plan 内多个 task 的归属必须先在 host planner 侧明确。

因此，当前不能把 `dropped_batches=0`、扩大 slot 数量或一次回归 PASS 当作完整诊断实现的证明。

## 当前完成度口径

不再用单一百分比表示完成度。按材料验收分层：

- **数据面回归层**：五类算子已通过一次 8 rank、20 轮正确性测试。
- **诊断记录层**：host 已开始分配并传入 `kernelIndex`，device 已停止用 `opCount % slot` 选择索引；跨 plan 聚合、最终回收和完整运行验收仍未完成。
- **分析器层**：已有基础参数、缺失、速率、P2P 边和 Ring/Tree 规则，但不是完整决策树。
- **论文级验收层**：未完成，尤其缺 LRU/callback、多 work 归属、协议精确计数、训练任务和性能数据。

## 下一步必须先完成的验证

1. 按 `ncclEnqueueCheck → taskAppend → plan → uploadProxyOps → kernel → callback` 画出实际生命周期，并确定操作 counter 的唯一分配点。
2. 在不编译的情况下，用合成记录验证 `TraceID`、`kernelIndex`、body/channel、回收状态机。
3. 明确 AllToAll/P2P parent operation 和多 work batch 归属字段，同时确认 FIFO/graph/persistent work 结构预算。
4. 明确 Simple/LL/LL128 的真实计数插入点和 mode 语义。
5. 通过状态机单测后，统一修改 host work 元数据、device body 和结束通知，再编译 RCCL。
6. 新库安装后重新跑干净目录的五算子回归，再测 100/1000 轮、多 communicator/stream 和长稳。
