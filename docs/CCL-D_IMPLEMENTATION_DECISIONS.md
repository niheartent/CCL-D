# CCL-D 实现语义决策

设计材料没有明确规定 body 耗尽策略和多 channel 操作的最终完成判据。本项目采用以下语义，作为实现和验收契约，不表述为材料原文。

## 资源耗尽

body 用尽时显式报告 `RESOURCE_EXHAUSTED`；已分配操作但个别 work/channel 无法记录时报告 `OBSERVATION_DROPPED`。两种情况都不阻塞通信、不覆盖活动记录、不自动扩容，也不归类为通信 hang。

## 操作结束

同一 TraceID 的所有预期 work/channel/plan 完成并消费后，才输出最终操作结果并回收 body。单个 kernel/channel 完成只更新局部状态。已分配记录缺少预期完成通知时标记 `MISSING_COMPLETION`；comm 销毁时清理并显式标记未完成记录。

## 采样与输出

运行中按 channel 采样 send/recv 进度；操作结束时按 TraceID 汇总。采样周期保持可配置，需通过检测延迟和运行开销实验确定合适值，不预设为已验收参数。

## 当前实现边界

host 分配 `kernelIndex`、planner 登记 body、device 记录 channel 数据、诊断线程观察完成并消费的路径已经接入。完整 plan/work 完成通知、跨 plan 操作收口及协议计数验证尚未完成。后续验收条件见 [`CCL-D_SLOT_LIFECYCLE_REVIEW.md`](CCL-D_SLOT_LIFECYCLE_REVIEW.md)。
