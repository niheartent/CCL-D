# CCL-D 操作记录生命周期验收条件

## 目标语义

- 用户操作身份：`TraceID = commId + counter + 扩展位`。
- `kernelIndex`：pinned body 的物理索引，与 TraceID 分离。
- 一个操作的多个 channel/work/plan 共用操作身份；body 按 channel 保存计数。
- 诊断线程负责采样；RCCL 完成通知负责标记完成；全部预期 body 被消费后才回收。
- 资源不足和观测丢失不能阻塞通信，也不能被报告为通信 hang。

## 当前差距

当前代码已在 host 分配 `kernelIndex` 并传入 work，planner 会登记 body；device 写入 channel body，诊断线程观察完成状态并消费记录。尚未证明跨 plan 的预期 body 归属完整，也没有把完整操作结束通知接到统一回收状态机。Simple、LL、LL128 仍共用计数宏路径，需逐项核对材料规定的位置和计数语义。

## 验收条件

1. 一个操作跨多个 channel、work、batch、plan 后仍只有一个 TraceID。
2. 任一单独 channel/kernel 完成不会提前回收操作记录。
3. 全部预期 body 完成并消费后恰好回收一次；重复或迟到通知不造成重复释放。
4. 记录资源耗尽输出 `RESOURCE_EXHAUSTED`，已分配记录缺少结束通知输出 `MISSING_COMPLETION`，观测缺口不影响 RCCL 通信。
5. 多 communicator、显式 P2P 和 AllToAll 子 work 的身份关联正确。
6. Simple、LL、LL128 的计数与已知字节规模一致。

测试步骤只维护在 [`CCL-D_TEST_PLAN.md`](CCL-D_TEST_PLAN.md)；实现顺序只维护在 [`ROADMAP.md`](ROADMAP.md)。
