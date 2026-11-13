# CCL-D 核心验收计划

只对核心复现缺口安排测试。所有运行实验保存源码 commit、RCCL build/install 前缀、rank 映射、环境、原始 JSONL、monitor 输出和退出码。

## 1. 操作身份与生命周期

构造同一操作跨多个 channel/work/batch/plan 的用例，并覆盖显式 P2P、AllToAll 子 work 和多个 communicator。

通过条件：

- 同一用户操作只有一个 TraceID，所有相关 work 都能关联到它；
- 单个 kernel/channel 完成不触发提前回收；
- 所有预期 body 消费后恰好释放一次；
- 重复通知、缺失通知和 communicator 销毁均有确定状态；
- 资源耗尽/观测丢失显式报告且不阻塞数据通信。

## 2. 协议计数

对 Simple、LL、LL128 分别选择能够稳定触发的消息规模，核对 primitive 插入点、channel、send/recv 方向和总字节数。mode 1/2 如保留，需证明 mode 1 的低开销采样和 mode 2 精确计数语义。

通过条件：日志协议归属正确，累计计数与已知传输规模相符；不同协议不能共用未经验证的推断结果。

## 3. RCCL 运行回归

完成上述实现并重编 gfx936 RCCL 后，先运行单节点，再运行 HPDPS-6/7 双节点 8 rank。覆盖 AllReduce、AllGather、ReduceScatter、Broadcast、AllToAll 和显式 P2P；先 20 轮冒烟，再 100/1000 轮容量验收。

通过条件：数据正确；TraceID、预期 body 和 completion 数量一致；没有非预期 `RESOURCE_EXHAUSTED`、`OBSERVATION_DROPPED`、错误回收或 analyzer 误报。

## 4. 诊断规则

使用已有合成故障矩阵验证参数错误、缺失 rank、slow、stall、missing completion 和 P2P/AllToAll 缺边。合成注入只证明 monitor 规则，不算真实 GPU/网络故障验收。

核心验收仅要求可重复的诊断证据注入和数据通信不受诊断资源耗尽影响。真实训练、驱动/网络故障及性能对照列为后续扩展实验，不阻塞核心实现完成。
