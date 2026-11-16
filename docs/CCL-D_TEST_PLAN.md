# CCL-D 最小验证计划

每次实验保存 RCCL commit、构建/安装前缀、节点和 rank 映射、环境变量、原始 JSONL、monitor 输出和退出码。先验证采集，再验证分析；不要用合成记录代替真实通信结果。

## 1. 采集正确性

对固定消息规模分别触发 Simple、LL、LL128，记录操作类型、算法、协议、开始/结束时间、channel、sendCount、recvCount。mode 0 应无计数；mode 1 与 mode 2 在单一稳定操作上应得到一致总量（允许原子更新时序差异，但不能改变最终计数）。

通过条件：计数位置与材料一致，方向和 channel 不串，已知传输规模可解释；不以 peer 字节分摊或额外字段替代材料要求的 send/recv 计数。

## 2. 操作级完成

运行单节点和双节点的 AllReduce、AllGather、ReduceScatter、Broadcast、AllToAll、P2P，校验通信数据，同时检查 TraceID、开始/结束时间和 callback 结果。多 plan 或多 channel 只作为当前工程实现的回归输入，不预设材料规定了某种收口算法。

通过条件：通信正确；每条输出能追溯到一个操作和通信域；没有重复完成或把不同操作的计数混在一起。

## 3. Ring/Tree 分析

分别保存 Ring 和 Tree 的实际拓扑及各 rank 原始指标。Ring 只做同一操作的跨 rank 比较；Tree 按同一层 rank 比较。每个 slow/hang 输出必须同时给出参与比较的 rank、sendRate/recvRate、时间区间和规则结果。

通过条件：改变一个 rank 的输入指标只影响应受影响的比较结果；同层关系来自真实拓扑，而不是固定 rank 编号假设。

## 4. 可控异常与开销

使用可重复的慢速/停滞场景验证端到端检测延迟；合成 JSON 仅用于 analyzer 单测。另做关闭诊断与启用诊断的吞吐、延迟、CPU 和 pinned-memory 对照。

通过条件：异常场景通信行为和诊断输出分别记录；报告误报、漏报和检测延迟；资源容量和故障注入接口只作为本项目扩展说明，不宣称为 参考材料 原始机制。
