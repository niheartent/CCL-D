# CCL-D 最小验收计划

每次实验保存源码 commit、构建前缀、环境变量、rank 映射、原始 JSONL 和 analyzer 输出。

1. **结构核对**：检查 TraceID 字段、header 字段、每个 channel 的 send/recv body、mode 0/1/2，以及 hostcallback 完成通知；不通过则不做性能实验。
2. **协议核对**：用已知消息规模分别触发 Simple、LL、LL128，核对 send/recv 数值、channel 归属和开始结束时间。
3. **决策核对**：构造真实可控的未进入、部分 rank 停滞、全体停滞和单 rank 低速场景，验证根因 rank、收发计数和 sendRate/recvRate 证据。JSON 合成数据只作为单元测试。
4. **拓扑核对**：运行 Ring 和 Tree，Ring 比较全 rank，Tree 按层比较；保存实际邻居/层级和每轮输入。
5. **开销核对**：对比关闭/开启诊断的吞吐、延迟、CPU 和 pinned memory。任何通信错误都判为失败。
