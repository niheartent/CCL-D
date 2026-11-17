# CCL-D 当前状态

本文只记录当前源码状态。

## 已实现

- TraceID 使用通信域 `opCount`，44-byte 扩展区保持未定义。
- pinned frame、slot LRU、mode 0/1/2 和资源耗尽丢观测策略。
- grouped P2P 父 counter 关联。
- 独立 diagnose thread、kernel callback 通知和多 plan 生命周期收口。
- sample count 增量、Ring/Tree 拓扑分组、slow/hang 状态决策。
- CCL-D 与 legacy collTrace 参数和线程路径隔离。
- CCL-D 运行参数通过 RCCL 环境变量读取。

## 静态验证

- Python 单元测试：52 项通过。
- fault simulation：stall、slow、missing 三类状态通过。
- 源码 diff 检查通过。

## 尚未执行

- 新版 gfx936 RCCL 编译和链接。
- 双节点真实 RCCL 运行回归。
- Simple、LL、LL128 逐协议计数核验；gfx936 LL128 需确认 RCCL 架构支持。
- AntCCL 评测中的 kernel hang、规模不一致、计算慢、通信慢和非通信 hang 场景。
- 训练级 CPU、pinned memory、GPU 开销对照。
