# CCL-D 实现评估

本项目以脱敏参考材料中的数据链路为实现依据。图示明确了四个层次：通信域提供 rank/channel/path 信息，操作层提供 collective、算法和协议，pinned memory 保存按 channel 的 send/recv body，diagnose thread 周期读取并由 kernel hostcallback 通知完成；分析器再按 Ring 全 rank、Tree 同层 rank 比较。

## 当前代码

| 项目 | 现状 | 结论 |
|---|---|---|
| TraceID | host counter 与 commHash 组合为记录身份 | 基础身份可用；图示 44-byte 扩展位尚未落地，不能称为完整布局 |
| pinned body | 按 channel/slot 保存 sendCount、recvCount，host 线程采样和完成快照 | 采集路径存在，协议数值尚未逐协议验收 |
| 生命周期 | RCCL communicator 绑定诊断线程；kernel 完成后回调；RecordPool 只保留 body 注册/消费 | 已去掉未接生产调用的 plan/channel 状态机；容量策略仍是工程行为 |
| Ring/Tree | JSONL 输出 ring 邻居、tree 层级，analyzer 有基础比较 | 真实拓扑诊断尚未验收 |
| slow/hang | 有参数/缺 rank/完成/速率规则及合成输入测试 | 尚未证明图片所示决策链在真实通信中正确定位根因 |

## 明确删除的错误扩展

不再把固定 slot 数、资源耗尽行为、跨 plan 收口算法、AllToAll 父子关联、peer 字节分摊和故障注入当作材料要求。它们只有在有实际诊断收益且不影响通信时才作为独立扩展；当前不继续扩大。

## 重做优先级

1. 先补齐 header/body 的真实字段映射：`kernelIndex`、`mode`、`counter`、`detectChannel`、TraceID 扩展位，并确认 device/host 对同一 body 的读写顺序。
2. 再按图示实现 slow/hang 最小决策：未进入、部分 rank 未完成、全体停滞三类；输入只使用已有 host/kernel 状态、开始结束时间和 send/recv rate。
3. 最后做 Simple、LL、LL128 数值核对和 Ring/Tree 双节点实验。

当前没有有效的新编译产物；旧构建目录已清理。下一次编译应在上述字段映射和决策实现完成后集中进行。
