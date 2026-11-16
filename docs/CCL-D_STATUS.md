# CCL-D 当前状态

本文只记录已经由代码或测试证明的事实。参考材料没有规定的实现选择，标记为 CCL-D 的工程策略。

## 已完成并验证

| 能力 | 证据 |
|---|---|
| gfx936 RCCL 独立构建和安装 | `ccld-artifacts/full-gfx936/`、独立安装前缀 |
| 双节点 8 rank 通信 | HPDPS-6/7，共享 `/workspace`，SSH 2234 |
| AllGather、ReduceScatter、Broadcast、AllToAll、P2P 数据正确性 | 五算子双节点回归通过 |
| TraceID 传递 | host enqueue、collective/P2P work 和 device body 使用同一 counter |
| AllToAll 父操作关联 | slot 不足时子 Send/Recv 不再继承其他操作的 batch 身份 |
| pinned body 观测 | channel send/recv、peer、算法、协议和时间戳输出 |
| LRU 容量配置 | `RCCL_CCLD_SLOTS` 运行时配置，无需重编译 |
| 采样与统计模式 | `RCCL_CCLD_SAMPLE_US`、`RCCL_CCLD_MODE` 运行时配置 |
| 资源耗尽语义 | 无 body 的 work 不阻塞通信、不写伪 completion，monitor 可报告观测丢失 |
| Simple/LL/LL128 插桩入口 | 已按材料放置 send/recv 计数点，待专项数值验收 |
| analyzer/monitor 基础规则 | 参数、缺 rank、P2P、AllToAll、Ring/Tree 和合成 slow/stall 测试通过 |
| Python 与 record pool 单测 | 48 项 Python 测试和 record pool 单测通过 |

## 当前已测结果

2026-11-16，双节点 8 rank、五算子、默认 16 slots：

- 20 轮：960/960 条 enqueue 为 `RECORDED`，无 `RESOURCE_EXHAUSTED`；
- 100 轮：4800/4800 条 enqueue 为 `RECORDED`，无 `RESOURCE_EXHAUSTED`；
- 100 轮测试中 10,000 条 completion 均来自 CCL-D body 路径，没有 legacy 伪 completion；
- 通信数据和 monitor 检查通过。

此前出现的资源耗尽来自两个已修复的实现错误：无 slot 的 work 被合并到前一个诊断 batch，以及旧 collTrace 路径重复输出无 body completion。不能再把旧测试结果当作当前版本结论。

## 尚需验收的内容

1. 跨多个 RCCL plan 的同一用户操作，验证 TraceID/body 在真实拆分场景下只收口一次。
2. Simple、LL、LL128 在已知消息规模下的逐协议计数数值核对，分别验证 mode 1 和 mode 2。
3. Tree 算法真实双节点长期回归，以及 Ring/Tree 首错 rank 的 slow/hang 判定。
4. 多 communicator、多 stream、HIP graph 和 communicator 销毁过程中的 body 回收。
5. 1000 轮及训练任务中的性能、pinned memory 和诊断线程开销。

## 运行时参数

```bash
RCCL_CCLD_SLOTS=16       # 每 channel 的 body 数
RCCL_CCLD_SAMPLE_US=200  # 采样周期，微秒
RCCL_CCLD_MODE=1         # 0 关闭，1 tid0 统计，2 所有 tid 原子统计
```

所有 rank 必须使用相同值。参数变化不需要重新编译；mode 2 会增加 device 原子操作开销。
