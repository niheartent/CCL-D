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
| plan 定向收口 | 每个 kernel plan 保存实际使用的 TraceID；flush host callback 只 finalize 本次 plan 集合，plan 回收释放集合 |
| Simple/LL/LL128 插桩入口 | 已按材料放置 send/recv 计数点，待专项数值验收 |
| analyzer/monitor 基础规则 | 参数、缺 rank、P2P、AllToAll、Ring/Tree 和合成 slow/stall 测试通过 |
| Python 与 record pool 单测 | 48 项 Python 测试和 record pool 单测通过 |

## 当前已测结果

plan 定向收口版本重新编译后，双节点 8 rank 测试结果：

- 构建产物：`ccld-artifacts/reinstall-gfx936-root/lib/librccl.so.1.0`；
- 100 轮 AllReduce：800/800 条 enqueue 为 `RECORDED`；12,800 条 channel completion 对应 800 个 rank/TraceID 组合；monitor 退出码 0；
- 20 轮 AllGather、ReduceScatter、Broadcast、AllToAll、P2P：默认 16 slots 下 960 条 enqueue 中 865 条 `RECORDED`、95 条 `RESOURCE_EXHAUSTED`；通信校验通过；
- 同一五算子回归改用 `RCCL_CCLD_SLOTS=64`：960/960 条 enqueue 为 `RECORDED`；4,240 条 completion 对应 880 个 rank/TraceID 组合；monitor 退出码 0。

plan 定向收口代码已构建并运行；五算子回归显示可配置容量能消除该混合负载下的观测丢失，但默认 16 slots 的容量与回收压力仍需验收。旧记录中“20 轮五算子默认容量无耗尽”无法在当前库复现，已移除。已修复的 batch 身份串用与 legacy 伪 completion 是独立问题，不意味着所有资源耗尽均已消除。

## 尚需验收的内容

1. 在真实跨多个 RCCL plan 的同一用户操作上验收定向 callback：同一 TraceID 的多 channel body 只收口一次，plan 之间不串 TraceID。
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
