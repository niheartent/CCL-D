# CCL-D 当前状态

实现依据是参考材料的图文定义：TraceID、pinned frame、kernel 完成通知、diagnose thread，以及 Ring 全 rank和 Tree 同层比较。44-byte 扩展区只保留空间，当前没有字段和语义。

| 部分 | 当前实现 | 验证状态 |
|---|---|---|
| TraceID/Header | `commId + counter + reserved[44]`；header 保存 KID、mode、counter、detectChannel | host 布局单测及 gfx936 双节点运行通过 |
| channel body | 每个 block 按 channel 保存 `sendCount/recvCount` 两个 64-bit 计数 | 五类算子双节点运行通过；协议分组专项验收待补 |
| 生命周期 | enqueue 保留 block；kernel callback 排队 counter；diagnose thread 按已登记 channel 采样、输出后释放；destroy 时 drain | gfx936 双节点运行通过 |
| 资源不足 | 没有可用 free block 时输出 `OBSERVATION_DROPPED`，通信继续，不覆盖 active block | analyzer 单测覆盖 |
| Ring/Tree | 输出 Ring 邻居或 Tree parent/children/depth；Ring 全 rank、Tree 同层比较 | Ring 双节点真实拓扑及合成规则通过；Tree 专项拓扑验收待补 |
| slow/hang | 缺 rank、未完成、基于双向收发速率的异常、同节点停滞和拓扑不对称规则；不以单次完成时长差异直接判定 slow | 合成规则通过；双节点受控故障注入待补 |

`RCCL_CCLD_SLOTS` 是运行时参数，默认值只用于初始化，可通过环境变量调整；代码不依赖固定 16 slot。通信观测只使用材料定义的 header、channel send/recv body 和生命周期事件，不加入 peer 字节矩阵、协议矩阵或自定义父操作状态。

gfx936 独立构建已完成并安装到 `rccl-artifacts/ccld-frame-gfx936/install/`；最终链接日志只有 dcc 未使用参数警告。4-rank gfx936 单节点 20 轮回归已通过 AllGather、ReduceScatter、Broadcast、AllToAll、P2P 数据校验。`RCCL_CCLD_SLOTS=256` 时无观测丢失，monitor 输出 416 个 OK、4 个 enqueue skew；`RCCL_CCLD_SLOTS=2` 时通信仍通过，923 个操作记录为 `OBSERVATION_DROPPED`，符合容量不足策略。

双节点验收已完成：HPDPS-6/7 各 4 rank（共 8 rank），AllGather、ReduceScatter、Broadcast、AllToAll、P2P 全路径回归通过。`CCLD_ITERS=10 CCLD_COUNT=4096 RCCL_CCLD_SLOTS=512` 时 2,800 条 complete、2,705 条 sample，8 个 rank 均有记录，`OBSERVATION_DROPPED=0`，monitor 返回 0。`RCCL_CCLD_SLOTS=16` 的同一多节点回归中通信校验仍通过，但出现 197 条 `OBSERVATION_DROPPED`；增大运行时容量后丢失消失。

对 slots512 的 290 条 monitor 结果重新分析后全部为 `OK`，此前由单次完成时长差异触发的 `COMM_SLOW` 已移除；Ring 中合法的单向收发角色不再被误判为慢 rank。

验收产物：`rccl-artifacts/ccld-frame-gfx936/multinode-collectives-slots512-20261117T034442Z/`。

后续只保留三项验证工作：

1. 在 HPDPS-6/7 的 8 rank 任务中，通过脚本模拟 missing、stall、slow，确认 monitor 分别定位未进入 rank、停滞 rank 和低速 rank。
2. 固定双节点拓扑，分别约束 Simple、LL、LL128，核对每种协议的 channel 收发计数和回收结果。
3. 固定 Tree 算法运行双节点算子，核对 parent/children/depth 的同层比较；Ring 已由当前回归覆盖。

当前保持 `RCCL_CCLD_SLOTS=512`。它是运行时容量参数，不需要重新编译；本轮双节点 10 轮五算子回归在该值下无观测丢失。host 故障矩阵（8 rank）已验证 `stall -> MISSING_COMPLETION`、`slow -> COMM_SLOW`、`missing -> MISSING_COMPLETION`；这些是分析器链路测试，仍需补充双节点任务中的可控注入版本。
