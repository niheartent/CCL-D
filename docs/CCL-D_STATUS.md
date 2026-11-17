# CCL-D 当前状态

实现依据是参考材料的图文定义：TraceID、pinned frame、kernel 完成通知、diagnose thread，以及 Ring 全 rank和 Tree 同层比较。44-byte 扩展区只保留空间，当前没有字段和语义。

| 部分 | 当前实现 | 验证状态 |
|---|---|---|
| TraceID/Header | `commId + counter + reserved[44]`；header 保存 KID、mode、counter、detectChannel | host 布局单测通过；待 gfx936 运行核对 |
| channel body | 每个 block 按 channel 保存 `sendCount/recvCount` 两个 64-bit 计数 | 结构迁移完成；待 Simple/LL/LL128 运行核对 |
| 生命周期 | enqueue 保留 block；kernel callback 排队 counter；diagnose thread 按已登记 channel 采样、输出后释放；destroy 时 drain | 代码完成；gfx936 增量链接中 |
| 资源不足 | 没有可用 free block 时输出 `OBSERVATION_DROPPED`，通信继续，不覆盖 active block | analyzer 单测覆盖 |
| Ring/Tree | 输出 Ring 邻居或 Tree parent/children/depth；Ring 全 rank、Tree 同层比较 | 合成单测通过；待双节点真实拓扑验证 |
| slow/hang | 缺 rank、未完成、基于双向收发速率的异常、同节点停滞和拓扑不对称规则；不以单次完成时长差异直接判定 slow | 合成单测及双节点回归通过 |

`RCCL_CCLD_SLOTS` 是运行时参数，默认值只用于初始化，可通过环境变量调整；代码不依赖固定 16 slot。通信观测只使用材料定义的 header、channel send/recv body 和生命周期事件，不加入 peer 字节矩阵、协议矩阵或自定义父操作状态。

gfx936 独立构建已完成并安装到 `rccl-artifacts/ccld-frame-gfx936/install/`；最终链接日志只有 dcc 未使用参数警告。4-rank gfx936 单节点 20 轮回归已通过 AllGather、ReduceScatter、Broadcast、AllToAll、P2P 数据校验。`RCCL_CCLD_SLOTS=256` 时无观测丢失，monitor 输出 416 个 OK、4 个 enqueue skew；`RCCL_CCLD_SLOTS=2` 时通信仍通过，923 个操作记录为 `OBSERVATION_DROPPED`，符合容量不足策略。

双节点验收已完成：HPDPS-6/7 各 4 rank（共 8 rank），AllGather、ReduceScatter、Broadcast、AllToAll、P2P 全路径回归通过。`CCLD_ITERS=10 CCLD_COUNT=4096 RCCL_CCLD_SLOTS=512` 时 2,800 条 complete、2,705 条 sample，8 个 rank 均有记录，`OBSERVATION_DROPPED=0`，monitor 返回 0。`RCCL_CCLD_SLOTS=16` 的同一多节点回归中通信校验仍通过，但出现 197 条 `OBSERVATION_DROPPED`；增大运行时容量后丢失消失。

对 slots512 的 290 条 monitor 结果重新分析后全部为 `OK`，此前由单次完成时长差异触发的 `COMM_SLOW` 已移除；Ring 中合法的单向收发角色不再被误判为慢 rank。

验收产物：`rccl-artifacts/ccld-frame-gfx936/multinode-collectives-slots512-20261117T034442Z/`。
