# CCL-D 当前状态

实现依据是参考材料的图文定义：TraceID、pinned frame、kernel 完成通知、diagnose thread，以及 Ring 全 rank和 Tree 同层比较。44-byte 扩展区只保留空间，当前没有字段和语义。

| 部分 | 当前实现 | 验证状态 |
|---|---|---|
| TraceID/Header | `commId + counter + reserved[44]`；header 保存 KID、mode、counter、detectChannel | host 布局单测及 gfx936 双节点运行通过 |
| channel body | 每个 block 按 channel 保存 `sendCount/recvCount` 两个 64-bit 计数 | 五类算子双节点运行通过；Ring+Simple、Ring+LL 双节点运行通过 |
| 生命周期 | enqueue 保留 block；kernel callback 排队 counter；diagnose thread 按已登记 channel 采样、输出后释放；destroy 时 drain | gfx936 双节点运行通过 |
| 资源不足 | 没有可用 free block 时输出 `OBSERVATION_DROPPED`，通信继续，不覆盖 active block | analyzer 单测覆盖 |
| Ring/Tree | 输出 Ring 邻居或 Tree parent/children/depth；Ring 全 rank、Tree 同层比较 | Ring、Tree 双节点运行通过；Tree 层级由 parent/children 重建，避免使用 kernel 级 depth 误判 |
| slow/hang | 缺 rank、未完成、基于双向收发速率的异常、同节点停滞和拓扑不对称规则；slow 输出计算慢/通信慢/混合慢分类，不以单次完成时长差异直接判定 slow | 51 项单测；既有双节点 slow/missing/stall 矩阵仍需用独立化新库重跑 |

`RCCL_CCLD_SLOTS` 是运行时参数，默认值只用于初始化，可通过环境变量调整；代码不依赖固定 16 slot。通信观测只使用材料定义的 header、channel send/recv body 和生命周期事件，不加入 peer 字节矩阵、协议矩阵或自定义父操作状态。

gfx936 独立构建已完成并安装到 `rccl-artifacts/ccld-frame-gfx936/install/`；最终链接日志只有 dcc 未使用参数警告。4-rank gfx936 单节点 20 轮回归已通过 AllGather、ReduceScatter、Broadcast、AllToAll、P2P 数据校验。`RCCL_CCLD_SLOTS=256` 时无观测丢失，monitor 输出 416 个 OK、4 个 enqueue skew；`RCCL_CCLD_SLOTS=2` 时通信仍通过，923 个操作记录为 `OBSERVATION_DROPPED`，符合容量不足策略。

双节点验收已完成：HPDPS-6/7 各 4 rank（共 8 rank），AllGather、ReduceScatter、Broadcast、AllToAll、P2P 全路径回归通过。`CCLD_ITERS=10 CCLD_COUNT=4096 RCCL_CCLD_SLOTS=512` 时 2,800 条 complete、2,705 条 sample，8 个 rank 均有记录，`OBSERVATION_DROPPED=0`，monitor 返回 0。`RCCL_CCLD_SLOTS=16` 的同一多节点回归中通信校验仍通过，但出现 197 条 `OBSERVATION_DROPPED`；增大运行时容量后丢失消失。

当前环境延长压力回归已通过：`CCLD_ITERS=100 CCLD_COUNT=4096 RCCL_CCLD_SLOTS=512`，8 rank 每 rank 完成 3,500 条 channel complete 记录（合计 28,000），覆盖五类算子且通信数据检查通过；8 个 rank 均有记录，`OBSERVATION_DROPPED=0`，monitor 2,900 条结果全部 `OK`。该程序每个算子后同步，属于长迭代串行压力，不覆盖同一 group 内多个 work 并发。产物：`rccl-artifacts/ccld-frame-gfx936/multinode-stress-slots512-20261117T055324Z/`。

对 slots512 的 290 条 monitor 结果重新分析后全部为 `OK`，此前由单次完成时长差异触发的 `COMM_SLOW` 已移除；Ring 中合法的单向收发角色不再被误判为慢 rank。

验收产物：`rccl-artifacts/ccld-frame-gfx936/multinode-collectives-slots512-20261117T034442Z/`。

当前保持 `RCCL_CCLD_SLOTS=512`。它是运行时容量参数，不需要重新编译；双节点五算子回归在该值下无观测丢失。双节点 8 rank 故障注入由 `tools/tests/run_multinode_ccld_faults.sh` 驱动：slow 在 rank 2 前插入 2 秒 stream 延迟，monitor 实时报告 `COMM_STALLED / RANK_PROGRESS_STALLED / slow_ranks=[2]`，任务最终校验通过；missing 让 rank 2 不进入 collective，monitor 报告 `MISSING_RANK` 并指出 rank 2；stall 在 rank 2 的 stream 上等待 30 秒，monitor 定位 rank 2，外层 8 秒超时终止任务。脚本退出码与预期一致。

专项实测：Ring+LL 在 8 rank 双节点通过，complete 记录协议枚举为 `0`；Tree+Simple 在 8 rank 双节点通过，complete 记录为算法 `0`、协议 `2`，无观测丢失。LL128 边界已从源码确认：当前构建用 `if(0)` 禁用 `ENABLE_LL128`；HIP kernel 生成代码即使启用也限定 `__gfx90a__`，planner 还要求 `RCCL_TOPO_XGMI_ALL`。因此 gfx936 双节点 Ring+LL128 的 `internal error` 是 RCCL 的架构/构建支持边界，不能作为 CCL-D 故障判定。

双节点故障矩阵产物：`rccl-artifacts/ccld-frame-gfx936/multinode-faults-finalverified-20261117T051732Z/`。monitor 现在会将新到达的 rank 进度结果纳入去重签名，避免先前的状态遮住后续精确定位结果。
