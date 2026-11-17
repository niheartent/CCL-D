# CCL-D 后续验证

当前双节点基线固定使用 `RCCL_CCLD_SLOTS=512`。HPDPS-6/7 的 8 rank 五算子回归已通过，10 轮、4096 个 float、2800 条 complete、0 条观测丢失，monitor 结果全部为 `OK`。

按优先级完成以下验证：

1. 在双节点运行 missing、单 rank stall、单 rank slow 的可控注入，检查真实 RCCL JSONL 到 monitor 的端到端定位；host 生成 JSONL 的故障矩阵已有覆盖，不能替代此项。
2. 分别触发 Simple、LL、LL128，核对实际 `protocol` 字段、按 channel 的计数及 block 回收。当前多节点回归实测到 Simple（协议枚举 2），LL 和 LL128 尚未覆盖。
3. 约束 Tree 算法进行双节点回归，检查真实 parent/children/depth 与按层分析。当前真实多节点回归实测到 Ring（算法枚举 1），Ring 拓扑规则已覆盖。
4. 在固定 512 slot 下延长多节点迭代并增加操作并发，记录 `OBSERVATION_DROPPED`、队列积压和每 rank 的完成比例，确认该容量适合目标负载。
