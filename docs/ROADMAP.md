# CCL-D 后续验证

当前双节点基线固定使用 `RCCL_CCLD_SLOTS=512`。HPDPS-6/7 的 8 rank 五算子回归已通过，10 轮、4096 个 float、2800 条 complete、0 条观测丢失，monitor 结果全部为 `OK`。

接下来完成以下验证：

1. 扩大 512 slot 的多节点负载测试，延长迭代并增加并发，记录 `OBSERVATION_DROPPED` 和每 rank 完成比例，评估容量在目标训练负载下是否足够。

LL128 边界已确认：本构建的 `CMakeLists.txt` 以 `if(0)` 禁用 `ENABLE_LL128`；即使启用，HIP kernel 生成代码也只对 `__gfx90a__` 生成 LL128，gfx936 不会生成该路径。planner 另外要求 `RCCL_TOPO_XGMI_ALL`。因此当前 gfx936 双节点的 Ring+LL128 `internal error` 属于 RCCL 构建/架构支持边界，不是 CCL-D 运行时故障；不再将它列为本环境待验收协议。
