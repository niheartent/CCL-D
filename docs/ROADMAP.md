# CCL-D 后续验证

当前双节点基线固定使用 `RCCL_CCLD_SLOTS=512`。HPDPS-6/7 的 8 rank 五算子回归已通过，10 轮、4096 个 float、2800 条 complete、0 条观测丢失，monitor 结果全部为 `OK`。

接下来完成以下验证：

1. 扩大 512 slot 的多节点负载测试，延长迭代并增加并发，记录 `OBSERVATION_DROPPED` 和每 rank 完成比例，评估容量在目标训练负载下是否足够。
2. 验证 LL128 的支持边界。当前 8 rank 双节点 Ring+LL128 在 RCCL 调用处返回 `internal error`，没有 complete 记录；确认是否有受支持的拓扑/参数组合后再测，不能把失败组合记作协议通过。
