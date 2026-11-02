# CCL-D 下一阶段目标

本文把当前实现缺口转换为可重复、可验收的开发目标。每个目标完成时，必须同时提交实现、测试和实际运行结果。

## 已完成基线

- gfx936 独立 RCCL/CCL-D 安装，和系统 RCCL 分离。
- UCX 1.16.0 在 HPDPS-6/7 独立安装，旧版 /opt/ucx 可回滚。
- 8 rank（每节点 4 rank）跨节点 AllReduce 正确性和 monitor 正常路径通过。
- 远端 rank 的 op_mismatch 故障注入能够报告 MISMATCH。
- Python analyzer/monitor 单元测试通过。
- 标准化 8 rank 验收脚本已通过 UCX 1.16.0 回归，远端 hostname 记录已改为 gethostname()。
- smoke 支持 `CCLD_ITERS`/`CCLD_COUNT` 多轮回归；20 轮、4096 元素、8 rank 跨节点测试通过。修正多节点未知算法下 device 计数差异的误报，改为 `TOPOLOGY_PENDING`，同时保留 enqueue 参数的严格 mismatch 检测。
- 新增 `tools/tests/run_multinode_fault_matrix.sh`；远端 rank 7 第 10 轮的 `op_mismatch`、`size_mismatch`、`skip_record`、`missing_rank` 矩阵全部通过，分别验证 `MISMATCH` 或 `MISSING_RANK`。

## P0：验收入口和正确性矩阵

新增标准化验收脚本，统一记录 UCX、MPI、RCCL、CCL-D 实际路径和 monitor 结果。补齐 rank 数 2/4/8、单节点与双节点、AllReduce/Broadcast/Reduce/AllGather/ReduceScatter，以及 1KB/1MB/256MB 数据规模。每个组合要求结果正确、所有 rank 有 enqueue/complete、monitor 为 OK。

## P1：故障注入覆盖

覆盖 op_mismatch、size_mismatch、skip_record、missing_rank、stall、slow、missing_completion；分别注入本节点和远端 rank，并覆盖首个/中间/最后一个 collective。明确哪些是诊断日志模拟，哪些是真实 RCCL/GPU 错误。

## P1：稳定计数和拓扑元数据

把 enqueue counter 与稳定 trace slot/opCount 关联；completion 必须带 algorithm、protocol、channel、peer/edge、start/end。为 Ring/Tree/LL/LL128/Simple 建立拓扑规则，消除不必要的 TOPOLOGY_PENDING。

## P1：长稳和资源生命周期

实现 1,000/10,000 次混合 collective、30 分钟持续测试和多 communicator 循环。验收 counter 不重复、communicator 地址复用后计数重置、无 rank hang、无损坏 JSONL、无 GPU/UCX/MPI 资源泄漏。

## P2：速率和跨节点时间

在同一 collective 内计算窗口化 sendRate/recvRate；跨节点只比较逻辑轮次和相对顺序。若需要绝对延迟，先实现时钟校准并记录误差，禁止直接比较不同节点 monotonic timestamp。

## P2：RCCL 路径覆盖

补充 graph、CollNet、NVLS、MSCCL、多 stream、多 communicator 和不同网络插件的插桩及测试；每条路径必须验证 enqueue、kernel completion 和 monitor 结果。

## P2：真实故障边界

在容器内增加可控的进程终止、GPU kernel 超时、网络接口隔离和 MPI daemon 故障实验。真实故障测试必须有超时、清理和独立日志目录，不能使用生产节点默认环境。

## 每次提交的最低证据

记录 Git commit、节点/容器、UCX/RCCL/MPI 实际加载路径、完整启动命令、退出码、monitor 输出和未解决问题。任何性能结论必须同时给出重复次数和 P50/P95/P99。
