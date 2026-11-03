# CCL-D 复现状态总览

本项目目标是复现所提供 CCL-D 材料中的集合通信 slow/hang 诊断能力。CCL-D 是脱敏实现分支，不能把 CCL-D 论文已有内容等同于 CCL-D 全部功能。

## 已完成并实测

| 能力 | 状态 | 证据 |
|---|---|---|
| gfx936 独立 RCCL/CCL-D 安装 | 完成 | ccld-artifacts/dtk-26.04-gfx936-try1/install |
| 双节点共享存储和 SSH | 完成 | HPDPS-6/7，SSH 2234 |
| UCX 独立升级 | 完成 | 两节点 /opt/ucx-1.16.0，旧版保留 |
| MPI 双节点通信 | 完成 | 2 rank、8 rank AllReduce |
| RCCL/CCL-D 8 rank | 完成 | 每节点 4 rank，结果 36 |
| 多轮通信 | 基础完成 | 20 轮、4096 元素通过 |
| 正常 monitor | 完成 | exit 0 |
| 远端 rank 故障检测 | 基础完成 | rank 7 第 10 轮可检出 |
| Python analyzer 单测 | 完成 | 16 项通过 |

## CCL-D 能力对照

| CCL-D 目标 | 当前状态 | 差距 |
|---|---|---|
| commId、counter、Trace ID | 部分完成 | 已有 comm_id/counter/trace_id，已加入 RCCL op_count，但尚未统一关联并完成回归 |
| host collective 开始记录 | 基础完成 | 已记录 op、count、bytes、datatype、root、rank、节点、时间 |
| kernel 开始和结束 | 第一版完成 | ncclCcldDiag 已记录 start/end、state、channel、opCount |
| 每 channel send/recv 总数 | 第一版完成 | device 累计，host 读取 |
| pinned memory 诊断通道 | 第一版完成 | 还没有完整 LRU trace slot 管理 |
| 诊断线程生命周期 | 部分完成 | 当前轮询 complete，未实现完整 callback/slot 回收模型 |
| kernel 速率离群检查 | 部分完成 | completion start/end 与 send/recv 总数可计算 rank 速率并输出 COMM_SLOW；新增可选连续窗口 COMM_SLOW_WINDOW |
| sendRate/recvRate 周期采样 | 未完成 | 没有周期采样、delta、窗口化停滞判断 |
| Ring 横向比较 | 部分完成 | 缺少 peer/edge 时只能降级为 TOPOLOGY_PENDING |
| Tree 同层比较 | 基础完成 | RCCL completion 已输出 tree_up/tree_down/tree_depth，分析器按 depth 比较同层边；仍需真实多节点 Tree 回归 |
| peer/edge 拓扑字段 | 部分完成 | completion 已输出并聚合 peer/channel 边字段，包含边级收发量和速率；Ring 对称性已实现，Tree 依赖 level 字段 |
| 算法和协议一致性 | 部分完成 | completion 有字段，路径覆盖和校验不足 |
| 计算慢识别 | 未完成 | 没有计算阶段与 collective 开始边界分析 |
| kernel/通信推进慢识别 | 部分完成 | 单轮 COMM_SLOW 与连续窗口 COMM_SLOW_WINDOW 已实现；真实 GPU/网络慢和拓扑分层仍未完成 |
| 未进入 collective 检测 | 基础完成 | enqueue 缺失可报 MISSING_RANK |
| 参数不一致检测 | 完成 | op/count/bytes/datatype/root |
| kernel 未完成检测 | 基础完成 | completion 缺失可报 MISSING_COMPLETION |
| 真实 GPU hang/abort | 未完成 | 当前是诊断侧注入和合成日志 |
| 真实网络故障 | 未完成 | 尚无 IB 隔离和 NET 错误实验 |
| 多 communicator 隔离 | 未完成验证 | 没有并发多 communicator 回归 |
| stream/graph/CollNet/NVLS/MSCCL | 未完成验证 | 没有测试矩阵 |
| 性能开销评估 | 未完成 | 没有训练 workload 和 CPU/pinned/GPU 对照 |

## 故障测试边界

真实 RCCL/CCL-D 记录矩阵已验证：op_mismatch 和 size_mismatch 报 MISMATCH，skip_record 和 missing_rank 报 MISSING_RANK。

容器内合成 completion 矩阵已验证：stall 报 MISSING_COMPLETION，slow 报 COMM_SLOW 并定位 slow rank，missing completion 报 MISSING_COMPLETION。它们验证插桩、日志聚合和 monitor 决策，不能等价于真实 GPU kernel hang、驱动错误或 IB 链路中断。

## 结论

当前是 CCL-D 诊断链路的第一版可运行复现，已具备 Trace/enqueue/completion/monitor 骨架和基础故障检测；还不是完整 CCL-D。CCL-D 已增加基于 completion start/end 和 send/recv 总数的第一版 COMM_SLOW 速率离群检查；周期 send/recv 采样、Ring/Tree 拓扑分层比较、根因决策树和真实 slow/hang 故障实验仍未完成。

## 下一阶段验收顺序

1. 完成 op_count 与 enqueue/device completion 的统一 Trace 关联，重建独立 RCCL 并完成 8 rank 回归。
2. 增加诊断线程周期采样，输出每 channel 的 sendRate、recvRate、窗口和停滞状态。
3. 导出 peer/edge、Ring/Tree 层级信息，建立 Ring 全局比较和 Tree 同层比较。
4. 扩展多 collective、stream、communicator、Graph、CollNet、NVLS、MSCCL 测试。
5. 增加计算慢、kernel 延迟、进程退出、网络隔离等可控真实故障实验。
6. 用训练 workload 评估吞吐、端到端时间、CPU、pinned memory 和 GPU 开销。

每项必须提交代码、单元测试、双节点实测日志和状态更新，不能只增加文档或合成日志。
