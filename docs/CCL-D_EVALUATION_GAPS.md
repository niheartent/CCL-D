# Evaluation 补充实施清单

参考材料第 5 节给出的性能与资源验收口径是：开启诊断后训练耗时、模型精度和资源占用影响控制在 1% 内；示例结果为单步 4.030 s 对 4.032 s、诊断线程 CPU 约 1%、pinned memory 2208 Byte、GPU 无额外开销。下面只记录 CCL-D 当前需要补齐的可测项，不把参考值当成当前结果。

## 1. 诊断线程 CPU

**当前情况**：诊断线程已经存在并周期调用 `rcclCcldSampleDiagnostics`，但没有记录线程 CPU 时间、采样次数或 CPU 占比。它目前复用 RCCL 的 `ncclCommThreadMain`，需在 `KERNEL_COLL_TRACE` 关闭的条件下测量，避免把其他 trace 工作算进 CCL-D CPU 指标。

**实现**：

- 在线程入口和退出处使用 `clock_gettime(CLOCK_THREAD_CPUTIME_ID)`；
- 维护每个 communicator 的 `thread_cpu_ns`、`thread_wall_ns`、`sample_count`；
- 在 communicator drain 时输出一条 `resource` 记录，字段至少包括 `rank`、`slots`、`sample_interval_us`、`thread_cpu_ns`、`thread_wall_ns`、`cpu_ratio`；
- 线程退出、初始化失败和诊断关闭时都要能输出完整或明确标记为 partial 的记录。

**验收**：同一 workload 分别关闭诊断、默认采样周期、20 ms 采样周期运行，比较每 rank 的 CPU 时间和占比；检查多节点 rank 间没有异常偏离。目标是给出实际数值，再与 1% 参考口径比较。

## 2. pinned memory

**当前情况**：pinned frame 已通过 `ncclCudaHostCalloc` 分配，但没有把分配大小写入日志，也没有和进程实际锁页内存交叉核对。当前 `ENABLE_COLLTRACE` 初始化分支无条件分配该 frame，即使 `RCCL_CCLD_ENABLE=0` 仍占用 pinned memory；这会污染 off/on 资源对照，必须改成只在诊断启用时分配，并保持运行期间不可动态切换开关的约束，或明确支持安全惰性初始化。

**当前结构的基线**：`Header` 为 96 B，`MAXCHANNELS=128`，每个 `ChannelBody` 为 16 B，因此 `sizeof(ncclCcldDiag)=2144 B`；`RCCL_CCLD_SLOTS=512` 时每个 communicator 的 frame 为 1,097,728 B。该数值与材料示例 2208 B 不同，不能直接沿用材料数字。

**实现与验收**：

- 先修复 off 状态仍分配 frame 的问题，再做基线对照；
- 初始化时输出 `frame_block_bytes`、`slots`、`frame_pinned_bytes`；
- 释放时输出对应 communicator 的生命周期；
- 用 `/proc/<pid>/status` 的 `VmPin` 或等价 allocator 统计交叉核对；
- 运行 slots=16、256、512，确认总量随 slots 线性变化，且通信结果不受影响；
- 将单 communicator、单 rank 和双节点总量分开报告。

## 3. GPU 开销

**当前情况**：Simple、LL、LL128（当前 gfx936 不支持 LL128）路径内已有 send/recv 计数更新；没有开关配对的设备时间和吞吐实验，因此不能声明 GPU 额外开销为零。

**实现与验收**：

- 使用同一个测试程序和相同 rank 映射，分别设置 `RCCL_CCLD_ENABLE=0/1`；
- warmup 后用 HIP event 记录每 step 时间，至少报告平均值、P50、P95、吞吐和显存占用；
- 按算子、消息大小、协议分别对照，保存原始 CSV/JSON；
- 对计数插桩路径使用 profiler 或 kernel 时间统计，区分通信本身波动与诊断计数开销；
- 以 `(enabled-disabled)/disabled` 计算相对影响，不能以一次运行结论替代配对实验。

## 4. 同一 group 内多 work 并发

**当前情况**：RCCL planner 已将多个 plan 的 CCL-D counter 收集到统一 finalize callback，说明底层路径具备聚合基础；现有五算子压力程序每个算子后同步，尚未验证 grouped work、多 plan 和混合 P2P 的边界。代码已实现一部分，缺的是端到端压力回归和生命周期证据，不是“完全没有实现”。

**实现测试程序**：

- 在 `ncclGroupStart/End` 内提交多个 AllReduce；
- 提交混合 AllGather、ReduceScatter、Broadcast、AllToAll 及 P2P Send/Recv；
- 让 work 数超过单个 plan 容量，强制产生多个 plan，再测试连续 group；
- 每个 rank 校验数据，并检查每个用户操作都有唯一 TraceID/counter、正确 op/count/algorithm/protocol；
- 检查一个 group 的所有 counter 都在统一 callback 后完成，且每个 slot 都能回收复用；
- 检查无 TraceID 串写、无重复 complete、无错误 `OBSERVATION_DROPPED`。

**验收指标**：操作数、enqueue 数、complete 数按 rank 对齐；`counter` 集合和预期操作集合一致；连续多轮后 active slot 回到零或可复用；通信数据全部正确。

## 5. 按 AntCCL 5.1 补齐故障场景

参考材料功能评测表包含 6 类场景。当前三个注入测试只能证明基础端到端链路，不能代表这些场景全部复现：

| AntCCL 场景 | 当前覆盖情况 | 需要补充的验收 |
|---|---|---|
| rank 2 未进入 collective | 已有近似覆盖：rank 2 跳过首个 collective，5 s 后由测试程序 abort | 保留每 rank 日志，核对 monitor 首次发现时间、定位 rank 和 root cause；此项不应把测试程序的 abort 当成诊断器发现 hang |
| rank 2 在 collective kernel 内异常导致 hang | 未覆盖。当前 stall 是 stream event 阻塞在通信开始前，不是 kernel 内某次 send/recv 后异常 | 在受控设备侧注入通信阶段停滞，验证收发计数停在哪个阶段、判出 rank 2 首错，并记录发现/定位耗时 |
| rank 2 的 collective 规模比其他 rank 少 1024 B | 未覆盖 | 只让 rank 2 使用不同 count，验证各算法的计数差异能定位首错 rank，并能解释其他 rank 因等待而未完成 |
| collective 结束后 hang 在非 collective 代码 | 未覆盖 | collective 正常完成后只阻塞一个 rank 的 host 线程；应报告 job/rank 停滞，但明确通信证据不足以把它归因于 collective kernel |
| 计算慢：调用 collective 前 rank 2 sleep 3 s | 当前 2 s 延迟发生在 collective 已 enqueue 后的 GPU stream 上，不能等同于该场景 | 在 host 调用 collective 前 sleep，验证由 enqueue 起始时间差识别为计算侧慢，而不是 channel 进度慢 |
| 通信慢：每次 send/recv 后 rank 2 sleep 2 us | 未覆盖。当前 2 s 单次 stream delay 不等价于通信迭代内持续降速 | 在设备通信循环内逐次延迟，检查 rank 2 的 send/recv progress rate 相对同 Ring/Tree 可比 rank 下降，并量化采样精度及检测阈值 |

测试矩阵按 Ring 和 Tree 分别运行；Tree 只比较同层 rank。每项保存实际 algorithm/protocol、count、per-channel send/recv、采样曲线、发现/定位时间、首错 rank 和 monitor 输出。当前 gfx936 不支持 LL128，测试只覆盖实际支持的协议。

## 完成定义

这四项全部需要代码或测试补充，并保存原始证据：

1. off 模式不分配 CCL-D frame；on 模式的 resource 记录能够给出诊断线程 CPU 和 pinned frame 的实际数值；
2. CCL-D off/on 配对实验给出 GPU/step/吞吐影响；
3. grouped-work 回归覆盖多 plan、混合 collective 和 P2P，完成记录与 slot 生命周期无误；
4. 结果写入 [Evaluation](CCL-D_EVALUATION.md)，再判断是否满足材料的 1% 参考口径。

第 5 节的六类功能场景也必须按上表逐项完成，才可以宣称 AntCCL 5.1 功能验收完整。
