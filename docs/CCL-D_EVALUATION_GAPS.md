# Evaluation 补充实施清单

参考材料第 5 节给出的性能与资源验收口径是：开启诊断后训练耗时、模型精度和资源占用影响控制在 1% 内；示例结果为单步 4.030 s 对 4.032 s、诊断线程 CPU 约 1%、pinned memory 2208 Byte、GPU 无额外开销。下面只记录 CCL-D 当前需要补齐的可测项，不把参考值当成当前结果。

## 1. 诊断线程 CPU

**当前情况**：AntCCL 图示是独立 diagnose thread；当前 RCCL 接口仍复用 `ncclCommThreadMain`，独立线程尚未实现。最终版 64-work、10-iteration 微基准中，8 rank 的共享线程 CPU/墙钟比为 6.18%–8.50%，采样函数 CPU/线程墙钟比为 0.007%–0.160%。这两者都不能直接与 AntCCL 的独立诊断线程约 1% 对比；callback 快照 CPU 也尚未单独统计。因此本项未达到验收标准。

**实现**：

- 在线程入口和退出处使用 `clock_gettime(CLOCK_THREAD_CPUTIME_ID)`；
- 维护 `thread_cpu_ns`/`thread_wall_ns`（共享线程总量）和 `diagnostic_sample_cpu_ns`/`diagnostic_sample_wall_ns`（采样区量）；
- 线程退出时输出 `sample_calls`、设备样本数、`sample_interval_ns` 和两个 CPU 比例；
- 线程退出、初始化失败和诊断关闭时都要能输出完整或明确标记为 partial 的记录。

**验收**：同一 workload 分别关闭诊断、默认采样周期、20 ms 采样周期运行，比较每 rank 的 CPU 时间和占比；检查多节点 rank 间没有异常偏离。目标是给出实际数值，再与 1% 参考口径比较。

## 2. pinned memory

**当前情况**：已修复 `RCCL_CCLD_ENABLE=0` 仍分配 frame 的问题；启用时写出 `frame_block_bytes`、`slots`、`frame_pinned_bytes`，关闭时不创建 CCL-D frame。512 slots 时单 communicator 申请 1,097,728 B；8 rank 共 8,781,824 B（8.375 MiB）。应用侧 VmPin 前后探针受 HIP/RCCL 初始化的 0–202,252 KiB 变化淹没，不能作为 CCL-D 分配量的独立证明；slot 矩阵和可比 allocator 交叉核对尚未完成，本项部分完成。

**当前结构的基线**：`Header` 为 96 B，`MAXCHANNELS=128`，每个 `ChannelBody` 为 16 B，因此 `sizeof(ncclCcldDiag)=2144 B`；`RCCL_CCLD_SLOTS=512` 时每个 communicator 的 frame 为 1,097,728 B。该数值与材料示例 2208 B 不同，不能直接沿用材料数字。

**实现与验收**：

- 先修复 off 状态仍分配 frame 的问题，再做基线对照；
- 初始化时输出 `frame_block_bytes`、`slots`、`frame_pinned_bytes`；
- 释放时输出对应 communicator 的生命周期；
- 用 `/proc/<pid>/status` 的 `VmPin` 或等价 allocator 统计交叉核对；
- 运行 slots=16、256、512，确认总量随 slots 线性变化，且通信结果不受影响；
- 将单 communicator、单 rank 和双节点总量分开报告。

## 3. GPU 开销

**当前情况**：Simple、LL 路径内已有 send/recv 计数更新；gfx936 当前不支持 LL128。HIP event off/on 微基准已做，但均值差异为 4096 元素 81.7%、65536 元素 19.7%；每配置只有 9 个样本，测试同时包含 RCCL/CCL-D callback 与后台工作，未拆出纯 kernel 成本。结果明确显示当前微基准有开销，本项未达到“GPU 无额外开销”的材料标准。

**实现与验收**：

- 使用同一个测试程序和相同 rank 映射，分别设置 `RCCL_CCLD_ENABLE=0/1`；
- warmup 后用 HIP event 记录每 step 时间，至少报告平均值、P50、P95、吞吐和显存占用；
- 按算子、消息大小、协议分别对照，保存原始 CSV/JSON；
- 对计数插桩路径使用 profiler 或 kernel 时间统计，区分通信本身波动与诊断计数开销；
- 以 `(enabled-disabled)/disabled` 计算相对影响，不能以一次运行结论替代配对实验。

## 4. 同一 group 内多 work 并发

**当前情况**：RCCL planner 已将多个 plan 的 CCL-D counter 收集到统一 finalize callback。原实现只排队完成事件，由周期线程稍后 drain；双节点 256×2、512 slots 时少数 rank 在下一轮开始前未及时释放已完成 slot，产生 4 条误丢观测。现实现为 stream-ordered callback 快照最终计数并立即释放 slot，后台线程再写日志；代码已编译安装，256×2、512 slots 回归通过，dropped=0。容量小于同时运行的 work 数时仍按设计丢观测并继续通信。

**实现测试程序**：

- 在 `ncclGroupStart/End` 内提交多个 AllReduce；
- 提交混合 AllGather、ReduceScatter、Broadcast、AllToAll 及 P2P Send/Recv；
- 让 work 数超过单个 plan 容量，强制产生多个 plan，再测试连续 group；
- 每个 rank 校验数据，并检查每个用户操作都有唯一 TraceID/counter、正确 op/count/algorithm/protocol；
- 检查一个 group 的所有 counter 都在统一 callback 后完成，且每个 slot 都能回收复用；
- 检查无 TraceID 串写、无重复 complete；容量足够时要求 dropped 为零，容量不足时量化 dropped 并确认通信继续。

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
