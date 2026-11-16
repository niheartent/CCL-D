# CCL-D 复现规格与代码差距

实现以 `/workspace/CCL-D/paper/【脱敏版】AntCCL_ A High-Precision Diagnostic System for Collective Communication.docx` 的文字和图片为依据。文字未定义的字段不擅自赋予语义；先按图保留空间并置零。

## 图片给出的设计

| 图示 | 设计约束 |
|---|---|
| 增强架构图 | 通信域层提供 rank、channel、path type；操作层区分算子、Ring/Tree 算法、Simple/LL/LL128 协议；传输层包含 channel 与 P2P 路径。TOPO、TraceID、Kernel Diagnose 和 Diagnose Metrics 进入 slow/hang Decision Analyzer。 |
| TraceID 布局图 | 固定 60 bytes：8-byte commId、8-byte counter、44-byte extension。正文明确扩展位为预留位置，不定义更多子字段；实现必须保留其长度并初始化为零。 |
| pinned buffer 图 | Header 保存 kernelIndex(KID)、mode、counter、detectChannel、traceid；每个 KID 的 body 按 channel 保存 sendCount/recvCount。 |
| 线程/事件图 | diagnose thread 从 pinned memory 读取；kernel 通过 DMA 更新；kernel 完成后触发 hostcallback，由 callback 通知 diagnose thread 更新读取位置；communicator destroy 时 join diagnose thread。 |
| diagnose thread 流程图 | 外循环跟踪通信域和 TraceID；操作期间循环计算收发速率、检查 TraceID 变化、通信域停滞和 hang 超时，再输出集合通信统计。 |
| slow 决策图 | trace 耗时超过基线后，结合 rank 粒度 trace 步调及 sendRate/recvRate 判断；步调差异明显时定位慢操作 rank，步调相近时定位速率最低 rank，混合情况联合给出候选。 |
| hang 决策图 | 先找未进入集合通信的 rank；若都进入，检查是否有非 hang rank；若全部 hang，结合算法拓扑及对比位置的收发次数定位候选 rank。 |
| case 图 | 至少区分正常、单 rank 未进入、集合通信规模不一致、单 rank 设备/驱动错误导致 hang。 |

文字还描述 mode 0/1/2：关闭、每 channel tid0 统计、每 channel 所有 tid 原子统计。Simple send/recv 插在 generic operation 末尾；LL/LL128 send 在 waitSend 后、recv 在 generic operation 末尾。Ring 跨 rank 比较，Tree 同层比较。

诊断空间正文给出 header 字段宽度：`kernelIndex`、`mode`、`counter`、`detectChannel` 各 8 bytes；TraceID 内 `commId` 为 8 bytes、`counter` 为 8 bytes、扩展位为 44 bytes。`detectChannel` 表示本次诊断操作涉及的 channel 数。body 中每个 channel 的 `sendCount` 与 `recvCount` 各 8 bytes。新 `ccld_data.h` 已按该布局建立独立类型并由静态/单元检查验证；旧生产结构尚未迁移，因此现在还不是实际运行时 ABI。

## 当前实现与差距

| 环节 | 当前代码 | 差距 |
|---|---|---|
| TraceID/Header | 新增固定 60-byte `TraceId` 与 8-byte 对齐 header 类型，扩展位保留并清零；尚未接入 RCCL 生产路径 | 仍需把现有 host/device 记录迁移到该布局，不能继续使用分散的旧字段作为最终 ABI |
| Body | pinned array 按 channel 和 slot 分区，保存收发计数及额外 peer/protocol 字段 | 有材料要求的 send/recv 计数，但结构混入不少扩展；需要明确 KID 与 channel body 的固定映射 |
| 事件生命周期 | communicator 内诊断线程轮询状态；flush callback 将 device body 标为完成；销毁时 join 线程 | 与图示事件方向基本对应，但没有显式 read memory index，需验证状态通知不会丢完成或复用中的 body |
| 速率/状态 | 线程采样计数；complete JSON 携带时间与计数 | 当前采样/输出需核对是否覆盖图示 TraceID 变化、通信域停滞、hang 超时和操作基线 |
| 决策树 | analyzer 存在缺 rank、未完成和 rate outlier 规则 | 现规则不是图示分支：slow 需使用 trace 耗时、步调和双向 rate；hang 需区分未进入、部分 hang、全 hang，并用同算法/拓扑位置的收发次数；不得用全局最小计数替代拓扑位置比较 |
| 拓扑 | 输出 Ring 邻居及 Tree parent/children/depth，analyzer 有基础对称性检查 | 真实 Ring 全 rank 和 Tree 同层定位尚未验收；拓扑位置与 hang 计数候选尚未连接 |

已删除未接生产调用的 RecordPool plan/channel 状态机。容量耗尽策略、AllToAll 父子关联、peer 字节估算和合成故障接口不是图示必要字段，当前不继续扩展。

## 最高效的实施方式

不重写整个 RCCL。先做一次结构设计评审并冻结 host/device ABI：60-byte TraceID 作为逻辑字段，44-byte extension 全零保留；header/body 使用 KID 与 channel 的直接索引；删除 device body 中与核心诊断无关的 peer byte 矩阵和重复协议统计。再把 callback 通知路径缩成一个明确完成标记/读取索引，保证诊断线程复制快照后才释放 body。Python analyzer 改为严格实现图示 slow/hang 分支，并用每个分支的最小合成案例单测。以上 RCCL 改动一次合并后再做一次 gfx936 编译和双节点验收。

估算：结构与生命周期梳理约 1 个开发日；analyzer 决策树和单测约 0.5–1 日；RCCL 编译链接约 30–60 分钟；协议、Ring/Tree、slow/hang 双节点验证约 0.5–1 日。若实际 44-byte 扩展字段在原始材料中另有定义，应先按该定义替换零保留方案，不猜测字段语义。
