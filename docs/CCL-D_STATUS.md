# CCL-D 当前状态

实现依据是参考材料的图文定义：TraceID、pinned frame、kernel 完成通知、diagnose thread，以及 Ring 全 rank和 Tree 同层比较。44-byte 扩展区只保留空间，当前没有字段和语义。

| 部分 | 当前实现 | 验证状态 |
|---|---|---|
| TraceID/Header | `commId + counter + reserved[44]`；header 保存 KID、mode、counter、detectChannel | host 布局单测通过；待 gfx936 运行核对 |
| channel body | 每个 block 按 channel 保存 `sendCount/recvCount` 两个 64-bit 计数 | 结构迁移完成；待 Simple/LL/LL128 运行核对 |
| 生命周期 | enqueue 保留 block；kernel callback 排队 counter；diagnose thread 输出后释放；destroy 时 drain | 代码完成；待设备任务验证 |
| 资源不足 | 没有可用 free block 时输出 `OBSERVATION_DROPPED`，通信继续，不覆盖 active block | analyzer 单测覆盖 |
| Ring/Tree | 输出 Ring 邻居或 Tree parent/children/depth；Ring 全 rank、Tree 同层比较 | 合成单测通过；待双节点真实拓扑验证 |
| slow/hang | 缺 rank、未完成、速率异常、同节点停滞和拓扑不对称规则 | 合成单测通过；待受控运行场景验证 |

`RCCL_CCLD_SLOTS` 是运行时参数，默认值只用于初始化，可通过环境变量调整；代码不依赖固定 16 slot。通信观测只使用材料定义的 header、channel send/recv body 和生命周期事件，不加入 peer 字节矩阵、协议矩阵或自定义父操作状态。

gfx936 独立构建已完成并安装到 `rccl-artifacts/ccld-frame-gfx936/install/`；最终链接日志只有 dcc 未使用参数警告。4-rank gfx936 单节点 20 轮回归已通过 AllGather、ReduceScatter、Broadcast、AllToAll、P2P 数据校验。`RCCL_CCLD_SLOTS=256` 时无观测丢失，monitor 输出 416 个 OK、4 个 enqueue skew；`RCCL_CCLD_SLOTS=2` 时通信仍通过，923 个操作记录为 `OBSERVATION_DROPPED`，符合容量不足策略。双节点 HPDPS-6/7 尚未在本轮运行。
