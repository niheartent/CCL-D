# CCL-D 后续路线

## 1. TraceID 生命周期专项验收

构造真实的多 plan、多 work、多 channel collective，验证同一用户操作只产生一次最终收口，且 AllToAll 展开的 P2P 始终保留父身份。覆盖重复完成通知、缺失通知和 communicator 销毁。

## 2. 协议计数专项验收

按材料对 Simple、LL、LL128 选择稳定消息规模，逐项核对 send/recv 插入点、channel 归属和累计字节数；分别测试 mode 1 和 mode 2。

## 3. 分析器专项验收

补齐真实 Tree 拓扑和长轮次 Ring/Tree 对比，验证 slow/hang 的首错 rank、同层比较和通信速率判定。

## 4. 长期与训练场景

执行 1000 轮、多 communicator、多 stream、graph 和真实训练任务，记录数据正确性、completion 收口、pinned memory、CPU 开销和吞吐影响。

Graph、CollNet、NVLS、MSCCL、真实驱动/网络故障和通信自愈属于扩展实验，不作为当前核心完成条件。
