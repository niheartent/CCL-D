# CCL-D 评测记录

当前源码已完成静态检查，新的 gfx936 编译尚未执行，因此本文不记录运行结果。

编译后需要记录：

- 六类 collective 的 TraceID、channel 计数和完成回收；
- Ring/Tree 同位置计数和采样 slow 判定；
- 未进入、部分完成、全部未完成、规模不一致和非通信 hang；
- 训练耗时、吞吐、loss、独立 diagnose thread CPU、pinned memory 和 GPU 开销。

性能比较使用相同 workload、相同 rank 映射和相同环境，仅切换 `RCCL_CCLD_ENABLE`。
