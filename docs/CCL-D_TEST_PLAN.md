# CCL-D 验证计划

## 编译前

```bash
git diff --check
python3 -m unittest discover -s tools/tests -q
bash tools/tests/run_fault_simulation_matrix.sh
```

新的 gfx936 验证必须使用新的安装前缀。

## 基本回归

在 gfx936 上验证 AllGather、ReduceScatter、Broadcast、AllToAll、P2P 和 AllReduce，检查 TraceID counter、grouped P2P 父身份、每 channel 收发计数、slot 回收和资源耗尽策略。

## 决策树场景

1. rank 不进入 collective：`HANG_MISSING_RANK`。
2. 已进入但部分 rank 未完成：`HANG_PARTIAL_COMPLETION`。
3. 全部进入且全部未完成：`HANG_ALL_RANKS`。
4. rank 2 少 1024B：`COUNT_MISMATCH` 和 rank 2 首错证据。
5. collective 前 rank 2 延迟 3 秒：`COMPUTATION_SLOW`。
6. 每次通信后增加 2 微秒延迟：`COMM_SLOW` 和异常 rank。
7. collective 完成后外部上报 hang：`NON_COMMUNICATION_HANG_UNLOCALIZED`。

## 性能

关闭和开启 CCL-D 使用同一 workload，对比训练耗时、吞吐、loss、diagnose thread CPU、pinned memory 峰值和 GPU 开销，按 1% 参考口径记录。
