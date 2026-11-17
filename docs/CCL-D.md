# CCL-D 复现规格

CCL-D 在 RCCL 通信路径记录 TraceID 和按 channel 的收发计数。kernel 更新 pinned body，完成事件通知 diagnose thread；线程读取 frame、计算速率并输出 JSONL，分析器按 Ring 全 rank、Tree 同层 rank 判断 slow/hang。

## 数据布局

- `TraceId`：`commId(8) + counter(8) + reserved[44]`，总计 60 bytes。预留区当前不定义、不读取、不参与决策。
- `Header`：`kernelIndex`、`mode`、`counter`、`detectChannel`、TraceID。
- 每个 block 的 body：每个 channel 只有 `sendCount` 和 `recvCount` 两个 64-bit 计数。
- `CCLD_SLOTS` 在运行时决定 block 数量；复用前检查 counter，active block 不覆盖。

## 诊断流程

1. enqueue 分配一个 free block 并初始化 header/body；无 free block 时写入 `OBSERVATION_DROPPED`，通信继续。
2. device kernel 按 block/channel 原子累加 send/recv。
3. callback 只排队完成 counter；diagnose thread 只采样该 TraceID 已登记的 channel，写入 enqueue、sample、complete JSONL，然后释放 block。sample 记录中的 `send_rate/recv_rate` 是一个采样周期内的 count 增量，`send_frequency_hz/recv_frequency_hz` 是按实际采样间隔换算的派生值。
4. communicator destroy 前 drain 队列并回收诊断状态。
5. analyzer 使用 host 进入状态、完成状态、开始/结束时间、send/recv rate，以及 Ring/Tree 拓扑位置做判断；slow 使用收发速率证据，停滞判断按 `node/local_ranks/local_rank` 在节点内比较。

## 运行时验证

```bash
export RCCL_CCLD_ENABLE=1
export RCCL_CCLD_LOG=/workspace/CCL-D/results/rank-{rank}.jsonl
export RCCL_CCLD_MODE=1
export RCCL_CCLD_SLOTS=32
export RCCL_CCLD_SAMPLE_US=1000
# Optional host-side initial capacity for a plan's counter list.
export RCCL_CCLD_PLAN_COUNTER_INITIAL=8
python3 tools/ccld_monitor.py '/workspace/CCL-D/results/rank-*.jsonl' --once --all
```

host 单测：`python3 -m unittest discover -s tools/tests -v`。gfx936 独立构建和产物位于 `rccl-artifacts/ccld-frame-gfx936/`。
