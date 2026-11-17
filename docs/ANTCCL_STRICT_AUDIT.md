# CCL-D 设计核对

## 结构

TraceID、header、每 channel 的 send/recv body、mode 0/1/2、pinned frame 和 44-byte 预留区均按设计实现。counter 使用通信域 `opCount`；RecordPool 不生成通信身份。

## 生命周期

RecordPool 只复用空闲 slot。grouped P2P 共享父 counter；同一 counter 的所有 kernel plan 完成后才释放 slot。资源耗尽只丢观测，不阻塞通信。

## 线程

CCL-D 使用独立 diagnose thread；kernel callback 通知该线程；communicator 销毁时停止并 join。legacy collTrace 使用独立配置 `RCCL_COLLTRACE_SAMPLE_US`。

## 分析

Ring 按 channel 比较全 rank，Tree 按 channel、层级和父子角色比较同位置 rank。slow 使用采样 count 增量；hang 使用未进入、部分完成、全部未完成和计数异常分支。

## 待运行验收

源码静态检查通过。新的 gfx936 构建、协议路径、双节点故障注入、非通信 hang 事件和训练级资源开销尚未执行。
