# CCL-D 工作边界

当前源码模块：

1. TraceID 与 RCCL `opCount` 身份；
2. pinned frame、slot 生命周期和资源耗尽策略；
3. 独立 diagnose thread 与 kernel callback；
4. 采样 count 增量；
5. Ring/Tree 拓扑位置比较；
6. slow/hang analyzer；
7. 运行时配置和 fault simulation。

当前阶段只进行编译前静态审计。完成静态审计后，使用独立安装前缀编译 gfx936，再执行双节点验证和 AntCCL 评测场景。
