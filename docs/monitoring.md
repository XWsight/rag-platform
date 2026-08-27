# Prometheus 监控与告警

应用在 `GET /metrics` 暴露 Prometheus 文本格式指标。该端点需要具有
`operator` 角色的 API Key；采集器只应持有专用、最小权限的 Key，绝不能使用
部署密钥、云模型密钥或具备写权限的日常调用方 Key。

仓库提供两份可审查的起始资产：

- `monitoring/prometheus/prometheus.example.yml`：Prometheus 抓取配置示例。
- `monitoring/prometheus/rules/rag-platform.yml`：应用可观测指标对应的告警规则。

它们是示例而非 Compose 服务。先复制到受控的监控配置目录，核对网络边界和访问权限，再由
现有 Prometheus/Alertmanager 部署加载。不要把真实 Key、Alertmanager URL 或生产主机名提交到 Git。

## 最小部署步骤

1. 为 Prometheus 单独创建一个仅含 `operator` 角色的高熵 API Key，并把**原始 Key**写入一个只允许 Prometheus 进程读取的文件，例如 `/etc/prometheus/secrets/rag-platform-operator.token`。`bearer_token_file` 会自动添加 `Authorization: Bearer`，文件中不应写 `Bearer ` 前缀。
2. 从示例复制配置与规则文件，替换示例 target。Prometheus 在宿主机运行时可使用私网地址；若它加入与应用相同的 Compose 网络，可使用 `api:8000`。不要为了方便把应用端口暴露到公网。
3. 在 Prometheus 中检查配置和规则：

   ```bash
   promtool check config /etc/prometheus/prometheus.yml
   promtool check rules /etc/prometheus/rules/rag-platform.yml
   ```

4. 重载 Prometheus，随后确认 Targets 页面中的 `rag-platform` 为 `UP`，并在告警系统中为每条规则设置实际负责人、通知路由和升级路径。

初始阈值故意保守：持续 5xx、p95 问答延迟、外部供应商失败、索引构建失败以及持续限流。运行一到两个业务周期后，应根据真实负载、RTO/SLO 与发布窗口校准阈值；不要将示例阈值直接视为产品 SLA。

## 指标边界

应用指标仅包含受限的低基数标签，例如 operation、route、provider 和 error type；不包含租户、API Key、问题、文档、会话或供应商响应内容。`rag_external_call_errors_total` 在回答生成、研究规划和联网检索出现已分类失败时递增，便于区分鉴权、协议、限流、超时、不可用与未知错误，而不会暴露原始异常消息。

`/health/ready` 只覆盖本地耐久层与任务执行器，并不探测模型下载、向量查询或外部供应商。因此 `UP`/readiness 告警也不能代替带有合规脱敏样本的定期端到端业务探针。

应用会导出当前等待 worker 的 `rag_job_queue_depth`、所有非终态任务数
`rag_job_active_count`，以及最老非终态任务年龄 `rag_job_oldest_active_seconds`。这些指标没有
租户、任务 ID 或请求内容标签，仅描述当前进程内的工作状态；进程重启后，旧任务会按恢复策略收敛，
因此不能将它们误读为跨进程持久队列深度。

进程重启次数、磁盘余量和备份年龄仍应由主机/容器 exporter 与备份作业覆盖；在把它们接入之前，
不应声称该告警包已覆盖全部运行风险。备份与隔离恢复演练仍按[运维手册](operations.md)执行。
