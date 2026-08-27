# 部署 Profile

当前基座只支持一个正式运行形态：`local-durable`。它将 SQLite、文档、向量索引和
任务快照放在同一受控卷上，并通过实例租约拒绝多个进程同时写入。

从 [`profiles/local-durable.env.example`](../profiles/local-durable.env.example) 创建私有
`.env` 后，先运行 `docker compose config --quiet`、`scripts/check.ps1`，再执行一次备份与
恢复演练。`RAG_JOB_WORKERS`、队列上限和租户存储上限是容量配置，不是高可用开关。

当任一目标需要多副本写入、跨节点任务、共享会话/限流或独立 RPO/RTO 时，不应复制该
profile；应新建 Runtime Profile，并先通过运行时行为契约、向量后端认证、迁移和多租户
故障演练，再将其列为正式部署形态。是否进入这一阶段必须以目标机器上的端到端负载数据决定：
记录语料规模、并发、冷/热模型、查询 P95、错误率、备份恢复耗时和允许的数据丢失窗口；不要从
本地精确索引微基准直接推导生产 SLA。
