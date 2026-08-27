# 向量后端扩展契约

`RAG Studio` 的默认后端是本地精确余弦扫描，适合本项目定义的 durable
single-node 部署。它不是隐藏的全局选择：组合根
[`build_service_from_settings`](../rag_system/bootstrap.py) 接受明确注入的
`IndexRepository`，派生项目可在自己的组合根提供 Milvus、Qdrant、pgvector
或受管向量服务适配器，而无需修改 HTTP、租户、问答或评测契约。

## 必须实现的接口

实现 [`IndexRepository`](../rag_system/ports.py) 与其返回的
[`VectorIndex`](../rag_system/ports.py)：

- `build(index_id, chunks)` 必须按调用方提供的 `index_id` 构建或原子复用一个
  租户范围内的集合，返回稳定的 `IndexRef`。
- `search(query, top_k)` 只能返回该集合的 `SearchHit`，并保证输入上限、确定性
  排序规则和关闭后的失败语义。
- `close()` 只能释放本进程句柄；`delete()` 才能永久删除底层集合。
- `delete(index_id)` 必须是按精确 ID 的幂等删除；`healthcheck()` 不得加载模型或
  扫描全量数据。

适配器在自己的隔离测试中还必须调用
`rag_system.vector_conformance.verify_index_repository(repository)`。该认证会验证
健康检查、两个独立集合的构建与结果隔离、结果边界、有限排序分数、关闭、精确幂等删除，
以及删除一个集合后另一个集合仍可查询；它不替代目标语料质量、真实租户凭据隔离或故障演练。

适配器不能从请求、环境变量或文档内容动态导入代码或拼接任意连接字符串。连接
配置、凭据、迁移和网络策略属于派生项目的受信组合根。

## 切换前的验收

1. 对目标语料运行检索与回答评测，保留开发/验证集逐题报告；不要用仓库样例声称
   领域质量。
2. 对删除、租户隔离、进程重启、部分写入、超时和后端不可用分别补充契约测试。
3. 执行容量与恢复演练，记录索引构建耗时、查询 P95、存储占用、RPO 和 RTO。
4. 若迁移到多实例，必须同时外置 Catalog、任务、会话和限流状态；仅替换向量库
   不能使当前单节点运行时变成高可用服务。
