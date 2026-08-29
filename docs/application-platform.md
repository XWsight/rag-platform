# 版本化应用平台

第一阶段提供 `knowledge_chat` 应用内核。租户在 Project 中创建稳定 Application 身份，配置写入不可变
Draft 可编辑并通过版本号进行乐观并发控制；Draft 快照为不可变 Revision，Deployment 原子切换生产 Revision；原有 `/v1/knowledge-bases` 和 `/v1/answers` 保持兼容。

每个 Revision 都包含 `model_profile_id`，它只能引用部署启动时由 `RAG_MODEL_PROFILE_IDS`
声明的受信任配置（默认仅 `default`）。`RAG_MODEL_PROFILE_MODELS` 可将每个 ID 显式绑定到实际
模型名（例如 `default=glm-5.2,fast=glm-4.5-air`），并且必须覆盖全部 ID；未设置时所有受信任
ID 回退为 `ZHIPU_MODEL`。运行时会解析该绑定并将模型名传给供应商适配器。配置引用不携带 Provider
地址、模型密钥或其他密钥材料；未知引用会在创建、发布和运行时被拒绝。

## 生命周期与权限

| 操作 | API | 角色 | 持久化结果 |
| --- | --- | --- | --- |
| 读取/更新 Draft | `GET` / `PUT .../draft` | `writer` | 版本号冲突拒绝、Draft 更新审计 |
| Draft 快照为 Revision | `POST .../draft/revisions` | `writer` | 不可变配置与资源绑定 |
| 创建项目/应用/Revision | `POST /v1/projects`、`/v1/applications`、`.../revisions` | `writer` | 审计事件和不可变配置 |
| 发布或回滚 | `POST .../deployments` | `operator` | 原子写 Deployment、审计事件与 active Revision |
| 运行 | `POST /v1/apps/{id}/answer` | `reader` | 响应包含 application、Revision 与 trace |
| 写入评测证据 | `POST .../evaluations` | `writer` | 校验不可变 Revision 和配置摘要后持久化 |
| 查询绑定、审计与评测 | `GET .../bindings`、`.../audit-events`、`.../evaluations` | `reader` | 只读发布证据与资源关系 |
| 退役 | `DELETE /v1/applications/{id}` | `operator` | 状态变为 archived；保留 Revision、Deployment 与审计历史 |

发布请求必须携带当前 `active_revision_id`（首次发布为 `null`）；陈旧请求返回 `409 application_conflict`，不会静默覆盖其他操作者的发布。应用只保存策略和平台资源 ID，不保存 Provider 密钥。知识库绑定在创建 Revision 和发布时必须属于同一
租户且为 READY；运行时再次验证。多个知识库会同时租用其索引，按可比较分数确定性融合证据，再执行一次
路由与生成。应用会话按租户、应用、Revision 和调用方 session 隔离。

## 数据与迁移

`/data/applications.sqlite3` 独立于 Catalog、jobs 与幂等仓储，使用 WAL、短事务和严格 schema。schema v8
增加持久化 Draft、`retrieval_profile=default|focused`、受信任模型配置引用、版本评测证据及显式 Deployment 状态；v1 会依次补齐 `allow_cloud=false`、退役审计事件、检索策略、Draft、模型配置、评测表和 Deployment 状态，再迁移到 v8。每个应用至多有一个 `active` Deployment，后续发布会将旧记录标记为 `superseded`。未知版本、表/索引损坏和无版本旧表
均拒绝启动。升级前必须执行停写全卷快照；旧镜像不能直接打开迁移后的数据库，回滚必须恢复旧镜像和快照。

## 发布验证与评测

发布后先运行通用只读探针，再验证指定应用的 active Revision 同时存在于 Revision 和 Deployment 历史：

```bash
python scripts/verify_runtime_probe.py --base-url http://127.0.0.1:8000 \
  --api-key-file /run/secrets/rag-platform-reader.token
python scripts/verify_application_probe.py --base-url http://127.0.0.1:8000 \
  --api-key-file /run/secrets/rag-platform-reader.token --application-id app_<id>
```

两个探针均只发送 GET，不执行回答或修改会话。评测发布候选时，用
`bind_application_evaluation(revision, benchmark, generated_at=...)` 将现有结构化回答报告绑定到不可变
Revision 和配置 SHA-256。发布流水线使用 writer 凭据将其 JSON 对象作为 `report` 提交到
`POST /v1/applications/{application_id}/revisions/{revision_id}/evaluations`；服务会验证路径、不可变
Revision 和配置摘要后持久化，再由 GET 接口读取。报告不复制密钥或问题之外的部署配置。报告、源码提交、
镜像 digest 和发布审批应作为同一个发布证据包保存。

## 运维边界

`/metrics` 中 `operation=application_read|application_manage|application_publish|application_answer` 提供低基数
应用流量、错误率和延迟信号；绝不以应用、Revision、租户或知识库 ID 作为指标标签。`session_policy.ttl_seconds` 在应用运行时强制清理过期上下文；`require_citations` 控制生成回答是否必须逐条引用证据。`focused` 检索策略只保留最高分证据。应用回答响应提供白名单化 `diagnostics`（计数、比例和通用错误代码），不会暴露 Provider 文本、密钥、租户或资源标识。配置与版本身份从
Revision/Deployment/Audit 查询，不从高基数监控标签推断。退役不会擦除 SQLite 页或备份；介质级删除仍按
全卷保留、加密和密钥销毁策略处理。
