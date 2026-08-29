# B0：可审计 Workflow DSL 契约

本文件记录通用 AI 应用平台阶段 B 的首个可合并切片。它先固定安全、可版本化的工作流语言，后续才接入
SQLite 存储、API、异步运行器、审批与发布门禁；不能以未受约束的 JSON、Python callback 或可执行脚本
替代该契约。

## 当前范围

`WorkflowSpec` 是 schema version 为 `1` 的不可变 DAG。它具备：

- 显式输入、节点、资源引用和公共输出；
- 严格 JSON 解码、精确字段集合、拒绝重复键和非有限数字；
- 规范化 JSON 与 SHA-256 digest，供 Revision、Run、评测和审计共同引用；
- 静态检查输入引用、节点输出、直接依赖、环和不连通节点；
- 仅可绑定 `knowledge_base` 与 `model_profile` 的不含密钥资源 ID。

首个原生节点集合是 `knowledge.retrieve`、`prompt.render`、`model.generate`、`grounding.validate`、
`condition` 和 `human.approval`。每一种节点都有固定输入、输出、资源类型和参数白名单。DSL 不包含代码、
shell、Provider URL、Provider 密钥、动态工具名或自由表达式。

## 设计边界

- 此切片不改变既有 `knowledge_chat` Application、其 API 或 SQLite schema。
- `human.approval` 目前只是语言契约；持久化等待、审批 API 与恢复语义属于 B2。
- 循环、通用重试和第三方 Tool/MCP 尚未进入 DSL。循环必须等执行预算和耐久状态机一并落地，不能先开放。
- `WorkflowSpec.digest` 是配置可追溯性，而不是模型输出逐字可复现性的承诺。

## 后续切片

1. 为 Workflow Draft/Revision/Deployment/Run/StepRun 建立存储和 API 契约，并将 Application 配置改为有
   discriminator 的联合类型；既有 Knowledge App 保持 v1 兼容。
2. 实现只读、同步的原生运行器：检索、Prompt、生成、归因验证和条件输出。
3. 引入单节点耐久运行状态机、预算、取消、审批和发布评测门禁。运行中非幂等节点在重启后必须标为
   `interrupted`，不能自动重放。
