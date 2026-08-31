# B0：可审计 Workflow DSL 契约

本文件记录通用 AI 应用平台阶段 B 已落地的基础运行时。它先固定安全、可版本化的工作流语言，再将
SQLite 存储、受限执行、审批暂停/恢复和发布评测门禁连接到同一契约；不能以未受约束的 JSON、回调
或可执行脚本替代该契约。

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

## 已实现的运行时保证

- `Workflow / Draft / Revision / Deployment / Run / StepRun / Approval / Evaluation` 均为租户隔离的耐久记录；
- Draft 采用乐观并发版本，Revision 和 Evaluation 不可变，发布会原子替换活跃 Revision；
- 每次执行都记录输入摘要、规范摘要、节点输入/输出摘要、节点状态和安全错误码；
- 原生节点只能由平台注册的执行器实现。运行时拒绝任意代码、动态导入、shell 和定义内凭据；
- 步数、生成调用次数和总时长均受 Revision 预算限制；任一限制触发时运行失败且保留审计记录；
- 人工审批会耐久地暂停 Run。获批后从受限、租户保护的状态恢复；拒绝会终止该 Run；
- 发布前必须存在一份与目标 Revision 摘要完全一致、全部用例通过的评测记录；
- 进程中断时，正在执行的 Run/Step 会标为 `interrupted`，不会被自动重放。

## 设计边界

- 此切片不改变既有 `knowledge_chat` Application 的兼容契约。
- 当前原生执行器以明确注册的端口接入；检索和生成的生产适配器必须在组合根中显式提供，不能由 DSL
  决定实现或连接位置。
- 循环、通用重试和第三方工具尚未进入 DSL。它们必须先具备对应的预算、幂等、审批和审计语义。
- `WorkflowSpec.digest` 是配置可追溯性，而不是模型输出逐字可复现性的承诺。

## 后续切片

1. 在 HTTP 与 SDK 边界公开 Workflow 的管理、执行、审批和评测资源，同时维持现有应用接口兼容。
2. 为检索和生成提供生产组合根适配器，并对资源级权限、证据覆盖和拒答语义做端到端评测。
3. 引入经过幂等设计的异步队列、超时取消与失败补偿；在此之前不开放通用重试或循环。
