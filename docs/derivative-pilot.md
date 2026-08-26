# 首个派生项目试点

首个派生项目的目标是验证基座边界和交付流程，而不是用虚构案例证明业务效果。请选择一个有明确负责人、
可合法使用的代表性领域，并把领域代码与数据治理放在派生层。

## 准备清单

1. 用 `scripts/init_derivative.py` 创建派生层，并在 `UPSTREAM.md` 记录基座 commit。
2. 明确数据负责人、用途、敏感级别、保留与删除策略；原始业务数据、API Key 和模型回答不能进入 Git。
3. 建立来源隔离的 development、validation 与 held-out test 集，覆盖可回答、无答案、权限边界和高风险请求。
4. 先冻结 `evals/governance.json` 指向的检索与回答 suite/contract，再调模型、阈值或 prompt。
5. 为每次候选变更保存脱敏的指标、失败 case ID、配置摘要、基座 revision 和回滚方案。

## 验收门槛

- `scripts/check.ps1`、派生适配器离线测试和领域 suite 校验均通过；
- `validate_derivative_evaluation.py --require-ready` 只在真实 held-out test 未消费时通过；
- 至少演练一次持久卷备份、隔离卷恢复、租户只读检查与抽样检索；
- 若改动检索、模型或路由，必须在 development 后于 validation 确认，不能把用于调参的 test 称为盲测；
- 性能结论必须标明机器、语料、并发、冷/热模型、端到端 P95 与错误率，不得从合成微基准外推。

完成以上项目后，才适合将该派生层作为真实用户试点，而不是仅展示界面或单个成功问答。
