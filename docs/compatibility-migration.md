# 兼容性迁移与退役

本项目在 v2 的目标是让所有**新**部署与派生代码使用 `rag-platform` 命名，同时让已有部署无数据丢失地升级。旧标识不是默认接口；它们只在下表所列的边界保留，且每一项都有明确的退出条件。

| 边界 | 当前行为 | 退役前提 |
| --- | --- | --- |
| Docker 数据卷 | 新 `.env.example` 使用 `rag-platform-data`；Compose 仍为未配置的原地升级保留旧回退值 | 已停写、备份、复制并验证每个旧卷；所有部署显式指向新卷 |
| 单节点锁 | 新运行时同时锁定 `.rag-platform.instance` 与 `.rag-studio.instance` | 不再支持任何只持有旧锁的二进制版本 |
| 浏览器会话键 | 首次读取旧 `sessionStorage` 键时复制到新键；旧键不会被自动删除 | 旧前端已停止服务，且浏览器标签会话自然结束或已由用户重连 |
| 派生入口 | 新脚手架默认 `asgi.py` 与 `workbench.py`；旧入口只转发 | 所有派生项目和自动化命令已改用新入口 |
| 派生清单 | `rag-studio` 基座身份仍可验证，并在结果中标记 `identity_upgrade_available` | 派生项目已审查其上游基线并把清单更新为 `rag-platform` |

不要在同一持久卷上并行运行旧、新两个 API 版本，也不要手动删除旧锁文件来绕过互斥保护。

## 数据卷迁移

该流程不会覆盖或删除任何卷。先停止旧服务，按[运维手册](operations.md)创建并核对完整备份，再只生成计划：

```powershell
python scripts\migrate_legacy_volume.py `
  --source-volume rag-studio-data `
  --destination-volume rag-platform-data
```

核对输出的源卷、目标卷和 Docker 命令后，才显式执行：

```powershell
python scripts\migrate_legacy_volume.py `
  --source-volume rag-studio-data `
  --destination-volume rag-platform-data `
  --execute
```

执行器要求源卷存在、目标卷尚不存在；它以只读方式读取源卷、以 tar 流复制所有数据，再比较两个卷内所有普通文件的 SHA-256 清单。任何失败都会保留源卷和目标卷以便人工检查，绝不自动清理。成功后把部署 `.env` 的 `RAG_DATA_VOLUME` 改为 `rag-platform-data`，启动新版本并验证 `/health/ready`、鉴权和代表性知识库；确认前保留旧卷。

## 派生项目和入口

新派生层使用 `uvicorn <package>.asgi:app` 与 `python -m <package>.workbench`。旧入口仍可运行，但不应再写入文档、CI 或部署脚本。

对历史派生层运行：

```powershell
python scripts\validate_derivative_compatibility.py <派生目录>\compatibility.json
```

输出 `identity_upgrade_available: true` 时，先核对 `base_revision` 对应的上游，再将清单中的 `base_project` 从 `rag-studio` 改为 `rag-platform` 并重新运行验证。该字段的改动不会自动迁移代码或数据，因此必须在派生项目自己的评测和 `scripts/check.ps1` 通过后提交。

## 下一次破坏性大版本

只有当每个生产部署都完成卷迁移、旧前端不再提供、派生项目都使用当前清单和入口后，才可以在下一次破坏性大版本删除旧数据卷回退、旧锁、旧会话键读取与旧入口转发。删除前必须执行恢复演练并记录受影响部署的版本、卷摘要、备份校验值与回滚结论。
