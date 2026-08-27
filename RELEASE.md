# 发布流程

`rag-platform` 当前处于 v3 基座演进阶段。只有稳定的语义化版本才能创建 Git Tag
和 GitHub Release；`3.0.0.dev0` 这类开发版本必须先通过发布 PR 转为稳定版本。

## 兼容性政策

- HTTP `/v1` 路径、字段、状态码、认证方式、幂等语义和错误 code 属于公开 v1 契约。
  修改前必须评估调用方影响，并更新 `contracts/openapi-v1.json`。
- 仅新增可选字段或可选端点仍需审阅契约差异；删除、重命名、收紧输入、改变状态码或
  幂等/权限语义属于破坏性变更，必须进入新的 API major。
- 数据卷、SQLite schema、环境变量、派生兼容性清单和 Python 导入路径同样是兼容性
  边界。需要迁移时，发布说明必须给出备份、迁移、验证和回滚步骤。
- 不重写已发布的 tag、镜像 tag 或 release 工件。修复通过新的补丁版本和新的不可变镜像
  digest 发布；部署回滚到前一个已验证 digest。

## 创建稳定发布

1. 在专用发布 PR 中把 `pyproject.toml` 与 `rag_system/__init__.py` 的版本同时改为
   稳定版本，例如 `3.0.0`；更新 `CHANGELOG.md` 的版本日期、兼容性和迁移说明。
2. 如果直接依赖变化，分别为 Python 3.11 和 3.12 重建带哈希的运行时锁，并运行
   `python scripts/verify_dependency_lock.py`。不要手工编辑单个哈希。
3. 运行完整本地验证：

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/check.ps1
   python scripts/verify_openapi_contract.py
   python scripts/verify_wheel.py
   ```

4. 若 API 变更已完成兼容性审阅，显式更新快照并将 JSON diff 与迁移理由放在 PR 中：

   ```powershell
   python scripts/verify_openapi_contract.py --update
   ```

5. 合并发布 PR，确认 `quality-gate` 成功；从该干净提交创建精确 tag，例如：

   ```powershell
   python scripts/verify_release.py --tag v3.0.0
   git tag -a v3.0.0 -m "RAG Platform v3.0.0"
   git push origin v3.0.0
   ```

6. Tag 触发 `release.yml`：它校验版本与 tag、安装锁定运行时、生成 SPDX SBOM 与发布
   清单、构建并推送 OCI 镜像、生成 provenance、进行 keyless Cosign 签名，并创建
   GitHub Release。该工作流目前发布 OCI 镜像与证据，**不自动发布 PyPI 包**。
7. 部署时只使用 Release 中记录的 `image@sha256:...`；验证镜像签名、provenance、
   就绪端点、认证和备份恢复。出现问题时回滚到前一已验证 digest，保留失败版本的
   工件和非敏感诊断供复盘。
