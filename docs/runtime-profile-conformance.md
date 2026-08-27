# Runtime Profile 一致性验证

`RuntimeProfile` 用于替换一整套运行基础设施，例如 SQLite、文件存储、线程池或本地向量索引。它不是
动态插件机制：派生项目必须在自己的组合根显式导入并传给 `build_production_runtime`，不能把 Python
类路径放入环境变量。

在接入生产前，用临时数据目录执行 `verify_runtime_profile`：它会构造组件、执行 Profile 定义的就绪探针，
随后释放任务、索引和服务资源。该检查不启动恢复工作流、不接收流量，也不替代迁移、并发、端到端或领域
评测。

```python
from dataclasses import replace
from pathlib import Path

from rag_system.config import Settings
from rag_system.runtime_profile import verify_runtime_profile
from rag_system.tenancy import Principal, TenantId

settings = replace(Settings(), persist_data=True, storage_root=Path(".tmp-profile-check"))
principal = Principal(
    subject="profile-check",
    tenant_id=TenantId("profile-check"),
    roles=frozenset({"operator"}),
)
result = verify_runtime_profile(my_profile, settings, (principal,))
print(result.probe_names)
```

Profile 必须：

1. 返回 `RuntimeComponents`，并负责其在平台接管前的所有权；
2. 返回至少一个名称唯一、无副作用、失败关闭的 `HealthProbe`；
3. 保持任务取消、幂等、租户隔离、恢复和删除语义；
4. 为任何持久化布局准备迁移、备份、恢复与回滚说明；
5. 在派生项目自己的评测集与容量门禁中证明质量和规模行为。

验证成功只证明基础契约在该临时环境可用，不能作为生产就绪或性能声明。基座的回归套件还会以
一个确定性 Profile 执行建库、问答、重启后重新装配、会话清理和删除；新 Profile 在接入生产前
应对相同生命周期补充自己的黑盒测试，并复用已有取消与恢复故障测试。对状态存储、任务队列或
向量后端的替换，不得仅以健康检查成功作为兼容性结论。
