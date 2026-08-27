# 密钥来源与轮换

本地开发可以使用 `.env` 中的 `RAG_API_KEYS_JSON` 和 `ZHIPU_API_KEY`。生产部署
优先使用只读文件挂载，避免将原始密钥写进容器环境变量、命令行或 Compose 配置。

应用对以下两项支持对应的文件变量：

| 环境变量 | 文件变量 | 文件内容 |
| --- | --- | --- |
| `RAG_API_KEYS_JSON` | `RAG_API_KEYS_JSON_FILE` | 完整 API Key JSON 对象 |
| `ZHIPU_API_KEY` | `ZHIPU_API_KEY_FILE` | 单个 Provider API Key |

文件路径必须是绝对路径、普通文件且不超过 64 KiB。每项只能设置一种来源；同时设置
变量和文件变量会失败关闭，避免部署悄然使用旧密钥。文件内容允许末尾换行，但不会在
错误、日志或对象表示中输出。

## Docker Compose 示例

1. 创建受限目录并写入密钥。`secrets/` 已被 Git 忽略，不要将该目录复制到镜像或提交。

   ```bash
   install -d -m 700 secrets
   printf '%s' '{"replace-with-a-long-random-key":{"subject":"production","tenant_id":"production","roles":["reader","writer","operator"]}}' > secrets/rag_api_keys_json
   printf '%s' 'provider-key-if-needed' > secrets/zhipu_api_key
   chmod 600 secrets/rag_api_keys_json secrets/zhipu_api_key
   ```

2. 删除 `.env` 中的 `RAG_API_KEYS_JSON` 与 `ZHIPU_API_KEY` 值，或保留占位符；覆盖文件会
   在容器内将它们置空，并改用 `/run/secrets/...`。

3. 使用受控覆盖文件启动：

   ```bash
   docker compose -f compose.yaml -f compose.secrets.example.yaml config --quiet
   docker compose -f compose.yaml -f compose.secrets.example.yaml up -d
   ```

提供商未使用时，`secrets/zhipu_api_key` 可以是空文件。API Key JSON 文件不能为空，且仍会
按启动时的严格凭据规则校验。部署平台已有 Vault、Kubernetes Secret 或云密钥管理服务时，
应以只读普通文件方式挂载到容器，并设置同名 `*_FILE` 变量；不要为了兼容而恢复明文环境变量。

## 轮换

先创建新文件、通过单独的维护窗口重启单节点服务并用新 API Key 验证；确认调用方切换后，
安全销毁旧文件。运行时不会热读取密钥，轮换必须重启。轮换前后仍应执行备份与就绪检查，且绝不把
原始 Key 粘贴到终端历史、工单、截图或日志。
