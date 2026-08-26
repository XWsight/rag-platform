# 单节点生产部署

本部署形态面向一台受控主机：一个 API 容器、一个持久卷，以及 SQLite + 受限本地向量索引。容器可以重建，`/data` 中的数据会保留；但主机、磁盘或持久卷损坏仍会造成数据丢失，所以必须配置离机备份。

这是一套 **durable single-node** 方案，不支持无共享水平扩展。后台任务、限流状态和部分会话状态位于单进程内，SQLite 与向量索引也只有一个写入节点。不要同时启动两个指向同一 `/data` 的 API 容器。

## 前置条件

- Docker Engine 25 或更新版本；Docker Compose v2.24 或更新版本。
- 至少 2 个 CPU、4 GiB 内存作为起始配置；最终规格必须由真实文档量和并发压测决定。
- 首次使用默认嵌入模型时需要访问 Hugging Face；只有启用云端生成或联网搜索时才需要访问智谱 API。
- 只允许受信主机访问发布端口，外网场景必须在前置代理终止 TLS。

## 首次启动

1. 从示例创建本地配置，并限制文件权限：

   ```bash
   cp .env.example .env
   chmod 600 .env
   python -c "import secrets; print(secrets.token_urlsafe(48))"
   ```

2. 编辑 `.env`：

   - 只有需要云端生成或联网搜索时才设置 `ZHIPU_API_KEY`；纯本地检索不需要。
   - 将上一步生成的随机值写入 `RAG_API_KEYS_JSON`。每个调用方使用独立密钥与最小角色集合。
   - 生产环境建议设置 `RAG_API_DOCS_ENABLED=false`。
   - `RAG_ALLOW_CLOUD_DEFAULT` 与 `RAG_ALLOW_WEB_DEFAULT` 仅是 Gradio UI 初始值；REST API 的每次回答都必须显式授权外发。
   - 可选地设置 `HF_TOKEN`，降低模型下载限流风险。不要把任何密钥写进镜像或 Git。

3. 检查最终配置并构建镜像：

   ```bash
   docker compose config --quiet
   docker compose build --pull
   ```

   Windows Docker Desktop 使用者可先运行 `python scripts/docker_preflight.py --compose`。该检查只验证
   Docker Engine、`.env` 和 Compose 解析，不会构建或启动容器。若报 `dockerDesktopLinuxEngine` 不可用，
   这是 Docker Desktop/WSL 环境故障；先修复引擎，再判断项目构建结果。

4. 启动并验证：

   ```bash
   docker compose up -d
   docker compose ps
   curl --fail http://127.0.0.1:8000/health/live
   curl --fail http://127.0.0.1:8000/health/ready
   ```

默认只绑定 `127.0.0.1:8000`。如需由同机反向代理访问，可保持该默认值；只有网络边界、TLS、认证和防火墙均已落实时，才把 `RAG_BIND_ADDRESS` 改为指定的非回环地址。不要直接把未加密的 API 暴露到公网。

受保护端点接受以下两种方式之一，不能同时提供：

```bash
curl -H 'X-API-Key: <调用方密钥>' http://127.0.0.1:8000/v1/knowledge-bases
# 或：Authorization: Bearer <调用方密钥>
```

`GET /health/live` 与 `GET /health/ready` 不返回敏感信息。`GET /metrics` 需要带 `operator` 角色的凭据。

`ready` 检查本地文档存储根、Catalog、向量目录、任务执行器和耐久 job 快照库，不加载 Embedding、不执行向量查询，也不探测智谱或 Hugging Face。它适合阻止本地持久层或任务管理器明显故障的实例接流量，但不能代替代表性业务探针。

## 容器安全与运行边界

- 进程固定以 UID/GID `10001:10001` 运行，不拥有 Linux capabilities，并启用 `no-new-privileges`。
- 容器根文件系统只读；仅 `/data` 和受限的 `/tmp` 可写。
- `/data` 包含目录型文档存储、`catalog.sqlite3`、`idempotency.sqlite3`、`jobs.sqlite3`、本地向量索引文件和模型缓存。
- 向量索引仅在当前进程内使用；Embedding 模型标识由受信部署配置固定。`/data/vector` 中的索引文件是受信持久化状态，不得导入不可信文件或在服务运行时由其他进程修改。索引文件按清单校验，写入采用临时文件加原子替换；发现损坏时服务关闭式失败并要求从原文重建。
- Compose 设置 CPU、内存、进程数、文件描述符和日志轮转边界。默认 4 GiB 只是起点，不能替代压测。
- API 固定一个 Uvicorn worker。增加 worker 数不会把本地状态变成分布式状态，反而会破坏任务、限流和存储的一致性。
- 启动时会在持久根获取 OS 级独占实例锁；第二个指向同一 `/data` 的进程会快速失败。该锁是最后一道保护，不替代编排层的单副本约束。
- SIGTERM 触发最多 30 秒的应用优雅关闭，Compose 等待 45 秒后才强制终止。

如果改用宿主机绑定目录而不是命名卷，应先创建专用目录并把它的属主设为 `10001:10001`，同时确认其中没有符号链接。不要为了绕过权限问题而让容器以 root 运行。

当前依赖例外列表为空。依赖审计会拒绝每一个新漏洞、可升级修复、版本漂移或陈旧例外；不得通过新增长期豁免来掩盖可利用风险。

## 构建与发布可复现性

Dockerfile 固定 Python 补丁版本，先生成 wheel 集合，再在运行阶段离线安装；运行镜像不会携带编译缓存。默认构建从 PyTorch 官方 CPU wheel 源解析 `torch`，避免在无 GPU 的单节点容器中携带 CUDA 运行时。若要启用 GPU，应维护单独的、经过压测和供应链审查的镜像，而不是修改通用部署镜像。`.dockerignore` 排除了 `.env`、密钥、Git 元数据、本地数据、测试缓存和报告。

镜像写入 OCI `org.opencontainers.image.revision` 标签。CI 用 GitHub 提交设置并在构建后校验该标签；本地构建默认值为 `unknown`，可在 `.env` 中设置 `RAG_SOURCE_REVISION=<已审查提交>` 以保留同样的追溯信息。

`requirements.txt` 固定易审查的直接依赖版本，`requirements-py311.lock` 与 `requirements-py312.lock` 分别固定对应解释器的运行时传递依赖 SHA-256 哈希。Docker 和 CI 都以 `--require-hashes` 使用对应锁文件；直接依赖变更必须同时有意更新并审查两个锁文件。生产发布仍应在 CI 中只构建一次，把镜像推送到受控仓库，并以不可变镜像 digest 在各环境间晋级；不要在生产主机临时重新解析依赖并把它视为相同制品。

示例标签流程：

```bash
RAG_IMAGE_TAG=2026.08.11-1 docker compose build --pull
docker image inspect rag-studio:2026.08.11-1 --format '{{json .RepoDigests}}'
```

正式环境应记录代码提交、镜像 digest、配置版本、数据快照和部署时间。输出为 `[]` 或 `null` 表示镜像尚未推送到仓库，此时不能声称它是可跨主机验证的不可变制品。

仓库 CI 还会保存 `release-manifest-<commit>` 工件，其中记录干净源码提交、包版本及 Dockerfile、Compose、`pyproject.toml`、运行/开发依赖清单和两个运行时哈希锁的 SHA-256。该清单不读取 `.env`，适合把一次镜像构建关联到其受控输入；它不替代镜像签名或 SBOM。

推送与稳定 `pyproject.toml` 版本完全一致的 `v<version>` Git tag 后，`release` workflow 会附上 SPDX
SBOM、release manifest 和不可变 `image@sha256:...` 引用并创建 GitHub Release。当前开发版本带 `.dev`，因此会被工作流拒绝，避免把开发
快照误标为正式发行。稳定 tag 会把镜像推送至 `ghcr.io/<owner>/rag-system`，生成 BuildKit provenance/SBOM，
并提交 GitHub build provenance attestation；部署时必须记录和固定该镜像 digest，不能只引用可变 tag。
首次发布前需在仓库 Settings 中确认 GitHub Actions 具有 Packages 写入权限，并按组织策略将 GHCR 包设为
公开或授权给目标运行环境。该证明只覆盖 GitHub Actions 构建链路，不替代运行时密钥管理、发布审批或离机备份。

## 反向代理

推荐让 Caddy、Nginx 或云负载均衡器在同机或受控内网终止 TLS，再转发到 `127.0.0.1:8000`。代理必须：

- 限制请求体大小和连接/响应超时；上限应与应用的文件限制一致。
- 不记录 `Authorization`、`X-API-Key`、上传正文或问题正文。
- 生成并透传请求 ID，但不能信任公网客户端提供的任意身份头。
- 只把可信代理地址加入转发头信任列表；当前容器命令默认不信任任意代理头。

## 单节点边界与扩展路径

此形态可用于需要持久化、认证、租户隔离、审计和可恢复性的单节点工作负载。它不提供跨可用区容灾、在线滚动升级或多副本写入。

若容量或可用性要求超过单节点，扩展前应先把以下状态外置：事务目录迁移到受管数据库，向量索引迁移到支持多副本的服务，后台任务迁移到持久队列，会话与限流迁移到共享状态存储，并增加对象存储、分布式追踪和经过验证的迁移工具。在这些改造和故障演练完成前，不应把本 Compose 文件复制成多个实例。
