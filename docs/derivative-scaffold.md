# 派生项目脚手架

这个脚手架为已经复制或 fork 的 RAG Studio 基座创建一个**独立派生层**；它不复制运行时、
不改写基座文件，也不生成或移动密钥。派生层通过明确的组合根注入自定义 Provider，因而可以
随时与上游基座同步。

## 创建派生层

在新的基座副本根目录执行：

```powershell
python scripts\init_derivative.py `
  --package-name legal_assistant `
  --output legal_assistant `
  --product-name "法律知识助手" `
  --product-tagline "可追溯证据工作台"
```

目标目录必须不存在；脚本拒绝覆盖已有路径。生成的 `legal_assistant/` 包含：

- 默认委托内置 Provider 的 `provider_factory.py`；
- 可直接运行的 `api_app.py` 和 `local_app.py` 组合根；
- 不访问外网的 Provider 工厂回归测试；
- 产品展示配置覆盖示例；
- 领域评测数据与治理说明。

脚手架先在同一父目录的临时路径渲染并校验模板，完成后才发布目标目录；渲染异常不会留下可被误提交的半成品派生层。

将生成目录、基座 `.env.example` 和你的领域配置一起提交；真实 `.env`、客户数据、模型响应和
密钥不能提交。通过 `uvicorn legal_assistant.api_app:app` 启动派生 API，或通过
`python -m legal_assistant.local_app` 启动本地工作台。

## 修改顺序

1. 先建立领域 development、validation 和 held-out test 数据；不要先调检索阈值。
2. 仅通过环境变量完成品牌和非敏感部署配置。
3. 在 `provider_factory.py` 中替换默认适配器，并按[提供商适配器一致性](provider-conformance.md)
   增加无网络 transport/错误/协议测试。
4. 再增加行业路由、审批或 UI；这些能力必须在派生层实现，而不是修改基座的证据与租户边界。
5. 每次同步上游前后都运行完整 `scripts/check.ps1` 和你的领域评测；记录基座 commit、数据摘要、
   指标和回滚方式。

脚手架本身不是新的部署模式：派生项目仍默认使用 durable single-node 约束。达到规模边界后，
应按基座部署文档把状态逐项外置，而不是通过复制单机容器获得多副本能力。
