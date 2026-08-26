# 提供商适配器一致性

`ProviderFactory` 是派生项目接入模型、联网搜索或查询规划器的唯一组合根扩展点。它只装配
适配器；检索路由、请求级 `allow_cloud` / `allow_web` 授权、claim/citation 复核和错误响应
仍由基座负责。

## 启动期契约

工厂必须实现 `create(settings) -> ProviderBundle`。`ProviderBundle` 在构造时会检查：

- `chat_model` 实现 `ChatModel`：`available` 与 `answer(...)`；
- `web_search` 实现 `WebSearchProvider`：`available` 与 `search(...)`；
- 可选 `query_planner` 实现 `QueryPlanner`：`available` 与 `plan_queries(...)`。

`build_service`、`build_service_from_settings` 和 `build_production_runtime` 都会通过
`create_provider_bundle` 创建该对象。缺少端口、工厂没有 `create` 或返回其他类型时，进程会
在启动阶段以明确的 `TypeError` 失败，而不是在用户请求期间产生不完整的服务。

最小实现形式：

```python
from rag_system.provider_factory import ProviderBundle


class AcmeProviderFactory:
    def create(self, settings):
        chat = AcmeChatModel(settings)
        return ProviderBundle(
            chat_model=chat,
            web_search=AcmeWebSearch(settings),
            query_planner=chat,
        )
```

## 派生项目的必测项

启动期结构检查不能替代适配器自身的离线测试。每个派生提供商至少要覆盖：

1. 缺失凭据时 `available` 为假，且不会发起网络请求；
2. 超时、认证、限流和响应格式错误映射为稳定的 `provider_errors.py` 错误，且不泄漏上游正文或密钥；
3. 正常回答只返回结构化 `GeneratedAnswer`，由 `AnswerProtocol` 解析；
4. 非法 claim、无效 citation 或不完整 JSON 失败关闭；
5. `close()`（如实现）可重复调用并释放 HTTP/SDK 资源；
6. 工厂可由 `create_provider_bundle(factory, Settings())` 在不访问外网的情况下创建。

内置智谱适配器是参考实现；其 transport、协议重试与工厂装配测试位于
[`tests/test_providers.py`](../tests/test_providers.py) 和
[`tests/test_provider_factory.py`](../tests/test_provider_factory.py)。

派生脚手架默认生成的工厂测试会调用 `verify_offline_provider_factory`：它在清空环境、阻止
socket 连接的条件下验证无凭据装配、严格布尔 `available` 状态和可选 `close()` 的幂等性。
这项通用检查不调用 `answer` 或 `search`，因此不能替代上述适配器专属的错误映射、响应协议和
引用失败关闭测试。
