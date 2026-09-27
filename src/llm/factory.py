from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from src.utils import get_logger

from .anthropic import AnthropicProvider
from .base import LLMProvider, LLMResponse, Message
from .config import AgentConfig, ModelConfig
from .openai_compat import OpenAICompatProvider
from .registry import match_provider_backend

logger = get_logger(__name__)

@dataclass(slots=True)
class ProviderSnapshot:
    # 快照把 provider 实例和关键配置签名绑定，后续可用于热刷新判断。
    """描述一次 provider 装配后的运行时快照。

    Attributes:
        provider: 已实例化的 provider 对象，可直接发起请求。
        model: 当前快照绑定的模型名。
        context_window_tokens: 模型上下文窗口大小，用于上层做截断或预算判断。
        signature: 由关键配置计算出的稳定签名，可用于热刷新、缓存失效判断。

    这个结构把“可执行实例”和“影响执行行为的关键信息”打包在一起，方便上层在
    配置热更新时判断是否需要重建 provider。
    """
    provider: LLMProvider
    model: str
    context_window_tokens: int | None
    signature: str
    # 中文注释：下面几项只用于阅读阶段。主节点收到每分钟请求数限制后，
    # 同一次论文阅读会按顺序试备用节点；冷却时间让其他论文暂时跳过已限流的节点。
    read_fallbacks: list[ProviderSnapshot] = field(default_factory=list)
    read_node_name: str = ""
    read_limited_until: float = 0.0
    last_read_rate_limit: LLMResponse | None = None
    global_read_fallback: ProviderSnapshot | None = None

    async def chat_for_reading(self, messages: Sequence[Message], **kwargs: Any) -> LLMResponse:
        """阅读模型遇到请求频率限制时，依次尝试已配置的备用节点。"""

        last_error = self.last_read_rate_limit
        # 中文注释：同一个智能体既在阅读备用列表里、又被选为全局兜底时，
        # 一篇论文不能因为它报错就立刻对同一 API 再请求一遍。
        attempted: set[int] = set()
        for snapshot in [self, *self.read_fallbacks]:
            if snapshot.read_limited_until > time.monotonic():
                continue
            attempted.add(id(snapshot))
            if snapshot is not self:
                logger.info("阅读模型改用备用节点：%s", snapshot.read_node_name)
            response = await snapshot.provider.chat(messages, **kwargs)
            if response.ok:
                return response
            if response.error_kind != "rate_limit" and response.error_status_code != 429:
                last_error = response
                break
            # 中文注释：服务商没有给出等待时间时，按一分钟冷却，避免并发阅读论文
            # 在同一个限流窗口里反复打向已经失败的 API 地址。
            wait_seconds = max(60.0, float(response.error_retry_after_s or 0))
            snapshot.read_limited_until = time.monotonic() + wait_seconds
            logger.warning("阅读节点 %s 被限流，暂停调用 %.0f 秒", snapshot.read_node_name, wait_seconds)
            last_error = response
            self.last_read_rate_limit = response
        # 中文注释：阅读专用节点只负责处理限流；如果仍然失败，最后再用全局兜底节点。
        fallback = self.global_read_fallback
        if fallback is not None and id(fallback) not in attempted and fallback.read_limited_until <= time.monotonic():
            logger.warning("阅读模型仍不可用，尝试全局备用节点：%s", fallback.read_node_name)
            response = await fallback.provider.chat(messages, **kwargs)
            if response.error_kind == "rate_limit" or response.error_status_code == 429:
                # 中文注释：全局兜底也可能被限流，记住冷却时间，后续论文先跳过它。
                fallback.read_limited_until = time.monotonic() + max(60.0, float(response.error_retry_after_s or 0))
            return response
        if last_error is not None:
            return last_error
        return LLMResponse(finish_reason="error", error_kind="rate_limit", error_status_code=429,
                           content="所有阅读节点仍处于限流等待期，请稍后从检查点继续。")

    async def aclose(self) -> None:
        """关闭快照里 provider 自己创建的异步客户端。"""

        # 中文注释：短生命周期调用（例如设置页连通性测试）结束后可以直接关快照，
        # 不需要知道底层 provider 用的是 OpenAI、Anthropic 还是兼容网关。
        await self.provider.aclose()
        for snapshot in self.read_fallbacks:
            await snapshot.aclose()
        if self.global_read_fallback is not None and self.global_read_fallback is not self and all(
            snapshot is not self.global_read_fallback for snapshot in self.read_fallbacks
        ):
            await self.global_read_fallback.aclose()


def make_provider(
    config: ModelConfig,
    agent_name: str | None = None,
    *,
    embedding_profile_name: str | None = None,
    client: Any | None = None,
    timeout_s: float = 60,
    enable_global_fallback: bool = True,
) -> ProviderSnapshot:
    """根据模型配置装配并返回一个 provider 快照。

    Args:
        config: 全局模型配置对象，包含 provider、agent 和 embedding profile 配置。
        agent_name: 需要解析的 Agent 名称；为空时使用 default_agent。
        embedding_profile_name: 需要解析的 embedding profile 名称；为空时不走 embedding 装配。
        client: 可选外部注入 client，通常用于测试、mock 或复用自定义 SDK 实例。
        timeout_s: provider 请求超时时间，单位秒；embedding 和 chat 装配都会透传。

    Returns:
        `ProviderSnapshot`，其中包含实例化后的 provider 和配置签名。

    工厂函数只负责“装配成一个可用 provider 实例”，不关心后续调用 chat 还是
    embedding。默认仍按 agent 装配；当传入 embedding_profile_name 时，则按
    embedding profile 装配同样的 provider 实例，但不新增单独的 make_embedding_provider。
    """
    if agent_name is not None and embedding_profile_name is not None:
        raise ValueError("agent_name 和 embedding_profile_name 不能同时指定，请只选择一种模型配置")

    if embedding_profile_name is not None:
        # embedding profile 只决定“用哪个 provider 和哪个向量模型”，具体请求仍由 provider.embed 负责。
        profile_name = embedding_profile_name or config.default_embedding_profile
        profile, provider_config = config.resolve_embedding_provider_config(profile_name)
        provider_name = profile.provider
        spec = match_provider_backend(provider_config.backend)
        kwargs: dict[str, Any] = {
            "model": profile.model_name,
            "api_key": provider_config.api_key,
            "api_base": provider_config.api_base,
            "generation": None,
            "extra_headers": provider_config.extra_headers,
            "extra_body": provider_config.extra_body,
            "client": client,
            # provider 配置里显式写了超时时间时优先生效；调用方传入的 timeout_s 作为兜底。
            "timeout_s": provider_config.timeout_s or timeout_s,
            "max_retries": provider_config.max_retries,
            "max_concurrency": provider_config.max_concurrency,
            "include_stream_usage": provider_config.include_stream_usage,
        }
        provider = _instantiate_provider(spec, kwargs)
        signature_payload = {"target_type": "embedding_profile", "target_name": profile_name, "profile": asdict(profile)}
        return ProviderSnapshot(provider, profile.model_name, None, _signature(provider_name, signature_payload, asdict(provider_config)))

    # 默认路径保持原来的 Agent 装配方式，避免影响搜索节点、阅读摘要和设置页模型连通性测试。
    agent = config.resolve_agent(agent_name)
    # 中文注释：不存在的智能体名称会回落到 default_agent。
    # 判断兜底是否与主模型相同时，必须使用真正解析到的名称，避免同一个模型被调用两遍。
    resolved_agent_name = agent_name if agent_name in config.agents else config.default_agent
    provider_name, provider_config = config.resolve_provider_config(agent)
    spec = match_provider_backend(provider_config.backend)
    kwargs = {
        "model": agent.model_name,
        "api_key": provider_config.api_key,
        "api_base": provider_config.api_base,
        "generation": agent.generation,
        "extra_headers": provider_config.extra_headers,
        "extra_body": provider_config.extra_body,
        "client": client,
        # provider 配置里显式写了超时时间时优先生效；调用方传入的 timeout_s 作为兜底。
        "timeout_s": provider_config.timeout_s or timeout_s,
        "max_retries": provider_config.max_retries,
        "max_concurrency": provider_config.max_concurrency,
        "include_stream_usage": provider_config.include_stream_usage,
    }
    provider = _instantiate_provider(spec, kwargs)
    # 中文注释：普通智能体都使用同一个全局兜底设置。备用模型本身不再装备用模型，
    # 否则两个失败配置可能互相反复调用。
    if enable_global_fallback and config.global_fallback_agent and config.global_fallback_agent != resolved_agent_name:
        if config.global_fallback_agent in config.agents:
            try:
                # 中文注释：备用 API 必须按自己的地址和密钥创建连接，不能复用主模型外部注入的 client。
                backup = make_provider(config, config.global_fallback_agent,
                                       timeout_s=timeout_s, enable_global_fallback=False)
                provider.global_fallback_provider = backup.provider
            except Exception as exc:
                logger.warning("全局备用模型无法装配：%s", exc)
    return ProviderSnapshot(provider, agent.model_name, agent.context_window_tokens, _signature(provider_name, _agent_signature(agent), asdict(provider_config)))


def _instantiate_provider(spec: Any, kwargs: dict[str, Any]) -> LLMProvider:
    """按 provider 后端类型创建具体适配器实例。"""

    if spec.backend == "openai_compat":
        # OpenAI 及大多数 OpenAI-compatible 网关都走统一适配器。
        return OpenAICompatProvider(spec=spec, **kwargs)
    if spec.backend == "anthropic":
        # Anthropic 原生协议与 Anthropic-compatible 网关共用同一适配器。
        return AnthropicProvider(spec=spec, **kwargs)
    raise ValueError(f"unsupported provider backend: {spec.backend}")


def _agent_signature(agent: AgentConfig) -> dict[str, Any]:
    return asdict(agent)


def _signature(provider_name: str, preset: dict[str, Any], provider_config: dict[str, Any]) -> str:
    """为当前 provider 装配结果生成稳定签名。

    Args:
        provider_name: 最终解析出的 provider 名称。
        preset: 序列化后的模型 preset 配置。
        provider_config: 序列化后的 provider 连接与鉴权配置。

    Returns:
        一个 SHA-256 十六进制摘要字符串。

    签名只覆盖会影响请求链路的关键字段。这样当模型、鉴权、base_url、生成参数
    或额外请求体发生变化时，外部系统可以快速识别“这个 provider 需要重建了”。
    """
    # 快照签名只包含影响请求链路的字段，用于之后做热刷新判断。
    raw = json.dumps(
        {"provider": provider_name, "preset": preset, "provider_config": provider_config},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
