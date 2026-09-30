from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

# 推理强度由模型配置决定，节点不再强制改成 medium，避免用户设置失效。
from src.llm import ModelConfig, ProviderSnapshot, SystemConfig, make_provider
from src.retrieval.hybrid import async_search_chunks, load_scoped_chunks, paper_scope
from src.utils.read_utils.cache import safe_cache_name
from src.utils.read_utils.chunkers import TextChunk, load_chunks_file

from .base import AgentContext, AgentSpec, BaseAgent
from .contracts import JsonObject
from .Prompts import WRITING_ABSTRACT_SYSTEM_PROMPT, WRITING_AGENT_SYSTEM_PROMPT, WRITING_REVIEW_SYSTEM_PROMPT


WritingAction = Literal["tool", "draft"]

# 中文说明：一次长综述会写很多小节。每节给出明确上限，避免模型反复检索、重写，
# 把同一批长证据重复发送几十次；达到上限时保留失败标记，不能冒充核查通过。
SECTION_TOKEN_BUDGET = 50000
SECTION_TIME_BUDGET_SECONDS = 480
WRITE_CALL_TIMEOUT_SECONDS = 120
# 中文说明：兼容节点的输出额度同时覆盖内部推理。第 23 轮 GCN 首稿
# 两次返回无法解析的 JSON；多留一些额度给完整的正文和逐句证据列表，
# 实际用量仍由供应商回执记录，并受每节总预算约束。
WRITE_MAX_TOKENS = 8192


class SectionLoopState(TypedDict, total=False):
    """单个小节写作循环内部使用的状态。

    中文注释：
    主工作流的 State 很大，里面有检索、阅读、分析、大纲等很多字段。
    单个小节写作只需要其中一小部分，所以这里单独放一个小状态，避免节点之间
    互相传一大包用不上的数据。
    """

    section_id: str
    task: str
    evidence_map: list[Any]
    previous_sections: list[JsonObject]
    word_count: int
    read_results: list[JsonObject]
    session_read_results: list[JsonObject]
    cache_dir: str
    available_paper_ids: list[str]
    tool_results: list[JsonObject]
    raw_model_outputs: list[str]
    revision_suggestions: list[str]
    draft: str
    cited_paper_ids: list[str]
    claim_evidence: list[JsonObject]
    action_type: WritingAction
    tool_name: str
    tool_arguments: JsonObject
    tool_call_count: int
    revision_count: int
    review: JsonObject
    completed: bool
    generation_failed: bool
    warnings: list[str]
    spent_tokens: int
    token_budget: int
    started_at: float
    # 中文说明：这是一个可选的界面通知函数，只把当前小节正在做什么告诉外层，
    # 不参与正文生成，也不会改变循环里的数据。
    progress_callback: Any


class WritingAgent(BaseAgent):
    """负责把大纲中的一个小节写成正文的 Agent。

    中文说明：
    这个 Agent 的核心不是“一次性让模型写完”，而是一个小循环：
    1. 模型先判断手头证据够不够；
    2. 不够就调用工具补充论文摘要或原文片段；
    3. 证据够了再写正文；
    4. 写完交给审查提示词检查逻辑和语言；
    5. 审查不通过就带着整改建议继续改。
    """

    spec = AgentSpec(
        name="writing_agent",
        role="write",
        description="根据写作大纲、论文证据和前置小节生成综述正文。",
        llm_profile="default_agent",
        tools=("get_extraction", "search_section", "get_chunk_by_embed"),
        skills=(),
        input_keys=("request", "writing_outline"),
    )

    def __init__(self, context: AgentContext):
        """初始化 WritingAgent，并保存当前 Agent 的固定配置。"""

        context.spec = self.spec
        super().__init__(context)
        self.max_tool_calls = 2
        # 中文说明：真实任务里一轮修改常常只能补齐一部分逐句引用。允许再改一次，
        # 但仍受单节 token 和时间上限约束；到上限时保留失败状态，绝不自动放行。
        self.max_revision_rounds = 2

    def _run(self, state: JsonObject) -> JsonObject:
        """BaseAgent 要求同步入口，但当前写作节点只使用异步入口。"""

        raise NotImplementedError("WritingAgent 请使用 async_write_section")

    async def async_write_section(
        self,
        *,
        section_id: str,
        task: str,
        evidence_map: list[Any],
        previous_sections: list[JsonObject],
        word_count: int,
        read_results: list[JsonObject],
        cache_dir: str,
        session_read_results: list[JsonObject] | None = None,
        available_paper_ids: list[str] | None = None,
        progress_callback: Any | None = None,
        token_budget: int | None = None,
    ) -> JsonObject:
        """写作单个小节，并返回正文、引用和审查结果。

        中文注释：
        外层写作节点会按大纲顺序逐节调用这个方法。这样“写完上一节再写下一节”
        的顺序很清楚，也方便当前小节读取已经完成的前置小节。
        """

        graph = _build_section_loop_graph(self)
        initial_state: SectionLoopState = {
            "section_id": section_id,
            "task": task,
            "evidence_map": list(evidence_map),
            "previous_sections": list(previous_sections),
            "word_count": max(100, int(word_count or 800)),
            "read_results": list(read_results),
            # 当前 State 只保存本轮结果，会话历史资料由写作节点单独传进来。
            "session_read_results": list(session_read_results or []),
            "cache_dir": cache_dir,
            "available_paper_ids": _deduplicate_strings(list(available_paper_ids or [])),
            "tool_results": [],
            "raw_model_outputs": [],
            "revision_suggestions": [],
            "draft": "",
            "cited_paper_ids": [],
            "claim_evidence": [],
            "action_type": "draft",
            "tool_name": "",
            "tool_arguments": {},
            "tool_call_count": 0,
            "revision_count": 0,
            "review": {},
            "completed": False,
            "warnings": [],
            "spent_tokens": 0,
            # 中文说明：多论文比较需核对多篇原文，可以由外层给出更高、但仍
            # 明确有上限的预算；普通单论文小节继续沿用五万 Token 上限。
            "token_budget": max(SECTION_TOKEN_BUDGET, int(token_budget or SECTION_TOKEN_BUDGET)),
            "started_at": time.monotonic(),
            "progress_callback": progress_callback,
        }
        final_state = await graph.ainvoke(initial_state)
        section_result = {
            "section_id": section_id,
            "task": task,
            "word_count": word_count,
            "content": str(final_state.get("draft") or "").strip(),
            "cited_paper_ids": _deduplicate_strings(list(final_state.get("cited_paper_ids") or [])),
            "claim_evidence": list(final_state.get("claim_evidence") or []),
            "tool_results": list(final_state.get("tool_results") or []),
            "review": dict(final_state.get("review") or {}),
            "revision_count": int(final_state.get("revision_count") or 0),
            "completed": bool(final_state.get("completed")),
            "generation_failed": bool(final_state.get("generation_failed")),
            "warnings": list(final_state.get("warnings") or []),
        }
        # 中文说明：结构化摘要里的来源标记和切片检索返回的 chunkId 都是“证据位置”，
        # 不能直接作为正文引用。这里在小节离开 Agent 前统一换成真正的 paperId，
        # 这样后面的参考文献解析只需要处理一种引用格式。
        return await asyncio.to_thread(
            normalize_writing_section_citations,
            section_result,
            read_results=list(read_results),
            session_read_results=list(session_read_results or []),
            cache_dir=Path(cache_dir),
        )

    async def async_write_abstract(
        self,
        *,
        topic: str,
        sections: list[JsonObject],
        word_count: int = 300,
        language: str = "zh",
        instruction: str = "",
        usage_callback: Any | None = None,
    ) -> tuple[str, str]:
        """根据已经完成的正文生成摘要。

        中文说明：摘要必须建立在最终正文之上，所以这个方法只在所有小节写完后调用。
        返回摘要文本和状态说明；模型不可用时仍返回一段根据正文拼出的保守摘要，保证
        写作产物结构完整。
        """

        if self.context.llm is None:
            return _fallback_abstract(topic, sections), "未配置摘要写作模型，已使用保守摘要"

        try:
            response = await self.context.llm.provider.chat(
                _abstract_messages(topic=topic, sections=sections, word_count=word_count,
                                   language=language, instruction=instruction),
                temperature=0.2,
            )
        except Exception as exc:
            return _fallback_abstract(topic, sections), f"摘要模型调用失败，已使用保守摘要：{exc}"

        self.report_usage(response, usage_callback)

        raw_output = str(getattr(response, "content", "") or "")
        if not getattr(response, "ok", False):
            return _fallback_abstract(topic, sections), f"摘要模型返回失败，已使用保守摘要：{raw_output}"

        parsed = _extract_json_object(raw_output)
        abstract = parsed.get("content") if parsed else None
        abstract = abstract.strip() if isinstance(abstract, str) else ""
        if not abstract:
            return _fallback_abstract(topic, sections), "摘要模型没有返回正文，已使用保守摘要"
        return abstract, "ok"

    async def plan_or_write(self, state: SectionLoopState) -> SectionLoopState:
        """让模型决定当前是补资料还是直接写正文。"""

        # 中文说明：工具节点可能因预算用尽而提前结束。此时直接返回，
        # 不再重复调用模型，也不重复写入同一条失败提示。
        if state.get("completed"):
            return state
        _notify_section_progress(state, "正在撰写小节正文")
        if _section_budget_exhausted(state):
            return _stop_writing(state, "本小节已达到 token 或时间预算，保留待核查草稿")
        if self.context.llm is None:
            return _stop_writing(state, "未配置可用的写作模型")

        messages = _write_messages(state)
        try:
            response = await asyncio.wait_for(self.context.llm.provider.chat(
                messages, temperature=0.2, max_tokens=WRITE_MAX_TOKENS,
            ), timeout=WRITE_CALL_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            # 中文说明：真实比较节偶发一次模型超时，旧流程立即保存为缺正文。
            # 只在本节总时间预算仍允许时，对完全相同的任务与证据补问一次；
            # 不把超时响应当正文，也不无限循环或偷偷追加来源。
            if _section_budget_exhausted(state):
                return _stop_writing(state, "写作模型超时且本小节预算已用尽")
            _notify_section_progress(state, "写作模型超时，正在重试同一请求")
            try:
                response = await asyncio.wait_for(self.context.llm.provider.chat(
                    messages, temperature=0, max_tokens=WRITE_MAX_TOKENS,
                ), timeout=WRITE_CALL_TIMEOUT_SECONDS)
                state = {**state, "warnings": [*list(state.get("warnings") or []), "写作模型首次调用超时，已同输入重试"]}
            except Exception as exc:
                return _stop_writing(state, f"写作模型超时重试失败：{type(exc).__name__}")
        except Exception as exc:
            return _stop_writing(state, f"写作模型未按时完成：{type(exc).__name__}")
        state = _record_section_usage(state, response)
        raw_output = str(getattr(response, "content", "") or "")
        raw_outputs = [*list(state.get("raw_model_outputs") or []), raw_output]
        if not response.ok:
            return _stop_writing(state, "写作模型调用失败", raw_outputs)

        parsed = _extract_json_object(raw_output)
        if parsed is None:
            # 中文说明：兼容节点偶尔会在成功 HTTP 响应中夹带解释文字或截断 JSON。
            # 不能把这种文本当正文交付；只以同一份状态重试一次，并把温度降到 0，
            # 让模型重新遵守既有协议。重试仍不合规时保留失败占位和原始输出。
            retry_messages = [
                *_write_messages(state),
                {
                    "role": "user",
                    "content": "上一响应不是可解析的单一 JSON 对象。请基于完全相同的输入重新作答，只输出一个符合既定 action 协议的 JSON 对象，不要解释、Markdown 或代码围栏。",
                },
            ]
            if _section_budget_exhausted(state):
                return _stop_writing(state, "首次输出格式错误，且小节预算已用尽", raw_outputs)
            try:
                retry = await asyncio.wait_for(self.context.llm.provider.chat(
                    retry_messages, temperature=0, max_tokens=WRITE_MAX_TOKENS,
                ), timeout=WRITE_CALL_TIMEOUT_SECONDS)
            except Exception as exc:
                return _stop_writing(state, f"写作格式重试失败：{type(exc).__name__}", raw_outputs)
            state = _record_section_usage(state, retry)
            retry_output = str(getattr(retry, "content", "") or "")
            raw_outputs.append(retry_output)
            if retry.ok:
                parsed = _extract_json_object(retry_output)
        action = str((parsed or {}).get("action") or "").strip().lower()
        if action == "tool" and int(state.get("tool_call_count") or 0) < self.max_tool_calls:
            return {
                **state,
                "action_type": "tool",
                "tool_name": str(parsed.get("tool_name") or "").strip(),
                "tool_arguments": dict(parsed.get("arguments") or {}) if isinstance(parsed.get("arguments"), dict) else {},
                "raw_model_outputs": raw_outputs,
            }

        if (action != "draft" or not isinstance((parsed or {}).get("content"), str)
                or not parsed["content"].strip()):
            # 中文说明：工具用满或两次回复都不是合法 JSON 时，最多补问
            # 一次现有资料下的正文。第 23 轮 GCN 尚剩工具次数却在两次
            # 格式错误后直接缺正文；这次补问仍须返回有效原文绑定，
            # 否则保留失败占位，绝不把自由文本冒充综述。
            if _section_budget_exhausted(state):
                return _stop_writing(state, "输出格式或资料工具异常，且小节预算不足以补写正文", raw_outputs)
            _notify_section_progress(state, "正在根据已有资料补问正文")
            try:
                forced = await asyncio.wait_for(self.context.llm.provider.chat(
                    [*_write_messages(state), {
                        "role": "user",
                        "content": "现在请停止请求工具，只根据已经取得的资料返回 action=draft 的单个合法 JSON。证据不足的要点请删去或说明边界，不要编造事实；content 必须是非空的本小节正文，每句仍须有真实原文切片绑定。",
                    }], temperature=0, max_tokens=WRITE_MAX_TOKENS,
                ), timeout=WRITE_CALL_TIMEOUT_SECONDS)
            except Exception as exc:
                return _stop_writing(state, f"补问正文模型调用失败：{type(exc).__name__}", raw_outputs)
            state = _record_section_usage(state, forced)
            forced_output = str(getattr(forced, "content", "") or "")
            raw_outputs.append(forced_output)
            parsed = _extract_json_object(forced_output) if forced.ok else None
            action = str((parsed or {}).get("action") or "").strip().lower()

        # 工具响应、空对象和其他协议内容不能作为论文正文交付。
        if action != "draft" or not isinstance((parsed or {}).get("content"), str) or not parsed["content"].strip():
            reason = "有界补问后仍未返回有效正文"
            return _stop_writing(state, reason, raw_outputs)
        draft = parsed["content"].strip()
        paper_ids = _deduplicate_strings(
            [
                *_string_list(parsed.get("paperIds")),
                *_string_list(parsed.get("paper_ids")),
                # 中文说明：工具原文可能包含论文自己的参考文献编号，如 [41]。
                # 这些编号不是本小节实际引用的论文，不能从 evidence_map 或工具返回中
                # 收集；只相信模型声明的 paperIds 和正文中实际写出的引用标记。
                *_paper_ids_from_any([draft]),
            ]
        )
        warnings = list(state.get("warnings") or [])
        return {
            **state,
            "action_type": "draft",
            "draft": draft,
            "cited_paper_ids": paper_ids,
            "claim_evidence": [item for item in parsed.get("evidence", []) if isinstance(item, dict)] if isinstance(parsed.get("evidence"), list) else [],
            "raw_model_outputs": raw_outputs,
            "warnings": warnings,
        }

    async def run_tool(self, state: SectionLoopState) -> SectionLoopState:
        """执行模型请求的工具，并把结果放回循环状态。"""

        if _section_budget_exhausted(state):
            return _stop_writing(state, "本小节达到预算，停止继续检索")
        result = await execute_writing_tool(
            tool_name=str(state.get("tool_name") or ""),
            arguments=dict(state.get("tool_arguments") or {}),
            read_results=list(state.get("read_results") or []),
            session_read_results=list(state.get("session_read_results") or []),
            cache_dir=Path(str(state.get("cache_dir") or "data/paper_cache")),
        )
        if str(state.get("tool_name") or "") == "search_section":
            missing_ids = _deduplicate_strings([
                str(item.get("paperId") or "")
                for item in (result.get("result") or [])
                if isinstance(item, dict) and item.get("status") == "missing_chunk_ids"
            ])
            query = str(state.get("task") or "").split("\n", 1)[0].strip()[:500]
            if missing_ids and query:
                # 中文说明：模型给出的切片编号可能写错或已经失效。
                # 直接在同一批论文全文中按小节任务重新定位，算作本次工具调用的
                # 降级结果，不再消耗下一次工具机会，也不把错误编号当成证据。
                _notify_section_progress(state, "切片编号无效，改用全文检索定位原文")
                try:
                    result["fallback"] = await get_chunk_by_embed(
                        query, paper_ids=missing_ids,
                        read_results=list(state.get("read_results") or []),
                        session_read_results=list(state.get("session_read_results") or []),
                        cache_dir=Path(str(state.get("cache_dir") or "data/paper_cache")),
                        top_k=3,
                    )
                except Exception as exc:
                    result["fallback_error"] = f"全文检索降级失败：{type(exc).__name__}"
        # 中文说明：查询向量也可能消耗 token，把服务实际返回的用量计入当前小节。
        search_result = result.get("fallback") or result.get("result")
        usage = dict(search_result.get("diagnostics", {}).get("dense", {}).get("usage") or {}) if isinstance(search_result, dict) else {}
        callback = state.get("progress_callback")
        if usage and callable(callback):
            callback("全文检索完成", usage)
        return {
            **state,
            "tool_results": [*list(state.get("tool_results") or []), result],
            "tool_call_count": int(state.get("tool_call_count") or 0) + 1,
            "spent_tokens": int(state.get("spent_tokens") or 0) + int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0),
            "tool_name": "",
            "tool_arguments": {},
        }

    async def review_draft(self, state: SectionLoopState) -> SectionLoopState:
        """审查正文是否逻辑通顺、语言是否足够学术化。"""

        _notify_section_progress(state, "正在审查写作内容")
        draft = str(state.get("draft") or "").strip()
        if _section_budget_exhausted(state):
            return _stop_writing(state, "本小节达到预算，未完成文风审查")
        if self.context.llm is None:
            return {
                **state,
                "review": {"passed": False, "status": "unverified", "suggestions": [], "message": "未配置审查模型，正文未验证。"},
                "completed": True,
            }
        if not draft:
            return {
                **state,
                "review": {"passed": False, "suggestions": ["正文为空，需要先生成小节正文。"]},
                "completed": False,
            }

        # 先检查可由程序确定的引用格式和证据位置，避免文风模型把漏引文的正文判为通过。
        # 这里只检查编号和原句绑定；是否真正支持结论仍由后面的独立事实审计判断。
        located = await asyncio.to_thread(
            normalize_writing_section_citations,
            {"content": draft, "claim_evidence": list(state.get("claim_evidence") or [])},
            read_results=list(state.get("read_results") or []),
            session_read_results=list(state.get("session_read_results") or []),
            cache_dir=Path(str(state.get("cache_dir") or "data/paper_cache")),
        )
        problems = []
        target_words = int(state.get("word_count") or 0)
        actual_words = _approximate_word_count(draft)
        if target_words > 0 and actual_words > int(target_words * 1.3):
            problems.append(
                f"正文约 {actual_words} 字/词，超过计划的 {target_words} 字/词；"
                "请保留任务必需的数字、条件和逐句引用，压缩至目标字数附近。"
            )
        for sentence in _citation_sentences(located["content"]):
            # 中文说明：仅看到少量片段，不能证明整篇原论文没有某项结果。前两次
            # 真实终审均查出“未提供表格/数字”“唯一有指标”等错误断言。写作端
            # 无法自动证明全文不存在某事，所以遇到这种句子就要求删除或改写成
            # 已有原文明确支持的正面事实，不能靠附一个论文编号直接放行。
            # 中文说明：“未见节点”“未见过的数据”是 GraphSAGE 的归纳任务，
            # 不是在断言原论文缺少数据。只把明确说论文没有报告某项材料的
            # 短语拦下；“未见”必须紧跟被声称缺少的材料名，不能隔着整句找。
            if (re.search(r"(?:未提供|未报告|未列出|缺少|缺失|没有).{0,40}"
                          r"(?:数据|指标|数值|结果|表格|曲线|证据|实验|模型|参数|计算量)", sentence)
                    or re.search(r"未见(?:相关|明确|足够)?(?:数据|指标|数值|结果|表格|曲线|证据|实验|模型|参数|计算量)", sentence)
                    or re.search(r"唯一.{0,50}(?:证据|指标|结果|实验)", sentence)):
                problems.append(f"事实句“{sentence[:50]}”断言论文缺少证据或仅某篇有证据；请删除该断言，仅写已逐项核实的结果。")
            # 中文说明：真实 GCN 任务把推导中的单个标量写成最终网络只有一个
            # 参数，而原论文后续公式已经改用权重矩阵。凡是概括“模型只有一个
            # 参数”的句子，要求明确核对最终模型公式；核对不到就删掉该概括。
            if re.search(r"(?:模型|网络).{0,35}(?:仅|只).{0,10}(?:单一|一个).{0,5}参数", sentence):
                problems.append(f"事实句“{sentence[:50]}”可能把推导特例当成最终模型参数量；请核对最终公式，不能证明就删去该概括。")
            # 科研小节中的分析结论同样来自论文证据。统一要求逐句引用，避免模型只在段末
            # 放一个编号，让读者无法判断它究竟支撑哪一个数字或比较。
            if len(re.sub(r"\s+", "", sentence)) >= 15 and not re.search(r"\[[^\[\]\n]+\]", sentence):
                problems.append(f"事实或分析句“{sentence[:40]}”缺少逐句引用。")
            # 方括号本身不能证明引用有效；逐个编号核对本句的真实证据，防止未知编号
            # 或其他句子的有效引用掩盖当前句缺少来源的问题。
            for marker in re.findall(r"\[([^\[\]\n]+)\]", sentence):
                if _is_numeric_interval_marker(marker):
                    continue
                for paper_id in re.split(r"[,;，；]\s*", marker):
                    if not any(
                        str(binding.get("paperId") or "").casefold() == paper_id.strip().casefold()
                        and str(binding.get("claim") or "") in sentence
                        and binding.get("chunks")
                        for binding in located.get("citation_evidence") or []
                    ):
                        problems.append(f"本句引用 [{paper_id.strip()}] 没有有效的原文切片绑定。")
        if not located.get("citation_evidence"):
            problems.append("缺少有效 evidence：每个引用的 paperId 都必须提供属于该论文的真实 chunkIds。")
        if located.get("invalid_citation_evidence"):
            problems.append("存在无效证据绑定，请修正论文编号和切片归属。")
        for binding in located.get("citation_evidence") or []:
            claim = str(binding.get("claim") or "")
            marker = f"[{binding['paperId']}]"
            start = located["content"].find(claim)
            # 引用必须紧跟在它支撑的事实后面；只在本段其他位置放一次编号，不能覆盖所有事实。
            nearby = located["content"][start + len(claim):start + len(claim) + 80] if start >= 0 else ""
            if not _text_cites_paper(claim + nearby, str(binding["paperId"])):
                problems.append(f"事实“{claim[:40]}”后缺少 {marker} 引用；仅在正文其他位置或 evidence 中写编号不够。")
        if problems:
            exhausted = int(state.get("revision_count") or 0) >= self.max_revision_rounds
            return {
                **state,
                "review": {"passed": False, "suggestions": problems, "message": "引用格式或证据绑定未通过"},
                "revision_suggestions": problems,
                "revision_count": int(state.get("revision_count") or 0) + (0 if exhausted else 1),
                "completed": exhausted,
                # 中文说明：正文已经存在，只是引用检查没过；不能说正文生成失败，
                # 否则后续独立核查会把这段真实正文整个跳过。
                "generation_failed": False,
            }

        try:
            response = await asyncio.wait_for(self.context.llm.provider.chat(
                _review_messages(state, located), temperature=0, max_tokens=2048,
            ), timeout=WRITE_CALL_TIMEOUT_SECONDS)
        except Exception as exc:
            return _stop_writing(state, f"审查模型未按时完成：{type(exc).__name__}")
        state = _record_section_usage(state, response)
        raw_output = str(getattr(response, "content", "") or "")
        raw_outputs = [*list(state.get("raw_model_outputs") or []), raw_output]
        if not response.ok:
            return {
                **state,
                "raw_model_outputs": raw_outputs,
                "review": {"passed": False, "status": "unverified", "suggestions": [], "message": "审查模型调用失败，正文未验证。"},
                "completed": True,
                "warnings": [*list(state.get("warnings") or []), f"审查模型调用失败：{raw_output}"],
            }

        parsed = _extract_json_object(raw_output) or {}
        passed = parsed.get("passed") is True
        suggestions = _string_list(parsed.get("suggestions"))
        if passed or int(state.get("revision_count") or 0) >= self.max_revision_rounds:
            message = str(parsed.get("message") or ("审查通过" if passed else "修改次数已达到上限，保留当前正文")).strip()
            return {
                **state,
                "raw_model_outputs": raw_outputs,
                "review": {"passed": passed, "suggestions": suggestions, "message": message},
                "completed": True,
            }
        return {
            **state,
            "raw_model_outputs": raw_outputs,
            "review": {"passed": False, "suggestions": suggestions, "message": str(parsed.get("message") or "")},
            "revision_suggestions": suggestions,
            "revision_count": int(state.get("revision_count") or 0) + 1,
            "completed": False,
        }


def _build_section_loop_graph(agent: WritingAgent):
    """构建单个小节内部的 LangGraph 循环。"""

    workflow = StateGraph(SectionLoopState)
    workflow.add_node("plan_or_write", agent.plan_or_write)
    workflow.add_node("run_tool", agent.run_tool)
    workflow.add_node("review_draft", agent.review_draft)
    workflow.add_edge(START, "plan_or_write")
    workflow.add_conditional_edges(
        "plan_or_write",
        _route_after_write_step,
        {"run_tool": "run_tool", "review_draft": "review_draft", "end": END},
    )
    workflow.add_edge("run_tool", "plan_or_write")
    workflow.add_conditional_edges(
        "review_draft",
        _route_after_review_step,
        {"plan_or_write": "plan_or_write", "end": END},
    )
    return workflow.compile(name="writing_section_loop")


def _approximate_word_count(text: str) -> int:
    """按汉字逐字、英文数字词组逐词估算篇幅，忽略引用编号。"""
    without_citations = re.sub(r"\[[^\[\]\n]+\]", "", text)
    return len(re.findall(r"[\u4e00-\u9fff]", without_citations)) + len(
        re.findall(r"[A-Za-z0-9]+(?:[-.][A-Za-z0-9]+)*", without_citations)
    )


def _route_after_write_step(state: SectionLoopState) -> str:
    """根据模型动作决定下一步是调用工具还是进入审查。"""

    if state.get("completed"):
        return "end"
    return "run_tool" if state.get("action_type") == "tool" else "review_draft"


def _route_after_review_step(state: SectionLoopState) -> str:
    """审查通过就结束；不通过就回到写作节点继续修改。"""

    return "end" if state.get("completed") else "plan_or_write"


def _notify_section_progress(state: SectionLoopState, message: str) -> None:
    """把小节当前阶段交给外层；没有回调时保持原来的静默行为。"""

    callback = state.get("progress_callback")
    if callable(callback):
        callback(message)


def _record_section_usage(state: SectionLoopState, response: object) -> SectionLoopState:
    """累计供应商实际返回的用量，并同步到界面。"""

    from src.llm.base import normalize_token_usage

    usage = normalize_token_usage(getattr(response, "usage", None))
    callback = state.get("progress_callback")
    if callable(callback):
        callback("模型调用完成", usage)
    return {**state, "spent_tokens": int(state.get("spent_tokens") or 0)
            + int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)}


def _section_budget_exhausted(state: SectionLoopState) -> bool:
    """超过本节用量或用时上限时，停止下一次付费模型调用。"""

    return (int(state.get("spent_tokens") or 0) >= int(state.get("token_budget") or SECTION_TOKEN_BUDGET)
            or time.monotonic() - float(state.get("started_at") or time.monotonic()) >= SECTION_TIME_BUDGET_SECONDS)


def get_extraction(
    paper_ids: list[str],
    *,
    read_results: list[JsonObject] | None = None,
    session_read_results: list[JsonObject] | None = None,
    cache_dir: str | Path = "data/paper_cache",
) -> list[JsonObject]:
    """根据 paperId 获取论文结构化摘要。

    中文注释：
    阅读节点如果已经把 extraction 放在 State 里，就优先读 State，因为这是本轮
    工作流最新的数据。State 里没有时，再读当前会话所有轮次的阅读产物；最后才
    去 data/paper_cache 里的 extraction.json 找，保证单独从缓存恢复写作时也能拿到资料。
    """

    results: list[JsonObject] = []
    for paper_id in _deduplicate_strings(paper_ids):
        extraction = _find_extraction_in_read_results(paper_id, list(read_results or []))
        source = "state"
        if extraction is None:
            extraction = _find_extraction_in_session_read(paper_id, list(session_read_results or []))
            source = "session_artifacts_read"
        if extraction is None:
            extraction = _find_extraction_in_cache(paper_id, Path(cache_dir))
            source = "paper_cache"
        if extraction is None:
            results.append({"paperId": paper_id, "status": "missing", "extraction": {}, "source": ""})
        else:
            results.append({"paperId": paper_id, "status": "ok", "extraction": extraction, "source": source})
    return results


def search_section(
    requests: list[JsonObject],
    *,
    cache_dir: str | Path = "data/paper_cache",
) -> list[JsonObject]:
    """根据 paperId 和 chunkId 获取论文原文片段。

    中文注释：
    这个工具不做复杂搜索，只做“按编号取原文”。模型如果已经知道需要哪几个
    chunkId，就可以用它把 chunk.json 里的原文片段取出来。
    """

    found: list[JsonObject] = []
    for item in requests:
        if not isinstance(item, dict):
            continue
        paper_id = str(item.get("paperId") or item.get("paper_id") or "").strip()
        chunk_ids = _string_list(item.get("chunkIds") or item.get("chunk_ids") or item.get("chunkId") or item.get("chunk_id"))
        if not paper_id or not chunk_ids:
            found.append({"paperId": paper_id, "status": "invalid_arguments", "chunks": []})
            continue
        chunks_path = _find_chunks_path(paper_id, Path(cache_dir))
        if chunks_path is None:
            found.append({"paperId": paper_id, "status": "missing_chunks", "chunks": []})
            continue
        chunks_by_id = {chunk.chunk_id: chunk for chunk in load_chunks_file(chunks_path)}
        selected = [_chunk_to_markdown(chunks_by_id[chunk_id]) for chunk_id in chunk_ids if chunk_id in chunks_by_id]
        found.append({"paperId": paper_id, "status": "ok" if selected else "missing_chunk_ids", "chunks": selected})
    return found


def normalize_writing_section_citations(
    section: JsonObject,
    *,
    read_results: list[JsonObject],
    session_read_results: list[JsonObject],
    cache_dir: Path,
) -> JsonObject:
    """把小节里可能出现的切片引用统一改成对应的论文编号。

    中文说明：全文结构化摘要会把证据位置写成 `[论文编号:p0001]`，切片工具返回的
    结果还会同时出现 `paperId` 和 `chunkId`。模型有时会把后者误写进正文，因此这里
    统一查找“切片编号 -> 论文编号”的关系，再处理正文和 cited_paper_ids 两个字段。
    """

    normalized = dict(section)
    source_content = str(normalized.get("content") or "")
    chunk_to_paper = _build_chunk_to_paper_map(
        [*read_results, *session_read_results, section.get("tool_results") or []],
        cache_dir=cache_dir,
    )
    content = _replace_chunk_citations(str(normalized.get("content") or ""), chunk_to_paper)
    cited_paper_ids: list[str] = []
    for value in list(normalized.get("cited_paper_ids") or []):
        paper_id = _resolve_citation_paper_id(str(value or ""), chunk_to_paper)
        if paper_id:
            cited_paper_ids.append(paper_id)

    normalized["content"] = content.strip()
    normalized["cited_paper_ids"] = _deduplicate_strings(cited_paper_ids)
    # 中文说明：正文仍使用论文编号供阅读，但另存原始正文和逐条证据位置。
    # 找到原文只表示来源有效，不代表已经通过事实蕴含审查。
    normalized["source_content"] = source_content
    scoped_chunks = {chunk.chunk_id: chunk for chunk in load_scoped_chunks(cache_dir, [*read_results, *session_read_results])}
    located: list[JsonObject] = []
    invalid: list[JsonObject] = []
    # 中文说明：模型写跨论文比较时常用分号分隔各论文的独立事实，且在 evidence
    # 里复制分句时省略末尾的句号或分号。忽略这些纯标点差异即可精确对应原句，
    # 不能因此把一个分句的原文证据借给同段其他分句。
    sentences = {sentence.rstrip("。！？!?；;").strip() for sentence in _citation_sentences(content)}
    for item in normalized.pop("claim_evidence", []):
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim") or "").strip()
        paper_id = str(item.get("paperId") or "").strip()
        chunk_ids = _string_list(item.get("chunkIds"))
        # 中文说明：旧的“每篇论文一组切片”会把同一组原文自动贴到这篇论文的
        # 所有句子上，令无依据的新主张也显示为已有切片。现在必须由模型逐句
        # 指明完整事实句；句子不存在或句中没有引用该论文时，不建立有效绑定。
        if not claim or claim.rstrip("。！？!?；;").strip() not in sentences:
            invalid.append({"claim": claim, "paperId": paper_id, "chunkIds": chunk_ids,
                            "reason": "证据主张必须是正文中含引用的完整事实句"})
            continue
        if not _text_cites_paper(claim, paper_id):
            invalid.append({"claim": claim, "paperId": paper_id, "chunkIds": chunk_ids,
                            "reason": "证据主张的句子没有引用这篇论文"})
            continue
        if not paper_id or not chunk_ids:
            invalid.append({"claim": claim, "paperId": paper_id, "reason": "证据缺少论文编号或切片编号"})
            continue
        valid_chunks = []
        for value in chunk_ids:
            # 中文说明：模型偶尔照抄了切片编号的后半段，漏掉前面的论文编号。
            # 只尝试补上当前 evidence 指定的同一篇论文编号，并要求补全后精确命中
            # 当前会话已有的原文切片；找不到时仍判为无效，不能模糊匹配或猜证据。
            full_id = value if value in scoped_chunks else f"{paper_id}:{value}"
            chunk = scoped_chunks.get(full_id)
            if chunk is not None and chunk.paperId.casefold() == paper_id.casefold():
                valid_chunks.append(chunk)
        if len(valid_chunks) != len(chunk_ids):
            invalid.append({"claim": claim, "paperId": paper_id, "chunkIds": chunk_ids, "reason": "切片不存在、超出会话范围或不属于这篇论文"})
            continue
        located.append({"claim": claim, "paperId": paper_id, "status": "source_located",
                        "binding_source": "model_claim", "chunks": [chunk.to_dict() for chunk in valid_chunks]})
    normalized["citation_evidence"] = located
    normalized["invalid_citation_evidence"] = invalid
    return normalized


def _citation_sentences(content: str) -> list[str]:
    """按常见中英文句末符号拆分正文，供逐句引用检查和证据绑定共同使用。"""

    raw_parts = [
        part.strip()
        # 中文说明：分号既能隔开两个事实，也能出现在 [P1;P2] 这样的合并引用里。
        # 只拆分引用方括号之外的分号，否则会把一个真实编号拆成两段假句子。
        for part in re.split(r"(?<=[。！？!?])\s*|(?<=[；;])(?![^\[\]]*\])\s*|(?<=\.)\s+|\n+", str(content or ""))
        if part.strip()
    ]
    sentences: list[str] = []
    leading_markers = re.compile(r"^(?P<markers>(?:\[[^\[\]\n]+\]\s*)+)(?P<rest>.*)$", re.DOTALL)
    for part in raw_parts:
        match = leading_markers.match(part)
        # 模型常写成“事实句。[P1]下一句”。句末编号在分句后看似位于下一段，
        # 实际仍属于前一句；先移回前句，避免误报漏引，同时不让它替下一句兜底。
        if match and sentences:
            markers = str(match.group("markers") or "").strip()
            sentences[-1] = f"{sentences[-1]}{markers}"
            remainder = str(match.group("rest") or "").strip()
            if remainder:
                sentences.append(remainder)
            continue
        sentences.append(part)
    return sentences


def _text_cites_paper(text: str, paper_id: str) -> bool:
    """识别 `[P1]` 与 `[P1,P2]` 两种写法，避免把同一句的合并引用误判为缺失。"""

    expected = str(paper_id or "").strip().casefold()
    return any(
        value.strip().casefold() == expected
        for marker in re.findall(r"\[([^\[\]\n]+)\]", str(text or ""))
        if not _is_numeric_interval_marker(marker)
        for value in re.split(r"[,;，；]\s*", marker)
    )


def _is_numeric_interval_marker(marker: str) -> bool:
    """识别从零开始的数字区间，避免把公式中的 `[0, 2]` 当成论文引用。"""

    # 中文说明：参考文献序号从 1 开始；“[0, 2]”是谱算子的
    # 数值区间，不是第 0 与第 2 篇论文。只跳过这类明确从 0
    # 开始的双端数字区间，其余未知方括号仍按无效引用拦截。
    return bool(re.fullmatch(r"\s*0(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*", marker))


def _build_chunk_to_paper_map(payloads: list[Any], *, cache_dir: Path) -> dict[str, str]:
    """从工具结果、阅读结果和本地切片缓存建立编号映射。"""

    mapping: dict[str, str] = {}
    _collect_chunk_mappings(payloads, mapping)

    # 中文说明：结构化摘要里只有正文中的 chunkId，未必会把 chunks_used 一起传给写作 Agent。
    # 因此再根据结果里出现过的 paperId 读取对应缓存，补齐摘要引用所需的映射。
    paper_ids = _collect_payload_paper_ids(payloads)
    for paper_id in paper_ids:
        chunks_path = _find_chunks_path(paper_id, cache_dir)
        if chunks_path is None:
            continue
        for chunk in load_chunks_file(chunks_path):
            _register_chunk_mapping(mapping, chunk.chunk_id, chunk.paperId or paper_id)
    return mapping


def _collect_chunk_mappings(value: Any, mapping: dict[str, str], inherited_paper_id: str = "") -> None:
    """递归读取 payload 中显式提供的 paperId、chunkId 和 chunks_used。"""

    if isinstance(value, dict):
        nested_paper = value.get("paper")
        nested_paper_id = ""
        if isinstance(nested_paper, dict):
            nested_paper_id = str(nested_paper.get("paperId") or nested_paper.get("id") or "").strip()
        paper_id = str(value.get("paperId") or value.get("paper_id") or nested_paper_id or inherited_paper_id).strip()
        for key, item in value.items():
            normalized_key = str(key)
            if normalized_key in {"chunkId", "chunk_id"}:
                _register_chunk_mapping(mapping, str(item or ""), paper_id)
                continue
            if normalized_key in {"chunkIds", "chunk_ids", "chunks_used"}:
                for chunk_id in _string_list(item):
                    _register_chunk_mapping(mapping, chunk_id, paper_id)
                continue
            _collect_chunk_mappings(item, mapping, paper_id)
        return
    if isinstance(value, list):
        for item in value:
            _collect_chunk_mappings(item, mapping, inherited_paper_id)


def _collect_payload_paper_ids(value: Any) -> list[str]:
    """只从结构化字段收集真实 paperId，不把正文里的方括号内容当成编号。"""

    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = str(key)
            if normalized_key in {"paperId", "paper_id"}:
                text = str(item or "").strip()
                if text:
                    found.append(text)
            elif normalized_key in {"paperIds", "paper_ids"}:
                found.extend(_string_list(item))
            else:
                found.extend(_collect_payload_paper_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_collect_payload_paper_ids(item))
    return _deduplicate_strings(found)


def _register_chunk_mapping(mapping: dict[str, str], chunk_id: str, paper_id: str) -> None:
    """登记完整 chunkId，并登记去掉末尾分段号后的短格式。"""

    chunk_text = str(chunk_id or "").strip()
    paper_text = str(paper_id or "").strip()
    if not chunk_text or not paper_text:
        return
    key = chunk_text.lower()
    mapping.setdefault(key, paper_text)
    # 中文说明：切片通常形如 `paper:p0001:s0001`，模型在摘要中常会省略最后的分段号，
    # 所以 `paper:p0001` 也必须能映射回同一篇论文。
    short_key = re.sub(r":s\d+$", "", key, flags=re.IGNORECASE)
    mapping.setdefault(short_key, paper_text)


def _resolve_citation_paper_id(value: str, chunk_to_paper: dict[str, str]) -> str:
    """把一个引用候选值解析为论文编号；无法解析时保留原值。"""

    text = str(value or "").strip().strip('"').strip("'").strip()
    if not text:
        return ""
    key = text.lower()
    mapped = chunk_to_paper.get(key)
    if mapped:
        return mapped
    # 中文说明：如果缓存中暂时没有对应 chunk.json，切片编号仍然保留了
    # `paperId:p0001` 的前缀。直接取前缀可以避免把切片编号继续传播到正文。
    prefix_match = re.match(r"^(?P<paper>.+):(?:p|c)\d{4}(?::s\d{4})?$", text, flags=re.IGNORECASE)
    return prefix_match.group("paper").strip() if prefix_match else text


def _replace_chunk_citations(content: str, chunk_to_paper: dict[str, str]) -> str:
    """只替换能确认是切片编号的方括号内容，避免破坏普通 Markdown。"""

    if not content:
        return content

    citation_pattern = re.compile(r"\[([^\[\]\r\n]+)\]")

    def replace(match: re.Match[str]) -> str:
        candidate = match.group(1).strip().strip('"').strip("'").strip()
        resolved = _resolve_citation_paper_id(candidate, chunk_to_paper)
        return f"[{resolved}]" if resolved != candidate else match.group(0)

    return citation_pattern.sub(replace, content)


async def get_chunk_by_embed(
    query: str,
    paper_ids: list[str] | None = None,
    *,
    read_results: list[JsonObject] | None = None,
    session_read_results: list[JsonObject] | None = None,
    cache_dir: str | Path = "data/paper_cache",
    top_k: int | None = None,
) -> JsonObject:
    """在当前会话全文中执行 BM25 与向量检索，返回子片段和父级上下文。"""

    return await async_search_chunks(
        query,
        read_results=[*list(read_results or []), *list(session_read_results or [])],
        cache_dir=cache_dir,
        paper_ids=paper_ids,
        top_k=top_k,
    )


async def execute_writing_tool(
    *,
    tool_name: str,
    arguments: JsonObject,
    read_results: list[JsonObject],
    session_read_results: list[JsonObject],
    cache_dir: Path,
) -> JsonObject:
    """执行写作工具；所有工具的论文范围都由会话阅读记录决定。"""

    allowed = {value.casefold() for value in paper_scope([*read_results, *session_read_results])}

    if tool_name == "get_extraction":
        paper_ids = _string_list(arguments.get("paperIds") or arguments.get("paper_ids") or arguments.get("paperId"))
        if not paper_ids or any(value.casefold() not in allowed for value in paper_ids):
            return {"tool": tool_name, "result": [], "message": "请求包含未获当前会话阅读记录确认的论文编号"}
        return {
            "tool": tool_name,
            "arguments": {"paperIds": paper_ids},
            "result": await asyncio.to_thread(
                get_extraction,
                paper_ids,
                read_results=read_results,
                session_read_results=session_read_results,
                cache_dir=cache_dir,
            ),
        }
    if tool_name == "search_section":
        requests = arguments.get("requests") or arguments.get("sections") or arguments.get("items") or []
        if not isinstance(requests, list):
            requests = []
        if not requests or any(not isinstance(item, dict) or str(item.get("paperId") or item.get("paper_id") or "").casefold() not in allowed for item in requests):
            return {"tool": tool_name, "result": [], "message": "请求包含未获当前会话阅读记录确认的论文编号"}
        return {"tool": tool_name, "arguments": {"requests": requests}, "result": await asyncio.to_thread(search_section, requests, cache_dir=cache_dir)}
    if tool_name == "get_chunk_by_embed":
        query = str(arguments.get("query") or "").strip()
        paper_ids = _string_list(arguments.get("paper_ids") or arguments.get("paperIds"))
        result = await get_chunk_by_embed(
            query, paper_ids=paper_ids, read_results=read_results,
            session_read_results=session_read_results, cache_dir=cache_dir, top_k=arguments.get("top_k"),
        )
        return {"tool": tool_name, "arguments": {"query": query, "paper_ids": paper_ids}, "result": result}
    return {"tool": tool_name, "arguments": arguments, "result": None, "message": "未知工具，未执行"}


def load_writing_agent_llm(
    agent_name: str | None = None,
    model_config_path: str | Path = "config/model.json",
    system_config_path: str | Path = "config/system.yaml",
    *,
    client: Any | None = None,
) -> ProviderSnapshot | None:
    """从本地模型配置里装配正文写作 Agent 使用的模型。"""

    model_path = Path(model_config_path)
    if not model_path.exists():
        return None
    try:
        data = json.loads(model_path.read_text(encoding="utf-8"))
        system = SystemConfig.load(system_config_path)
        config = ModelConfig.from_dict(data, system)
        resolved_agent_name = agent_name or WritingAgent.spec.llm_profile
        return make_provider(config, resolved_agent_name, client=client)
    except Exception:
        # 中文注释：配置读取失败时返回 None，让写作节点可以生成保守草稿，不让流程直接崩掉。
        return None


def build_writing_agent(llm: ProviderSnapshot | None | str = "auto") -> WritingAgent:
    """构建一个可直接使用的 WritingAgent。"""

    resolved_llm = load_writing_agent_llm() if llm == "auto" else llm
    context = AgentContext(spec=WritingAgent.spec, llm=resolved_llm)
    return WritingAgent(context)


def _write_messages(state: SectionLoopState) -> list[JsonObject]:
    """构造写作提示词，让模型用 JSON 表达下一步动作。"""

    # 中文说明：每轮写作都复用统一规则，确保工具调用和正文输出格式稳定。
    system_prompt = WRITING_AGENT_SYSTEM_PROMPT
    # 中文说明：原先每轮都带上全部矩阵、前文和所有工具全文，模型调用越往后越贵。
    # 这里保留当前小节的任务、来源编号以及最近两次工具证据；截断只影响模型可见
    # 的线索，不会改写本地原文，最终引用仍必须通过逐句证据审计。
    evidence = [{"field": str(item.get("field") or item.get("全局分析字段") or ""),
                 "content": _compact_text(item.get("content") or item.get("内容") or "",
                                          max_chars=4000 if item.get("field") == "实证矩阵的原文位置（不是事实结论）" else 1800)}
                for item in (state.get("evidence_map") or []) if isinstance(item, dict)][:5]
    previous = [{"section_id": item.get("section_id"),
                 "content": _compact_text(item.get("content") or "", max_chars=1000)}
                for item in (state.get("previous_sections") or []) if isinstance(item, dict)][-3:]
    tool_results = [_compact_writing_tool_result(item) for item in (state.get("tool_results") or [])[-2:]]
    user_prompt = json.dumps(
        {
            "section_id": state.get("section_id"),
            "小节任务": str(state.get("task") or "")[:4500],
            "计划字数": state.get("word_count"),
            "上游分析的待核线索（不是原文证据）": evidence,
            "已经写好的前置小节": previous,
            "已调用工具得到的资料": tool_results,
            "允许引用的真实论文编号": state.get("available_paper_ids") or [],
            "当前草稿": str(state.get("draft") or "")[:6000],
            # 中文说明：同一篇论文漏切片会在多句中反复出现，先去重再传给模型。
            # 过去只取前五条，后面的错误类型根本没有进入修订输入。
            "审查整改建议": [_compact_text(item, max_chars=350)
                         for item in _deduplicate_strings(list(state.get("revision_suggestions") or []))[:20]],
            "工具次数要求": "最多调用 2 次工具；现已调用 %d 次，达到上限后只能返回 draft。" % int(state.get("tool_call_count") or 0),
            "可用工具": [
                "get_extraction(List[paperId])：获取论文结构化摘要",
                "search_section(List[{paperId, List[chunkId]}])：按 chunkId 获取论文原文片段",
                "get_chunk_by_embed(query, paper_ids=None)：在当前会话论文中混合检索，返回原文、页码、父级上下文和实际检索方式；不传 paper_ids 时检索当前会话全部可用全文",
            ],
        },
        ensure_ascii=False,
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def _compact_writing_tool_result(item: JsonObject) -> str:
    """只把真实切片的编号、页码和短原文放进下一轮写作提示词。"""

    if not isinstance(item, dict):
        return _compact_text(item, max_chars=6000)
    if item.get("tool") == "get_extraction" and isinstance(item.get("result"), list):
        # 中文注释：结构化摘要可能一次返回多篇论文。过去从头截取 6000 字，
        # 前一篇的作者和摘要会占满位置，后一篇连切片编号都看不到。现在每篇
        # 只保留简短说明和完整的原文编号，让模型能继续按编号核对，而不是猜来源。
        records: list[JsonObject] = []
        field_names = ("research_topic", "research_object", "methods", "conclusions", "contributions", "limitations")
        for record in item["result"][:8]:
            if not isinstance(record, dict):
                continue
            paper_id = str(record.get("paperId") or "").strip()
            extraction = record.get("extraction") or {}
            if not isinstance(extraction, dict):
                continue
            fields: JsonObject = {}
            for field_name in field_names:
                value = str(extraction.get(field_name) or "").strip()
                if not value:
                    continue
                # 中文注释：原文编号常放在段尾，先截文字会把它切断。因此从
                # 完整字段中单独取出编号，再截短解释文字；编号必须原样保留。
                chunk_ids = [marker for marker in re.findall(r"\[([^\[\]\n]+)\]", value)
                             if paper_id and marker.casefold().startswith(paper_id.casefold() + ":")]
                fields[field_name] = {
                    "text": re.sub(r"\[[^\[\]\n]+\]", "", value).strip()[:260],
                    "chunkIds": _deduplicate_strings(chunk_ids)[:4 if field_name == "methods" else 2],
                }
            note = extraction.get("note") if isinstance(extraction.get("note"), dict) else {}
            paper_info = extraction.get("paper") if isinstance(extraction.get("paper"), dict) else {}
            records.append({
                "paperId": paper_id,
                "status": record.get("status"),
                "source": record.get("source"),
                "title": str(paper_info.get("title") or "")[:180],
                "fields": fields,
                "abstract_only_note": {
                    "summary": str(note.get("short_summary") or "")[:350],
                    "evidence_level": note.get("evidence_level"),
                } if note else None,
            })
        return json.dumps({"tool": "get_extraction", "papers": records,
                           "note": "切片编号只是核对入口；摘要或笔记不足以证明正文主张。"}, ensure_ascii=False)
    # 中文说明：检索结果中的 parent_content 往往比当前切片还长。过去直接把整个
    # 工具结果截到 6000 字，第一条的父级上下文就占满了空间，后面论文的 chunkId
    # 被截掉。原始工具结果仍完整留在状态和产物中；这里只整理模型下一轮确实要看的
    # 几项信息，不修改原文，也不把检索命中当成事实已经得到支持。
    payload = item.get("fallback") or item.get("result")
    if not isinstance(payload, dict) or not isinstance(payload.get("chunks"), list):
        return _compact_text(item, max_chars=6000)
    shown: list[JsonObject] = []
    per_paper: dict[str, int] = {}
    for chunk in payload["chunks"]:
        if not isinstance(chunk, dict):
            continue
        paper_id = str(chunk.get("paperId") or "").strip()
        chunk_id = str(chunk.get("chunkId") or "").strip()
        if not paper_id or not chunk_id or per_paper.get(paper_id.casefold(), 0) >= 2:
            continue
        shown.append({
            "paperId": paper_id,
            "chunkId": chunk_id,
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "content": str(chunk.get("content") or "")[:650],
        })
        per_paper[paper_id.casefold()] = per_paper.get(paper_id.casefold(), 0) + 1
        if len(shown) >= 10:
            break
    return json.dumps({
        "tool": item.get("tool"),
        "query": payload.get("query"),
        "status": payload.get("status"),
        "chunks": shown,
        "note": "这些只是带原文位置的候选片段；写作前仍须核对是否真的支持主张。",
    }, ensure_ascii=False)


def _abstract_messages(*, topic: str, sections: list[JsonObject], word_count: int,
                       language: str = "zh", instruction: str = "") -> list[JsonObject]:
    """构造摘要提示词，只把已经完成的小节正文交给模型。"""

    # 中文说明：摘要只读取已完成的小节正文，系统提示词不在这里重复维护。
    system_prompt = WRITING_ABSTRACT_SYSTEM_PROMPT
    body = [
        {
            "小节标题": str(section.get("section_title") or section.get("section_id") or ""),
            "正文": str(section.get("content") or "").strip(),
        }
        for section in sections
        if isinstance(section, dict) and str(section.get("content") or "").strip()
    ]
    user_prompt = json.dumps(
        # 明确传递任务语言，避免摘要跟随英文论文或旧章节的语言。
        {"用户主题": topic, "输出语言": language, "语言要求": "摘要使用输出语言，保留必要的专有名词。",
         "摘要建议字数": max(100, int(word_count or 300)),
         "用户本次摘要修订要求": instruction[:4000], "已完成正文": body},
        ensure_ascii=False,
        indent=2,
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def _fallback_abstract(topic: str, sections: list[JsonObject]) -> str:
    """模型不可用时从各小节开头拼出一段保守摘要。"""

    summaries: list[str] = []
    for section in sections:
        content = re.sub(r"\s+", " ", str(section.get("content") or "")).strip()
        if not content:
            continue
        first_sentence = re.split(r"(?<=[。！？.!?])\s*", content, maxsplit=1)[0].strip()
        summaries.append(first_sentence or content[:180])
        if len(summaries) >= 4:
            break
    if not summaries:
        return f"本文围绕“{topic}”梳理相关研究，并总结现有工作的主要进展、研究不足与后续方向。"
    return f"本文围绕“{topic}”梳理相关研究。" + "".join(summaries)


def _review_messages(state: SectionLoopState, located: JsonObject | None = None) -> list[JsonObject]:
    """构造审查提示词，附上已绑定原文的短上下文以检查方法归属。"""

    # 中文说明：审查范围保持窄而明确，避免模型把审查变成重新设计全文。
    system_prompt = WRITING_REVIEW_SYSTEM_PROMPT
    # 中文说明：自动修订时 task 后面会附上旧版审计失败句和旧正文，供写作模型
    # 修改。文风审查若也看到它们，常把“上一版引用无效”误当成当前草稿仍无效；
    # 引用位置已有程序单独检查，这里只保留用户范围与本节标题等原始要求。
    review_task = str(state.get("task") or "").split("\n独立核查要求：", 1)[0]
    review_task = review_task.split("\n上一版正文（仅作为修订草稿", 1)[0]
    source_excerpts: list[JsonObject] = []
    for binding in (located or {}).get("citation_evidence") or []:
        # 中文说明：第二十一轮 GAT 的切片从 LLE 句子中间开始，单看子片段
        # 会把“别人的方法”误读成 GAT。带上父段中紧邻片段的前文，
        # 让审查能识别是谁做了这个操作；长段和多处绑定都设定上限。
        for chunk in binding.get("chunks") or []:
            content = str(chunk.get("content") or "")
            parent = str(chunk.get("parent_content") or "")
            position = parent.find(content[:min(40, len(content))]) if content and parent else -1
            before = parent[max(0, position - 420):position] if position >= 0 else ""
            source_excerpts.append({
                "claim": str(binding.get("claim") or "")[:220],
                "chunkId": str(chunk.get("chunkId") or ""),
                "原文前文": before,
                "已绑定原文": content[:520],
            })
            if len(source_excerpts) >= 12:
                break
        if len(source_excerpts) >= 12:
            break
    user_prompt = json.dumps(
        {
            "section_id": state.get("section_id"),
            "小节任务": review_task,
            "计划字数": state.get("word_count"),
            "正文草稿": state.get("draft") or "",
            "已绑定原文与紧邻前文（仅供检查主语和方法归属；不是独立审计）": source_excerpts,
        },
        ensure_ascii=False,
        indent=2,
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def _stop_writing(state: SectionLoopState, reason: str, raw_outputs: list[str] | None = None) -> SectionLoopState:
    """调用或格式失败后结束小节；保留上一版正文，不把内部任务、证据 JSON 拼进正文。"""
    draft = str(state.get("draft") or "").strip()
    # 中文说明：修订失败时如果已有上一版正文，应标记为“尚未核查通过”；
    # 只有连一版正文都没有时，才算真正的正文生成失败。
    return {
        **state,
        "action_type": "draft",
        "draft": draft or "本节正文未能生成，需重新生成并核查。",
        "raw_model_outputs": raw_outputs if raw_outputs is not None else list(state.get("raw_model_outputs") or []),
        "completed": True,
        "generation_failed": not bool(draft),
        "review": {"passed": False, "status": "unverified", "message": reason},
        "warnings": [*list(state.get("warnings") or []), reason],
    }


def _find_extraction_in_read_results(paper_id: str, read_results: list[JsonObject]) -> JsonObject | None:
    """优先从阅读节点结果里查找结构化摘要。"""

    for item in read_results:
        if not isinstance(item, dict):
            continue
        paper = dict(item.get("paper") or {})
        candidates = {str(paper.get(key) or "").casefold() for key in ("paperId", "id", "doi")}
        candidates.add(str(item.get("paperId") or "").casefold())
        if paper_id.casefold() not in candidates:
            continue
        extraction = item.get("extraction")
        if isinstance(extraction, dict) and any(str(value).strip() for value in extraction.values()):
            return dict(extraction)
    return None


def _find_extraction_in_session_read(paper_id: str, read_results: list[JsonObject]) -> JsonObject | None:
    """从当前会话所有轮次的阅读产物中查找论文资料。

    中文说明：同一篇论文可能在多个轮次被重新阅读，所以从后往前查找，优先使用
    最近一次保存的结果。如果阅读产物没有全文提取结果，就把论文信息和阅读笔记
    一起返回，写作 Agent 仍然可以使用摘要阅读阶段已经整理好的内容。
    """

    for item in reversed(read_results):
        if not isinstance(item, dict):
            continue
        paper = dict(item.get("paper") or {})
        candidates = {str(paper.get(key) or "").casefold() for key in ("paperId", "id", "doi")}
        candidates.add(str(item.get("paperId") or "").casefold())
        if paper_id.casefold() not in candidates:
            continue

        extraction = item.get("extraction")
        if isinstance(extraction, dict) and any(str(value).strip() for value in extraction.values()):
            return dict(extraction)

        note = item.get("note")
        if isinstance(note, dict) and any(str(value).strip() for value in note.values()):
            # 保留论文摘要和标题，避免只把笔记交给模型后失去原始论文上下文。
            return {"paper": paper, "note": dict(note)}
    return None


def _find_extraction_in_cache(paper_id: str, cache_dir: Path) -> JsonObject | None:
    """从论文缓存目录中的 extraction.json 查找结构化摘要。"""

    for directory in _paper_cache_dirs(paper_id, cache_dir):
        path = directory / "extraction.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # 中文说明：目录名清理后可能重名，因此摘要文件本身的论文编号也必须
        # 属于已核对的元数据，不能把另一篇论文的摘要贴到当前论文名下。
        if not isinstance(payload, dict) or str(payload.get("paperId") or "").casefold() not in _cache_paper_aliases(directory):
            continue
        extraction = payload.get("extraction") if isinstance(payload, dict) else None
        if isinstance(extraction, dict):
            return dict(extraction)
    return None


def _find_chunks_path(paper_id: str, cache_dir: Path) -> Path | None:
    """定位某篇论文缓存目录下的 chunk.json。"""

    for directory in _paper_cache_dirs(paper_id, cache_dir):
        chunks_path = directory / "chunk.json"
        aliases = _cache_paper_aliases(directory)
        chunks = load_chunks_file(chunks_path)
        # 中文说明：同时核对文件里的每个片段，避免正文与目录元数据不一致时
        # 把其他论文的原文交给写作模型或用于建立引用对应关系。
        if chunks and all(chunk.paperId.casefold() in aliases for chunk in chunks):
            return chunks_path
    return None


def _paper_cache_dirs(paper_id: str, cache_dir: Path) -> list[Path]:
    """根据 paperId 找可能的缓存目录。"""

    directories: list[Path] = []
    direct = cache_dir / safe_cache_name(paper_id)
    if direct.is_dir() and _cache_dir_matches_paper_id(direct, paper_id):
        directories.append(direct)
    if not cache_dir.exists():
        return directories
    for candidate in cache_dir.iterdir():
        if not candidate.is_dir() or candidate in directories:
            continue
        if _cache_dir_matches_paper_id(candidate, paper_id):
            directories.append(candidate)
    return directories


def _cache_dir_matches_paper_id(directory: Path, paper_id: str) -> bool:
    """通过 metadata.json 判断缓存目录是否属于目标论文。"""

    return paper_id.casefold() in _cache_paper_aliases(directory)


def _cache_paper_aliases(directory: Path) -> set[str]:
    """读取缓存元数据中的论文编号与 DOI，目录名称本身不作为归属依据。"""

    metadata_path = directory / "metadata.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(payload, dict):
        return set()
    paper = payload.get("paper") if isinstance(payload.get("paper"), dict) else {}
    candidates = {str(payload.get("paperId") or "").casefold()}
    candidates.update(str(paper.get(key) or "").casefold() for key in ("paperId", "id", "doi"))
    candidates.discard("")
    return candidates


def _chunk_to_markdown(chunk: TextChunk) -> JsonObject:
    """把 chunk 整理成写作 Agent 容易阅读的 Markdown 片段。"""

    header = f"### {chunk.paperId} / {chunk.chunk_id}"
    if chunk.section:
        header += f" / {chunk.section}"
    return {
        "paperId": chunk.paperId,
        "chunkId": chunk.chunk_id,
        "markdown": f"{header}\n\n{chunk.content.strip()}",
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "content": chunk.content,
        "section": chunk.section,
        "parent_chunk_id": chunk.parent_chunk_id,
        "parent_content": chunk.parent_content,
    }


def _extract_json_object(text: str) -> JsonObject | None:
    """从模型输出中提取 JSON 对象。"""

    stripped = text.strip()
    if not stripped:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.IGNORECASE | re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        candidates.append(stripped[start : end + 1])
    candidates.append(stripped)
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _string_list(value: Any) -> list[str]:
    """把字符串或字符串数组整理成干净数组。"""

    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _deduplicate_strings(values: list[str]) -> list[str]:
    """按原顺序给字符串去重。"""

    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _paper_ids_from_any(value: Any) -> list[str]:
    """从任意嵌套数据里尽量提取 paperId。"""

    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in {"paperId", "paper_id"} and str(item).strip():
                found.append(_clean_paper_id_candidate(str(item)))
            elif str(key) in {"paperIds", "paper_ids"}:
                found.extend(_clean_paper_id_candidate(text) for text in _string_list(item))
            else:
                found.extend(_paper_ids_from_any(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_paper_ids_from_any(item))
    elif isinstance(value, str):
        found.extend(_clean_paper_id_candidate(match) for match in re.findall(r"\[([^\[\]]+)\]", value)
                     if match.strip() and not _is_numeric_interval_marker(match))
    return _deduplicate_strings(found)


def _clean_paper_id_candidate(value: str) -> str:
    """清理从正文或证据里提取到的 paperId 候选值。"""

    return value.strip().strip('"').strip("'").strip()


def _compact_text(value: Any, *, max_chars: int = 500) -> str:
    """把证据或前置小节压成短文本，供兜底正文使用。"""

    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]
