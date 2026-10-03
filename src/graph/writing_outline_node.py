from __future__ import annotations

import asyncio
import json
import re
from typing import Any, cast

from src.agents.writingOutlineAgent import (
    OVERALL_ANALYSIS_FIELDS,
    WritingOutlineAgent,
    build_writing_outline_agent,
    load_writing_outline_agent_llm,
)
from src.graph.runtime import WorkflowRuntimeContext
from src.graph.runtime_resources import WorkflowRuntimeResources
from src.graph.state_models import JsonObject, State
from src.llm import ProviderSnapshot
from src.models.sessions import utc_now
from src.repositories.sessions.base import SessionRepository


# 中文说明：标题字段和正文范围发生了变化，使用新版本号便于识别旧产物。
WRITING_OUTLINE_VERSION = "1.1"


def run_writing_outline_node():
    """生成论文写作大纲节点。

    中文说明：
    这个节点只负责“写作前的规划”，也就是章节和小节怎么安排。
    小节正文写作还没有设计好，所以这里不会生成正文，避免后面不好改。
    """

    async def _node(state: State) -> State:
        """从分析报告里读取 overall_framework，并生成结构化写作大纲。"""

        request = state.get("request")
        if request is None:
            raise ValueError("写作大纲节点缺少用户综述主题，无法继续生成大纲")

        reporter = _resolve_reporter(state)
        analysis_report = dict(state.get("analysis_report") or {})
        llm = _resolve_llm(state)
        agent = build_writing_outline_agent(llm)

        if reporter is not None:
            reporter.started("正在根据分析结果生成写作大纲", stage="writing_outline_start")

        def report_outline_usage(usage: JsonObject) -> None:
            """把写作大纲模型返回的真实 token 用量更新到大纲卡片。"""

            if reporter is not None:
                reporter.progress("写作大纲模型调用完成", stage="writing_outline", **usage)

        outline, raw_model_output, reason = await _generate_outline(
            state,
            agent=agent,
            usage_callback=report_outline_usage,
        )
        used_llm = outline is not None and reason == "ok"
        if outline is None:
            outline = _fallback_outline(topic=request.topic, analysis_report=analysis_report)
        # 中文说明：用户明确要求“每篇单独一节，再用一节比较”时，仅检查总节数
        # 会漏掉某篇被拆成两节、比较节消失的情况。第 26 轮正是六节数量正确，
        # 但 GraphSAGE 占两节且没有五篇比较。先把具体缺口告诉同一模型补写
        # 一次大纲；若仍不符合，就停在写作前，避免为注定不合题的正文付费。
        structure_issue = _explicit_named_paper_outline_problem(outline, request)
        if structure_issue:
            retry_state = dict(state)
            retry_state["outline_structure_feedback"] = structure_issue
            retry_outline, retry_raw, retry_reason = await _generate_outline(
                cast(State, retry_state), agent=agent, usage_callback=report_outline_usage,
            )
            raw_model_output += "\n--- 按用户明确结构补写大纲 ---\n" + retry_raw
            # 中文说明：结构失败也必须保留两次真实模型输出与失败原因，便于区分
            # 模型漏写和校验格式问题；失败产物明确标记，不能进入正文写作。
            failure_issue = (f"{structure_issue}；{retry_reason}" if retry_outline is None
                             else _explicit_named_paper_outline_problem(retry_outline, request))
            if failure_issue:
                failed_ref = await _persist_outline_if_possible(state, {
                    "outline_version": WRITING_OUTLINE_VERSION,
                    "topic": request.topic,
                    "writing_outline": retry_outline or outline,
                    "execution_metadata": {"status": "failed", "created_at": utc_now(),
                                           "message": failure_issue, "attempt_count": 2},
                    "diagnostics": {"raw_model_output": raw_model_output,
                                    "initial_structure_issue": structure_issue},
                })
                if failed_ref and reporter is not None:
                    reporter.artifact(failed_ref, stage="writing_outline_failed_artifact_ready")
            if retry_outline is None:
                raise ValueError(f"大纲未满足用户明确结构，补写失败：{structure_issue}；{retry_reason}")
            retry_issue = _explicit_named_paper_outline_problem(retry_outline, request)
            if retry_issue:
                raise ValueError(f"大纲补写后仍未满足用户明确结构：{retry_issue}")
            outline = retry_outline
            used_llm = True
            reason = "ok"
        # 中文说明：模型偶尔会无视用户明确写出的“共六节”，额外安排综合小节。
        # 在进入逐节写作前按原有顺序执行这个数量上限，避免为已越界的大纲付费写作；
        # 被删节数同时写进产物，不能悄悄把裁剪后的大纲称为模型完全遵守要求。
        section_limit = _requested_section_limit(request)
        outline, removed_sections = _trim_outline_sections(outline, section_limit)

        report = {
            "outline_version": WRITING_OUTLINE_VERSION,
            "topic": request.topic,
            # 中文说明：writing_outline 是后续正文写作最应该直接读取的核心对象。
            "writing_outline": outline,
            "execution_metadata": {
                "used_llm": used_llm,
                "model_used": llm.model if isinstance(llm, ProviderSnapshot) else "unavailable",
                "created_at": utc_now(),
                "message": ("已使用模型生成写作大纲" if used_llm else reason)
                           + (f"；按用户小节数量上限移除 {removed_sections} 节" if removed_sections else ""),
                "section_limit": section_limit,
                "removed_sections": removed_sections,
            },
        }

        artifact_refs = list(state.get("writing_outline_artifact_refs") or [])
        persisted = await _persist_outline_if_possible(state, report)
        if persisted:
            artifact_refs.append(persisted)
            if reporter is not None:
                reporter.artifact(persisted, stage="writing_outline_artifact_ready")

        diagnostics = dict(state.get("diagnostics") or {})
        diagnostics["writing_outline"] = {
            "used_llm": used_llm,
            "status": "ok" if used_llm else "fallback",
            "message": report["execution_metadata"]["message"],
            "raw_model_output": raw_model_output,
        }

        if reporter is not None:
            reporter.completed(
                "写作大纲节点已完成",
                stage="writing_outline_done",
                chapter_count=len(outline),
                used_llm=used_llm,
            )

        updated = dict(state)
        updated.update(
            writing_outline=outline,
            writing_outline_report=report,
            writing_outline_artifact_refs=artifact_refs,
            diagnostics=diagnostics,
            current_step="write_outline",
        )
        return cast(State, updated)

    return _node


async def _generate_outline(state: State, *, agent: WritingOutlineAgent, usage_callback: Any | None = None) -> tuple[JsonObject | None, str, str]:
    """调用 Agent 生成大纲，并把空结果当作失败处理。"""

    outline, raw_model_output, reason = await agent.async_generate_outline(dict(state), usage_callback=usage_callback)
    if not _outline_is_complete(outline):
        return None, raw_model_output, reason if reason != "ok" else "模型返回的大纲为空"
    return outline, raw_model_output, reason


def _outline_is_complete(outline: JsonObject | None) -> bool:
    """检查大纲是否真的包含章节和小节。

    中文说明：
    模型有时会返回一个能解析的 JSON，但里面缺字段。
    这种结果对后续写正文没有帮助，所以这里直接判定为不可用，让节点走兜底大纲。
    """

    if not outline:
        return False
    for chapter in outline.values():
        if not isinstance(chapter, dict):
            return False
        if not str(chapter.get("title") or "").strip():
            return False
        if not str(chapter.get("description") or "").strip():
            return False
        sections = chapter.get("Sections")
        if not isinstance(sections, dict) or not sections:
            return False
        for section in sections.values():
            if not isinstance(section, dict):
                return False
            if not str(section.get("title") or "").strip():
                return False
            for key in ("title", "task", "evidence-map", "ref-sections", "word-count"):
                if key not in section:
                    return False
    return True


def _explicit_named_paper_outline_problem(outline: JsonObject, request: Any) -> str:
    """只核对用户同时写明的逐篇小节与比较节，不推测普通主题的隐含结构。"""

    topic = str(getattr(request, "topic", "") or "")
    paper_ids = list(dict.fromkeys(re.findall(r"arxiv\s*:\s*(\d{4}\.\d{4,5})", topic, re.IGNORECASE)))
    limit = _requested_section_limit(request)
    if not ("单论文" in topic and "比较小节" in topic and limit == len(paper_ids) + 1 and len(paper_ids) >= 2):
        return ""
    sections = [section for chapter in outline.values() if isinstance(chapter, dict)
                for section in (chapter.get("Sections") or {}).values() if isinstance(section, dict)]
    comparison = [section for section in sections
                  if re.search(r"比较|对比|综合|差异|对照", str(section.get("title") or ""))]
    singles = [section for section in sections if section not in comparison]
    counts = {paper_id: 0 for paper_id in paper_ids}
    for section in singles:
        task = str(section.get("task") or "")
        mentions = {paper_id for paper_id in paper_ids
                    if re.search(r"arxiv\s*:\s*" + re.escape(paper_id) + r"\b", task, re.IGNORECASE)}
        if len(mentions) != 1:
            return "单论文小节的任务必须明确且只指定一篇用户点名的 arXiv 原论文"
        counts[next(iter(mentions))] += 1
    if len(sections) != limit or len(comparison) != 1 or any(count != 1 for count in counts.values()):
        return (f"用户要求共 {limit} 节：{len(paper_ids)} 篇原论文各占一节，另有一节定性比较；"
                f"现有 {len(sections)} 节、比较节 {len(comparison)} 节、逐篇次数 {counts}")
    return ""


def _requested_section_limit(request: Any) -> int | None:
    """只识别用户明确给出的总节数上限，不猜“每篇一节”等隐含数量。"""

    configured = (getattr(request, "constraints", None) or {}).get("max_sections")
    if configured is not None:
        try:
            value = int(configured)
            return value if value > 0 else None
        except (TypeError, ValueError):
            return None
    topic = str(getattr(request, "topic", "") or "")
    match = re.search(r"(?:共|总共|总计|最多)\s*([1-9]\d?|[一二三四五六七八九十])\s*(?:个)?(?:小节|节)", topic)
    if not match:
        return None
    numerals = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    value = numerals.get(match.group(1)) or int(match.group(1))
    return value


def _trim_outline_sections(outline: JsonObject, limit: int | None) -> tuple[JsonObject, int]:
    """保留大纲顺序中不超过用户上限的小节，并记录实际去掉的数量。"""

    if limit is None:
        return outline, 0
    kept: JsonObject = {}
    count = 0
    removed = 0
    for chapter_key, chapter in outline.items():
        sections: JsonObject = {}
        for section_key, section in (chapter.get("Sections") or {}).items():
            if count >= limit:
                removed += 1
                continue
            sections[section_key] = section
            count += 1
        if sections:
            kept[chapter_key] = {**chapter, "Sections": sections}
    return kept, removed


def _fallback_outline(*, topic: str, analysis_report: JsonObject) -> JsonObject:
    """模型不可用时生成一份保守大纲。

    中文说明：
    这份兜底大纲不假装自己做了复杂判断，只把分析节点已有的信息安排进常见综述结构。
    后续用户可以拿到一个字段完整的对象，前端和下一步节点也不会因为空值出错。
    """

    topic_text = topic or str(analysis_report.get("topic") or "当前主题")
    overall_analysis = dict(analysis_report.get("overall_analysis") or {})

    # 中文说明：兜底结构从正文第一章开始，不再放摘要、引言和参考文献。
    # 每个章节和小节都写出独立标题，前端展示和后续正文写作都可以直接使用。
    outline: JsonObject = {
        "Chapter1": {
            "title": "相关研究现状",
            "description": f"围绕《{topic_text}》梳理已有研究的主要方向、代表性发现及其适用范围。",
            "Sections": {
                "section1": {
                    "title": "总体研究现状",
                    "task": "梳理该领域的研究范围、核心问题、主要方向和总体发展状态。",
                    "evidence-map": _available_evidence_fields(
                        overall_analysis,
                        "领域整体研究概况",
                        "各子主题横向差异对比分析",
                    ),
                    "ref-sections": [],
                    "word-count": 1000,
                }
            },
        },
        "Chapter2": {
            "title": "研究方法与技术演进",
            "description": "比较不同研究采用的方法、技术路线和演进趋势，说明方法差异如何影响研究结果。",
            "Sections": {
                "section1": {
                    "title": "方法与技术路线",
                    "task": "归纳各研究使用的方法和技术路线，说明它们分别解决了哪些问题。",
                    "evidence-map": _available_evidence_fields(overall_analysis, "领域技术与研究方法迭代脉络"),
                    "ref-sections": ["Chapter1"],
                    "word-count": 900,
                },
                "section2": {
                    "title": "研究时序演化",
                    "task": "按照研究发展顺序梳理关键变化，说明研究重点如何从早期问题逐步转向当前问题。",
                    "evidence-map": _available_evidence_fields(overall_analysis, "领域研究时序演化脉络"),
                    "ref-sections": ["Chapter1"],
                    "word-count": 800,
                },
            },
        },
        "Chapter3": {
            "title": "研究争议、空白与发展方向",
            "description": "在前文研究现状和方法比较的基础上，归纳主要争议、研究不足及可继续推进的方向。",
            "Sections": {
                "section1": {
                    "title": "研究共识与核心争议",
                    "task": "归纳不同研究之间的一致点和分歧点，说明争议来自方法、数据还是研究对象差异。",
                    "evidence-map": _available_evidence_fields(
                        overall_analysis,
                        "领域全域共性研究共识",
                        "领域核心研究争议与矛盾体系",
                    ),
                    "ref-sections": ["Chapter2"],
                    "word-count": 900,
                },
                "section2": {
                    "title": "研究空白与后续方向",
                    "task": "总结仍然缺少研究的问题，并提出与这些空白对应的后续研究方向。",
                    "evidence-map": _available_evidence_fields(
                        overall_analysis,
                        "领域系统性研究空白与局限",
                        "领域整体总结与研究展望",
                    ),
                    "ref-sections": ["Chapter3.section1"],
                    "word-count": 800,
                },
            },
        },
    }
    return outline


def _available_evidence_fields(overall_analysis: JsonObject, *fields: str) -> list[str]:
    """只保留当前全局分析中确实有内容的字段名。"""

    return [
        field
        for field in fields
        if field in OVERALL_ANALYSIS_FIELDS and str(overall_analysis.get(field) or "").strip()
    ]


def _resolve_llm(state: State) -> ProviderSnapshot | None:
    """优先使用外部注入的写作大纲模型，没有注入时读取默认配置。"""

    # 显式 None 表示关闭；只有省略字段或使用 auto 才读取本地配置。
    injected = state.get("writing_outline_node_llm", "auto")
    if isinstance(injected, ProviderSnapshot):
        return injected
    if injected == "auto":
        snapshot = load_writing_outline_agent_llm()
        # 中文注释：本节点自己创建的模型连接，在任务结束时连同备用模型一起关闭。
        resources = getattr(state.get("runtime_context"), "resources", None)
        if isinstance(resources, WorkflowRuntimeResources):
            resources.track_model_snapshot(snapshot)
        return snapshot
    return None


def _resolve_reporter(state: State):
    """从运行上下文里取出写作大纲节点的进度上报器。"""

    runtime = cast(WorkflowRuntimeContext | None, state.get("runtime_context"))
    if runtime is None or runtime.sync_port is None:
        return None
    return runtime.sync_port.for_node("write_outline", "写作大纲")


async def _persist_outline_if_possible(state: State, report: JsonObject) -> JsonObject | None:
    """如果当前有会话仓库，就把写作大纲保存成 JSON 产物。"""

    repo = cast(SessionRepository | None, state.get("session_repo"))
    session_key = _optional_text(state.get("session_key"))
    turn_id = _optional_text(state.get("turn_id"))
    if repo is None or not session_key or not turn_id:
        return None
    try:
        record = await asyncio.to_thread(
            repo.write_artifact,
            session_key,
            "writing_outline",
            "writing_outline.json",
            json.dumps(report, ensure_ascii=False, indent=2),
            relative_path=f"artifacts/writing_outline/{turn_id}/writing_outline.json",
            metadata={"turn_id": turn_id, "format": "json", "outline_version": WRITING_OUTLINE_VERSION},
        )
    except Exception:
        return None
    return {
        "artifact_id": str(record["id"]),
        "artifact_type": str(record["artifact_type"]),
        "name": str(record["name"]),
        "path": str(record["path"]),
        "size": int(record["size"]),
        "created_at": str(record["created_at"]),
        "metadata": dict(record.get("metadata") or {}),
    }


def _optional_text(value: Any) -> str | None:
    """把可选值整理成非空字符串。"""

    text = str(value or "").strip()
    return text or None
