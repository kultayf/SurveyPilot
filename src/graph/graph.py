from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents.contracts import ReviewRequest
from src.graph.analyse_node import run_analyse_node
from src.graph.evidence_node import (
    run_matrix_node, run_supplement_node, run_audit_node, run_revision_node,
    route_after_matrix, route_after_supplement, route_after_audit,
)
from src.graph.matrix_review_node import run_matrix_review_node
from src.graph.citation_node import run_citation_node
from src.graph.conflict_node import run_conflict_node
from src.graph.reply_node import missing_requested_arxiv_ids, run_compose_reply_node
from src.graph.read_node import run_read_node
from src.graph.runtime import InlineWorkflowSyncPort, WorkflowRuntimeContext
from src.graph.search_node import run_search_agent_node
from src.graph.state_models import State
from src.graph.writing_outline_node import run_writing_outline_node
from src.graph.writing_node import run_writing_node
from src.graph.workflow_checkpoint import restore_checkpoint, save_checkpoint
from src.paper_retrieval.models import PaperDocument
from src.repositories.sessions.base import SessionRepository


@dataclass(slots=True)
class GraphRunResult:
    """封装图执行完成后的稳定返回结构。"""

    papers: list[PaperDocument] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


GraphState = State


def _entrypoint(state: State) -> str:
    """根据是否带阅读恢复现场决定从检索还是阅读节点开始。"""

    resume_node = str(state.get("workflow_resume_node") or "")
    if resume_node:
        return resume_node
    checkpoint = state.get("read_resume_checkpoint") or {}
    if checkpoint.get("resume_stage") == "matrix_review":
        return "run_matrix_review"
    if checkpoint:
        return "run_read"
    constraints = state["request"].constraints
    if constraints.get("citation_snowball") is True and constraints.get("citation_seed_ids"):
        return "run_citation_discovery"
    return "run_search_agent"


def _route_after_search(state: State) -> str:
    """检索无法继续或没有选出论文时，直接整理说明并结束本轮任务。"""

    search_summary = dict(state.get("search_summary") or {})
    # 中文说明：search_halted 表示检索条件模型没有给出可用计划；空 search_results
    # 表示数据源没有结果、请求失败，或候选论文在粗筛时被全部排除。无论是哪一种，
    # 后面的阅读、矩阵、分析和写作都没有可靠输入，继续执行只会产出空白或误导性草稿。
    if search_summary.get("search_halted") is True:
        return "reply"
    if not state.get("search_results"):
        return "reply"
    # 中文说明：第 34 轮一次 arXiv 超时只检出指定五篇中的三篇，旧流程仍继续
    # 阅读和写作，可能生成看似完整却缺两篇原论文的综述。用户明确写出编号时，
    # 检索名单必须逐篇覆盖；不足就在此有界停止并给出缺失编号，不用其他论文顶替。
    if missing_requested_arxiv_ids(state["request"].topic, list(state.get("search_results") or [])):
        return "reply"
    return "continue"


def _route_after_read(state: State) -> str:
    """仅在已经选出可精读论文时才继续进入证据与写作链路。"""

    read_summary = dict(state.get("read_summary") or {})
    indexed_count = int(read_summary.get("indexed_paper_count") or 0)
    # 中文说明：即使论文拿到了精读名额，下载、解析或建索引失败时仍没有可审计的
    # 原文片段。只有至少一篇论文完成索引，后续矩阵、分析和写作才有可靠输入。
    if indexed_count < 1:
        return "reply"
    # 中文说明：检索到论文不等于已解析全文。若指定编号在下载或切块阶段失败，
    # 写作也必须停止；只检查 status=indexed 的本轮阅读记录，不用旧缓存凑数。
    indexed_papers = [result.get("paper") or {} for result in state.get("read_results") or []
                      if (result.get("full_text") or {}).get("status") == "indexed"]
    if missing_requested_arxiv_ids(state["request"].topic, indexed_papers):
        return "reply"
    return "continue"


def build_graph():
    """构建当前论文工作流使用的执行图。"""

    workflow = StateGraph(State)
    workflow.add_node("run_search_agent", _with_cancellation_boundary("run_search_agent", run_search_agent_node()))
    workflow.add_node("run_read", _with_cancellation_boundary("run_read", run_read_node()))
    workflow.add_node("run_analyse", _with_cancellation_boundary("run_analyse", run_analyse_node()))
    workflow.add_node("run_writing_outline", _with_cancellation_boundary("run_writing_outline", run_writing_outline_node()))
    workflow.add_node("run_writing", _with_cancellation_boundary("run_writing", run_writing_node()))
    workflow.add_node("compose_reply", _with_cancellation_boundary("compose_reply", run_compose_reply_node()))
    for name, node in (("run_matrix", run_matrix_node), ("run_supplement", run_supplement_node),
                       ("run_audit", run_audit_node), ("run_revision", run_revision_node), ("run_matrix_review", run_matrix_review_node)):
        workflow.add_node(name, _with_cancellation_boundary(name, node))
    workflow.add_conditional_edges(START, _entrypoint, {
        name: name for name in (
            "run_search_agent", "run_read", "run_matrix_review", "run_citation_discovery",
            "run_matrix", "run_supplement", "run_conflict", "run_analyse",
            "run_writing_outline", "run_writing", "run_audit", "run_revision", "compose_reply",
        )
    })
    workflow.add_node("run_citation_discovery", _with_cancellation_boundary("run_citation_discovery", run_citation_node))
    workflow.add_conditional_edges(
        "run_search_agent",
        _route_after_search,
        {"continue": "run_citation_discovery", "reply": "compose_reply"},
    )
    workflow.add_edge("run_citation_discovery", "run_read")
    workflow.add_conditional_edges(
        "run_read",
        _route_after_read,
        {"continue": "run_matrix", "reply": "compose_reply"},
    )
    workflow.add_conditional_edges("run_matrix", route_after_matrix, {"supplement": "run_supplement", "analyse": "run_matrix_review"})
    workflow.add_conditional_edges("run_supplement", route_after_supplement, {"read": "run_read", "analyse": "run_matrix_review"})
    workflow.add_node("run_conflict", _with_cancellation_boundary("run_conflict", run_conflict_node))
    workflow.add_edge("run_matrix_review", "run_conflict")
    workflow.add_edge("run_conflict", "run_analyse")
    workflow.add_edge("run_analyse", "run_writing_outline")
    workflow.add_edge("run_writing_outline", "run_writing")
    workflow.add_edge("run_writing", "run_audit")
    workflow.add_conditional_edges("run_audit", route_after_audit, {"revise": "run_revision", "reply": "compose_reply"})
    workflow.add_edge("run_revision", "run_writing")
    workflow.add_edge("compose_reply", END)
    return workflow.compile(name="paper_graph")


def _with_cancellation_boundary(node_name: str, node):
    """给所有图节点加上统一的用户停止检查，避免节点里重复写样板代码。"""

    async def _guarded(state: State) -> State:
        """开始和结束节点时各检查一次，已完成的当前节点不会再启动下一个节点。"""

        runtime = state.get("runtime_context")
        cancellation = getattr(runtime, "cancellation", None)
        if cancellation is not None:
            cancellation.raise_if_requested()

        result = await node(state)

        # 中文说明：节点完成后先保存下一步的位置。用户刚好在这个边界点停止时，
        # 已经完整生成的结果仍能恢复；执行到一半的节点不会被当作已完成。
        merged = {**state, **(result or {})}
        next_node = _checkpoint_next_node(node_name, merged)
        if next_node is not None:
            await save_checkpoint(merged, next_node)
        if cancellation is not None:
            cancellation.raise_if_requested()
        return result

    _guarded.__name__ = f"{node_name}_with_cancellation"
    return _guarded


def _checkpoint_next_node(node_name: str, state: State) -> str | None:
    """根据刚完成的阶段，记录下一次应从哪里接着做。"""

    fixed = {
        "run_citation_discovery": "run_read", "run_matrix_review": "run_conflict",
        "run_conflict": "run_analyse", "run_analyse": "run_writing_outline",
        "run_writing_outline": "run_writing", "run_writing": "run_audit",
        "run_revision": "run_writing",
    }
    if node_name in fixed:
        return fixed[node_name]
    if node_name == "run_search_agent":
        return "compose_reply" if _route_after_search(state) == "reply" else "run_citation_discovery"
    if node_name == "run_read":
        return "compose_reply" if _route_after_read(state) == "reply" else "run_matrix"
    if node_name == "run_matrix":
        return "run_supplement" if route_after_matrix(state) == "supplement" else "run_matrix_review"
    if node_name == "run_supplement":
        return "run_read" if route_after_supplement(state) == "read" else "run_matrix_review"
    if node_name == "run_audit":
        return "run_revision" if route_after_audit(state) == "revise" else "compose_reply"
    return None


async def run_graph(
    request: ReviewRequest,
    *,
    runtime: WorkflowRuntimeContext | None = None,
    session_repo: SessionRepository | None = None,
    session_key: str | None = None,
    turn_id: str | None = None,
    state_overrides: dict[str, Any] | None = None,
) -> GraphRunResult:
    """异步运行执行图，并把运行上下文一并注入共享状态。"""

    graph = build_graph()
    initial_state = _build_initial_state(
        request,
        runtime=runtime,
        session_repo=session_repo,
        session_key=session_key,
        turn_id=turn_id,
        state_overrides=state_overrides,
    )
    # 中文说明：检索还没完成就失败时，也需要有一个能重新进入检索的起点。
    # 新任务和恢复任务都记录本轮入口，重复点击“继续”时可找到稳定位置。
    await save_checkpoint(initial_state, _entrypoint(initial_state))
    final_state = await graph.ainvoke(initial_state, config={"recursion_limit": 32})
    papers = list(final_state.get("search_results") or [])
    diagnostics = dict(final_state.get("diagnostics") or {})
    return GraphRunResult(
        papers=papers,
        state=dict(final_state),
        diagnostics=diagnostics,
    )


def run_graph_sync(
    request: ReviewRequest,
    *,
    runtime: WorkflowRuntimeContext | None = None,
    session_repo: SessionRepository | None = None,
    session_key: str | None = None,
    turn_id: str | None = None,
    state_overrides: dict[str, Any] | None = None,
) -> GraphRunResult:
    """给旧同步入口保留一个很薄的兼容壳。"""

    with asyncio.Runner() as runner:
        return runner.run(
            run_graph(
                request,
                runtime=runtime,
                session_repo=session_repo,
                session_key=session_key,
                turn_id=turn_id,
                state_overrides=state_overrides,
            )
        )


def _build_initial_state(
    request: ReviewRequest,
    *,
    runtime: WorkflowRuntimeContext | None,
    session_repo: SessionRepository | None,
    session_key: str | None,
    turn_id: str | None,
    state_overrides: dict[str, Any] | None,
) -> State:
    """整理图执行需要的初始状态，避免同步和异步入口各自拼一遍。"""

    initial_state = State(
        request=request,
        search_results=[],
        search_scores=[],
        search_summary={},
        search_output={},
        search_artifact_refs=[],
        read_results=[],
        read_summary={},
        read_artifact_refs=[],
        analysis_report={},
        analysis_artifact_refs=[],
        writing_outline={},
        writing_outline_report={},
        writing_outline_artifact_refs=[],
        writing_sections=[],
        writing_partial_sections=[],
        writing_report={},
        writing_artifact_refs=[],
        final_artifact_refs=[],
        diagnostics={},
        current_step="init",
        origin_turn_id=str(turn_id or ""),
        assistant_message="",
        assistant_message_metadata={},
    )

    # 中文注释：直接从脚本调用且传入会话时，也建立最小进度上报能力，
    # 这样图里产生的进度和产物仍然会落到同一个会话仓库里。
    if runtime is None and session_repo is not None and session_key and turn_id:
        runtime = WorkflowRuntimeContext(
            session_key=session_key,
            turn_id=turn_id,
            workflow_name="paper_graph",
            sync_port=InlineWorkflowSyncPort(
                _build_repository_emitter(session_repo, session_key),
                session_key=session_key,
                turn_id=turn_id,
                workflow_name="paper_graph",
            ),
        )

    if session_repo is not None:
        initial_state["session_repo"] = session_repo
    if session_key:
        initial_state["session_key"] = session_key
    if turn_id:
        initial_state["turn_id"] = turn_id
    if runtime is not None:
        initial_state["runtime_context"] = runtime
    if state_overrides:
        initial_state.update(state_overrides)
    workflow_checkpoint = initial_state.get("workflow_checkpoint")
    if isinstance(workflow_checkpoint, dict):
        return restore_checkpoint(initial_state, workflow_checkpoint)
    return _merge_read_checkpoint(initial_state)


def _merge_read_checkpoint(state: State) -> State:
    """把恢复现场中的请求、检索结果和已读结果合并回初始状态。"""

    checkpoint = state.get("read_resume_checkpoint")
    if not isinstance(checkpoint, dict):
        return state
    request_payload = checkpoint.get("request")
    if isinstance(request_payload, dict) and str(request_payload.get("topic") or "").strip():
        state["request"] = ReviewRequest(
            topic=str(request_payload.get("topic") or ""),
            constraints=dict(request_payload.get("constraints") or {}),
            language=str(request_payload.get("language") or "zh"),
        )
    search_results = _papers_from_checkpoint(checkpoint.get("search_results"))
    if search_results and not state.get("search_results"):
        state["search_results"] = search_results
    if checkpoint.get("read_results") and not state.get("read_results"):
        state["read_results"] = list(checkpoint.get("read_results") or [])
    if checkpoint.get("read_artifact_refs") and not state.get("read_artifact_refs"):
        state["read_artifact_refs"] = list(checkpoint.get("read_artifact_refs") or [])
    # 阅读因资源暂停时，还要恢复补搜预算和矩阵，防止恢复后再次补搜。
    for key in ("research_round", "audit_revision", "supplemental_paper_ids", "supplemental_search", "evidence_matrix", "research_artifact_refs", "search_summary", "search_output", "citation_graph", "read_summary", "search_artifact_refs"):
        if key in checkpoint:
            state[key] = checkpoint[key]
    # 中文注释：旧 checkpoint 只有阅读模型恢复一种情况，所以这里以前固定写成
    # read_waiting_model。现在 embedding 也可能等待恢复，优先使用 checkpoint 自带步骤。
    state["current_step"] = str(checkpoint.get("current_step") or "read_waiting_model")
    return state


def _papers_from_checkpoint(value: Any) -> list[PaperDocument]:
    """从阅读 checkpoint 恢复检索论文列表。"""

    if not isinstance(value, list):
        return []
    papers: list[PaperDocument] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        paper_id = str(item.get("id") or "").strip()
        title = str(item.get("title") or "").strip()
        if not paper_id or not title:
            continue
        papers.append(
            PaperDocument(
                id=paper_id,
                title=title,
                authors=[str(author).strip() for author in item.get("authors") or [] if str(author).strip()],
                abstract=str(item.get("abstract")) if item.get("abstract") is not None else None,
                year=_optional_int(item.get("year")),
                venue=str(item.get("venue")) if item.get("venue") is not None else None,
                url=str(item.get("url")) if item.get("url") is not None else None,
                pdf_url=str(item.get("pdf_url")) if item.get("pdf_url") is not None else None,
                doi=str(item.get("doi")) if item.get("doi") is not None else None,
                source=str(item.get("source")) if item.get("source") is not None else None,
                paperId=str(item.get("paperId")) if item.get("paperId") is not None else None,
                publication_date=str(item.get("publication_date") or ""),
                journal_conference=str(item.get("journal_conference") or item.get("journal/conference") or ""),
                volume=str(item.get("volume") or ""),
                issue=str(item.get("issue") or ""),
                language=str(item.get("language") or ""),
                metadata=dict(item.get("metadata") or {}) if isinstance(item.get("metadata"), dict) else {},
            )
        )
    return papers


def _optional_int(value: Any) -> int | None:
    """把 checkpoint 里的可选数字恢复为整数。"""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_repository_emitter(repo: SessionRepository, session_key: str):
    """构造直接写入会话仓储的进度发送函数，供没有 API 外层的调用场景使用。"""

    def _emit(event: dict[str, Any]) -> dict[str, Any]:
        """把工作流事件写入会话记录后原样返回，保持同步端口的调用约定。"""

        repo.append_event(
            session_key,
            str(event.get("event") or "workflow_event"),
            content=str(event.get("message") or event.get("content") or ""),
            metadata=dict(event),
        )
        return event

    return _emit
