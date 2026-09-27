from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from src.agents.contracts import ReviewRequest
from src.paper_retrieval.models import PaperDocument

from .state_models import State


# 中文说明：这里只保存恢复执行真正需要的数据。仓库、模型连接和运行时对象
# 都由新一轮请求重新创建，不能写入磁盘，也不能把旧连接带进新的进程。
SAVED_FIELDS = (
    "search_results", "search_scores", "search_summary", "search_output", "search_artifact_refs",
    "read_results", "read_summary", "read_artifact_refs", "read_paper_statuses",
    "conflict_report", "citation_graph", "evidence_matrix", "citation_audit",
    "supplemental_search", "supplemental_paper_ids", "research_round", "audit_revision",
    "research_artifact_refs", "analysis_report", "analysis_artifact_refs",
    "writing_outline", "writing_outline_report", "writing_outline_artifact_refs",
    "writing_sections", "writing_report", "writing_artifact_refs", "writing_partial_sections",
    "final_artifact_refs", "diagnostics", "current_step", "read_resume_checkpoint",
    "writing_target_ids", "writing_instruction", "origin_turn_id",
)


def _safe_value(value: Any) -> Any:
    """把论文对象等状态值转成普通 JSON 数据。"""

    if isinstance(value, PaperDocument):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _safe_value(value.to_dict())
    raise TypeError(f"恢复现场含有无法保存的字段：{type(value).__name__}")


def build_checkpoint(state: State, next_node: str) -> dict[str, Any]:
    """记录已经完成的工作和接下来要运行的节点。"""

    request = state.get("request")
    if request is None:
        raise ValueError("缺少原始调研请求，无法保存恢复现场")
    return {
        "schema_version": 1,
        "next_node": next_node,
        "origin_turn_id": str(state.get("origin_turn_id") or state.get("turn_id") or ""),
        "request": {
            "topic": request.topic,
            "constraints": dict(request.constraints or {}),
            "language": request.language,
        },
        "state": {key: _safe_value(state[key]) for key in SAVED_FIELDS if key in state},
    }


async def save_checkpoint(state: State, next_node: str) -> None:
    """把完成的阶段写成独立文件；一次写入失败会明确终止任务。"""

    repo = state.get("session_repo")
    session_key = str(state.get("session_key") or "")
    turn_id = str(state.get("turn_id") or "")
    if repo is None or not session_key or not turn_id:
        return
    checkpoint = build_checkpoint(state, next_node)
    name = f"{uuid.uuid4().hex}.json"
    await asyncio.to_thread(
        repo.write_artifact, session_key, "workflow_checkpoint", name,
        json.dumps(checkpoint, ensure_ascii=False),
        relative_path=f"artifacts/workflow_checkpoint/{turn_id}/{name}",
        metadata={"turn_id": turn_id, "next_node": next_node, "schema_version": 1},
    )


def restore_checkpoint(state: State, checkpoint: dict[str, Any]) -> State:
    """从文件恢复数据；模型和仓库仍使用本轮新建的运行环境。"""

    if checkpoint.get("schema_version") != 1 or not isinstance(checkpoint.get("state"), dict):
        raise ValueError("恢复现场格式不受支持")
    payload = dict(checkpoint["state"])
    request = dict(checkpoint.get("request") or {})
    if not str(request.get("topic") or "").strip():
        raise ValueError("恢复现场缺少原始主题")
    state.update(payload)
    state["request"] = ReviewRequest(
        topic=str(request["topic"]),
        constraints=dict(request.get("constraints") or {}),
        language=str(request.get("language") or "zh"),
    )
    papers = []
    for item in list(state.get("search_results") or []):
        if not isinstance(item, dict) or not item.get("id") or not item.get("title"):
            continue
        # 中文说明：论文模型没有 from_dict；只取它定义过的字段，
        # 避免 to_dict 中的展示别名影响恢复。
        values = {key: item.get(key) for key in PaperDocument.__dataclass_fields__ if key in item}
        values["year"] = int(item["year"]) if str(item.get("year") or "").isdigit() else None
        papers.append(PaperDocument(**values))
    state["search_results"] = papers
    state["workflow_resume_node"] = str(checkpoint.get("next_node") or "")
    state["origin_turn_id"] = str(checkpoint.get("origin_turn_id") or state.get("turn_id") or "")
    return state
