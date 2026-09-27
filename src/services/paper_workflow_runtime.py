from __future__ import annotations

from typing import Any

from src.agents import ReviewRequest
from src.graph import run_graph
from src.models.sessions import utc_now
from src.graph.runtime import InlineWorkflowSyncPort, WorkflowCancellation, WorkflowRuntimeContext
from src.repositories.sessions.base import SessionRepository
from src.services.sessions import RuntimeEventEmitter
from src.services.conversation import (
    answer_chat, checkpoint_for_edit, classify_ambiguous_message, classify_message, latest_artifact_json,
    latest_report_checkpoint, latest_workflow_checkpoint, legacy_report_checkpoint,
    missing_body_sections, missing_or_unverified_sections,
    select_sections,
)


JsonObject = dict[str, Any]


def build_paper_workflow_message_handler(repo: SessionRepository):
    """构建把会话消息交给论文工作流执行的处理器。"""

    async def _handler(chat_id: str, content: str, frame: JsonObject, emit: RuntimeEventEmitter) -> None:
        """把会话输入交给异步工作流执行，并把运行时能力注入图状态。"""

        turn_id = str(frame.get("turn_id") or "")
        run_id = str(frame.get("run_id") or "") or None
        runtime = frame.get("runtime_context")
        if not isinstance(runtime, WorkflowRuntimeContext):
            # 中文注释：同步接口没有独立的 run service，所以这里兜底创建一个本地运行上下文。
            runtime = WorkflowRuntimeContext(
                session_key=chat_id,
                run_id=run_id,
                turn_id=turn_id,
                workflow_name="paper_graph",
                sync_port=InlineWorkflowSyncPort(
                    emit,
                    session_key=chat_id,
                    run_id=run_id,
                    turn_id=turn_id,
                    workflow_name="paper_graph",
                ),
                cancellation=frame.get("cancellation") if isinstance(frame.get("cancellation"), WorkflowCancellation) else None,
            )

        record = repo.get(chat_id)
        report = latest_artifact_json(repo, record, "writing")
        audit = latest_artifact_json(repo, record, "citation_audit")
        if report and audit:
            report["citation_audit_status"] = audit.get("status")
        explicit_workflow = frame.get("workflow_checkpoint")
        read_checkpoint = frame.get("read_resume_checkpoint")
        kind = "resume" if isinstance(explicit_workflow, dict) or isinstance(read_checkpoint, dict) else classify_message(content, has_report=bool(report))
        if kind == "ambiguous":
            kind = await classify_ambiguous_message(content)

        def reply(message: str) -> None:
            """普通对话也发标准助手事件，让前端正常结束当前回合。"""

            emit({"event": "message", "role": "assistant", "content": message,
                  "turn_id": turn_id, "timestamp": utc_now()})

        if kind == "chat":
            reply(await answer_chat(content, record, report))
            return
        if kind == "pause":
            reply("当前没有正在执行的报告任务。若任务正在运行，请使用输入框旁的停止按钮。")
            return

        workflow_checkpoint = explicit_workflow if isinstance(explicit_workflow, dict) else None
        if kind in {"resume", "repair"} and workflow_checkpoint is None and not isinstance(read_checkpoint, dict):
            # 中文说明：先使用阅读节点保存的更细断点，再使用通用阶段断点。
            # 这样阅读到一半失败时无需重新处理已经完成的论文。
            read_checkpoint = _latest_legacy_read_checkpoint(record)
            if read_checkpoint is None:
                workflow_checkpoint = latest_workflow_checkpoint(repo, record)
                if workflow_checkpoint is None:
                    targets = (missing_or_unverified_sections(report or {}, audit) if kind == "repair"
                               else missing_body_sections(report or {}))
                    if report and targets:
                        workflow_checkpoint = checkpoint_for_edit(
                            legacy_report_checkpoint(repo, record, report), report,
                            targets=targets, instruction="继续补写缺失或待核查的小节。",
                            allow_auto_revision=kind == "repair")
                    else:
                        reply("没有找到可继续的恢复现场。请重新发起调研，或指定要修改的报告章节。")
                        return
                source_turn = _latest_checkpoint_turn(record)
                source_status = _turn_status(record, source_turn)
                if source_status == "completed":
                    targets = (missing_or_unverified_sections(report or {}, audit) if kind == "repair"
                               else missing_body_sections(report or {}))
                    if not targets:
                        reply("上次报告已生成。可以直接提问、指定章节纠错；如需处理未通过的核查项，请发送“继续修复核查问题”。")
                        return
                    workflow_checkpoint = checkpoint_for_edit(
                        workflow_checkpoint, report or {}, targets=targets,
                        instruction="继续补写缺失小节或修复核查问题；保留其他章节。",
                        allow_auto_revision=kind == "repair",
                    )

        if kind == "edit":
            if not report:
                reply("当前会话尚无报告。请先提供调研主题。")
                return
            targets = select_sections(content, report)
            if not targets:
                reply("请指出要修改的章节、摘要或具体段落，例如“修正第三章第二节的结论”。")
                return
            base = latest_report_checkpoint(repo, record) or legacy_report_checkpoint(repo, record, report)
            workflow_checkpoint = checkpoint_for_edit(base, report, targets=targets, instruction=content)

        if kind == "refresh":
            base = latest_workflow_checkpoint(repo, record)
            previous_request = _request_from_checkpoint(base) or ReviewRequest(
                topic=str((report or {}).get("topic") or content), constraints={})
            extra = "" if content.strip() in {"重新检索", "重新调研", "更新文献"} else f"\n本次更新要求：{content.strip()}"
            request = ReviewRequest(
                topic=previous_request.topic + extra,
                constraints={**previous_request.constraints, **_constraints_from_frame(frame)},
                language=previous_request.language,
            )
            state_overrides = None
        else:
            request = (_request_from_checkpoint(workflow_checkpoint)
                       or _request_from_checkpoint(read_checkpoint)
                       or ReviewRequest(topic=content, constraints=_constraints_from_frame(frame)))
            state_overrides = (
                {"workflow_checkpoint": workflow_checkpoint} if isinstance(workflow_checkpoint, dict)
                else {"read_resume_checkpoint": read_checkpoint} if isinstance(read_checkpoint, dict)
                else None
            )
        await run_graph(
            request,
            runtime=runtime,
            session_repo=repo,
            session_key=chat_id,
            turn_id=turn_id,
            state_overrides=state_overrides,
        )

    return _handler


def _latest_checkpoint_turn(record: Any) -> str:
    """找到最近一个通用断点所属的原任务回合。"""

    for artifact in reversed(record.artifacts):
        if artifact.get("artifact_type") == "workflow_checkpoint":
            return str((artifact.get("metadata") or {}).get("turn_id") or "")
    return ""


def _turn_status(record: Any, turn_id: str) -> str:
    """查询指定回合是否已经正常结束。"""

    for event in reversed(record.events):
        metadata = event.get("metadata") or {}
        if event.get("event_type") == "turn_end" and str(metadata.get("turn_id") or "") == turn_id:
            return str(metadata.get("status") or "")
    return ""


def _latest_legacy_read_checkpoint(record: Any) -> JsonObject | None:
    """兼容已有阅读断点；只读取最近一次未完成调研的恢复现场。"""

    for event in reversed(record.events):
        if event.get("event_type") == "turn_end" and (event.get("metadata") or {}).get("status") == "completed":
            break
        metadata = event.get("metadata") or {}
        runtime_metadata = metadata.get("metadata") or {}
        checkpoint = runtime_metadata.get("checkpoint") or metadata.get("checkpoint")
        if isinstance(checkpoint, dict) and checkpoint.get("request"):
            return checkpoint
    return None


def _constraints_from_frame(frame: JsonObject) -> JsonObject:
    """从一次会话请求中取出约束；没有传约束时返回空字典。"""

    value = frame.get("constraints")
    return dict(value) if isinstance(value, dict) else {}


def _request_from_checkpoint(checkpoint: Any) -> ReviewRequest | None:
    """恢复执行时优先沿用 checkpoint 里的原始请求。"""

    if not isinstance(checkpoint, dict):
        return None
    payload = checkpoint.get("request")
    if not isinstance(payload, dict):
        return None
    topic = str(payload.get("topic") or "").strip()
    if not topic:
        return None
    return ReviewRequest(
        topic=topic,
        constraints=dict(payload.get("constraints") or {}),
        language=str(payload.get("language") or "zh"),
    )
