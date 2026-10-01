from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, cast

from src.agents.writingAgent import (
    WritingAgent,
    build_writing_agent,
    load_writing_agent_llm,
)
from src.graph.runtime import WorkflowRuntimeContext
from src.graph.runtime_resources import WorkflowRuntimeResources
from src.graph.state_models import JsonObject, State
from src.graph.workflow_checkpoint import save_checkpoint
from src.llm import ProviderSnapshot, SystemConfig
from src.models.sessions import utc_now
from src.paper_retrieval.models import PaperDocument
from src.repositories.sessions.base import SessionRepository
from src.utils.read_utils.cache import safe_cache_name


# 中文说明：本版本开始，写作产物同时包含摘要和按正文引用整理的参考文献。
WRITING_VERSION = "1.1"


def run_writing_node():
    """生成论文正文写作节点。

    中文说明：
    大纲节点只负责“怎么写”，这个节点负责“真正写出来”。
    它会把大纲里的每个小节当成一个独立写作任务，按顺序交给 WritingAgent。
    """

    async def _node(state: State) -> State:
        """按大纲顺序逐节写作，并把结果保存进共享状态。"""

        request = state.get("request")
        if request is None:
            raise ValueError("写作节点缺少用户综述主题，无法继续生成正文")
        outline = dict(state.get("writing_outline") or {})
        if not outline:
            raise ValueError("写作节点缺少写作大纲，无法知道要写哪些小节")

        reporter = _resolve_reporter(state)
        llm = _resolve_llm(state)
        agent = build_writing_agent(llm)
        search_results = list(state.get("search_results") or [])
        read_results = list(state.get("read_results") or [])
        # State 可能只带有本轮阅读结果，因此额外读取同一会话历史轮次的论文笔记。
        # 这样从已有会话恢复写作时，也能找到没有进入本轮 State 的论文摘要。
        session_read_results = await asyncio.to_thread(_load_session_read_results, state)
        cache_dir = SystemConfig.load().read.paper_cache_dir
        available_paper_ids = _collect_available_paper_ids(
            read_results=read_results,
            session_read_results=session_read_results,
        )
        section_tasks = _flatten_outline(outline)
        # 中文说明：模型大纲虽收到“简明、单一机制”的要求，本轮仍把每节写成
        # 350 到 400 字，作者为填篇幅加入了实验数据集与具体采样配置。用户没有
        # 指定每节字数时，把大纲字数当上限收紧；若用户明确给出字数，则尊重用户。
        topic_text = str(request.topic or "")
        named_arxiv_ids = list(dict.fromkeys(re.findall(r"arXiv\s*:\s*(\d{4}\.\d{4,5})", topic_text, re.IGNORECASE)))
        brief_mechanism = "简明" in topic_text and "只讨论" in topic_text
        # 中文说明：视觉 Transformer 任务同样明确要求“简明综述”并点名五篇，
        # 但没有逐篇写“只讨论”。正文仍尊重大纲字数；摘要则应沿用简明任务
        # 的保守三句范围，避免第 35 轮扩写无逐篇证据的跨方法分类标签。
        brief_abstract = "简明" in topic_text and len(named_arxiv_ids) >= 3
        explicit_section_length = re.search(r"(?:每节|每小节|每段)\s*(?:约|不超过|至少)?\s*\d{2,4}\s*字", topic_text)
        if brief_mechanism and not explicit_section_length:
            for section_task in section_tasks:
                section_task["word_count"] = min(int(section_task["word_count"]), 240)
        written_sections: list[JsonObject] = []
        target_ids = {str(value) for value in (state.get("writing_target_ids") or [])}
        partial_sections = {
            str(section.get("section_id") or ""): section
            for section in (state.get("writing_partial_sections") or [])
            if isinstance(section, dict)
        }
        # 中文说明：独立核查后的第二轮只重写出问题的小节。旧版会把所有小节
        # 重新写一遍，长综述的 token 和用时几乎翻倍；通过的小节从上一轮报告
        # 取回带真实论文编号的原始正文，仍会随整篇报告再次接受独立核查。
        prior_sections = {
            str(section.get("section_id") or ""): section
            for section in (state.get("writing_report") or {}).get("sections", [])
            if isinstance(section, dict)
        } if (state.get("audit_revision") and (state.get("citation_audit") or {}).get("sections")) or target_ids else {}
        audit_issues = {
            str(section.get("section_id") or "")
            for section in (state.get("citation_audit") or {}).get("sections", [])
            if section.get("status") != "passed"
        }
        audit_issues.update(str(value) for value in (state.get("citation_audit") or {}).get("generation_errors", []))
        audit_issues.update(str(value) for value in (state.get("citation_audit") or {}).get("review_errors", []))
        if target_ids:
            # 中文说明：用户指定重写的章节必须进入写作；其他章节直接沿用
            # 上一版正文，避免一句纠错指令把整篇报告重新调用模型。
            audit_issues = target_ids

        if reporter is not None:
            reporter.started(
                f"准备撰写 {len(section_tasks)} 个小节",
                stage="writing_start",
                total=len(section_tasks),
            )

        for index, section_task in enumerate(section_tasks, start=1):
            prior = prior_sections.get(str(section_task["section_id"]))
            partial = partial_sections.get(str(section_task["section_id"]))
            if partial and str(partial.get("content") or "").strip() != "本节正文未能生成，需重新生成并核查。":
                # 中文说明：进程在后续小节中断时，已保存的小节可直接沿用。
                written_sections.append(dict(partial))
                if reporter is not None:
                    reporter.progress("已从断点恢复小节正文", stage="writing_section_done",
                        runtime_status="completed", completed=index, total=len(section_tasks),
                        section_id=section_task["section_id"],
                        event_key=f"writing_section:{section_task['section_id']}",
                        stage_title=str(section_task.get("section_title") or section_task["section_id"]))
                continue
            if prior and section_task["section_id"] not in audit_issues and prior.get("source_content"):
                reused = {**prior, "content": prior["source_content"]}
                written_sections.append(reused)
                if reporter is not None:
                    reporter.progress("上一轮已通过核查，复用小节正文", stage="writing_section_done",
                        runtime_status="completed", completed=index, total=len(section_tasks),
                        section_id=section_task["section_id"],
                        event_key=f"writing_section:{section_task['section_id']}",
                        stage_title=str(section_task.get("section_title") or section_task["section_id"]))
                await save_checkpoint({**state, "writing_partial_sections": written_sections}, "run_writing")
                continue
            section_usage = {"input_tokens": 0, "output_tokens": 0}

            def report_section_phase(message: str, usage: JsonObject | None = None, *, task=section_task) -> None:
                """把当前小节的内部阶段更新到它自己的卡片上。"""

                if reporter is not None:
                    if usage:
                        section_usage["input_tokens"] += int(usage.get("input_tokens") or 0)
                        section_usage["output_tokens"] += int(usage.get("output_tokens") or 0)
                    reporter.progress(
                        message,
                        stage="writing_section",
                        completed=index - 1,
                        total=len(section_tasks),
                        section_id=task["section_id"],
                        event_key=f"writing_section:{task['section_id']}",
                        stage_title=str(task.get("section_title") or task["section_id"]),
                        **(section_usage if usage else {}),
                    )

            if reporter is not None:
                reporter.progress(
                    "正在撰写小节正文",
                    stage="writing_section",
                    completed=index - 1,
                    total=len(section_tasks),
                    section_id=section_task["section_id"],
                    # 中文说明：每个小节都使用独立事件键，避免所有小节被前端合并成一张卡。
                    event_key=f"writing_section:{section_task['section_id']}",
                    stage_title=str(section_task.get("section_title") or section_task["section_id"]),
                )
            previous_sections = _resolve_previous_sections(
                requested_refs=list(section_task.get("ref_sections") or []),
                written_sections=written_sections,
            )
            # 中文说明：第十三轮的综合分析错误地把 GCN 的线性扩展写成内存瓶颈。
            # 这类模型归纳不再送给正文作者；大纲里的 evidence-map 仍可用于规划，
            # 写作时只给原文切片位置，让作者自己打开原文核对。
            section_evidence: list[JsonObject] = []
            matrix_rows = (state.get("evidence_matrix") or {}).get("rows") or []
            # 中文说明：矩阵只是检索入口，不能证明某项实验不存在。小节任务若
            # 明确点名论文，就只带这些论文的矩阵行，避免每次模型调用反复发送
            # 五篇乃至更多无关论文；综合小节未点名时仍保留全部行，不截掉末尾论文。
            evidence_text = str(section_task.get("task") or "")
            related_rows = [row for row in matrix_rows if str(row.get("paperId") or "")
                            and str(row["paperId"]).casefold() in evidence_text.casefold()]
            selected_rows = related_rows or matrix_rows
            matrix_excerpt = json.dumps([
                {"paperId": row.get("paperId"), "chunkIds": list(dict.fromkeys(
                    str(citation.get("chunkId") or "")
                    for key, cell in (row.get("cells") or {}).items()
                    if key in {"method", "evaluation_data", "metrics", "results", "baselines"}
                    for citation in (cell.get("evidence") or [])
                    if isinstance(citation, dict) and citation.get("chunkId")
                ))} for row in selected_rows
            ], ensure_ascii=False)
            if matrix_rows:
                section_evidence.append({"field": "实证矩阵的原文位置（不是事实结论）", "content": matrix_excerpt})
            # 中文说明：第 23 轮 GCN 已有方法章节的传播规则原文，但作者
            # 只打开摘要和实验切片，最后偏写训练架构。阅读提取中保存的
            # 方法切片编号只作为“去哪里查”的入口；正文仍必须用工具打开
            # 原文，逐句绑定真实子片段，不能把提取说明当成事实证明。
            selected_paper_ids = {str(row.get("paperId") or "").casefold() for row in selected_rows}
            method_locations: list[JsonObject] = []
            seen_method_papers: set[str] = set()
            for result in [*read_results, *session_read_results]:
                paper = result.get("paper") or {}
                paper_id = str(paper.get("paperId") or paper.get("id") or "")
                if not paper_id or paper_id.casefold() not in selected_paper_ids or paper_id.casefold() in seen_method_papers:
                    continue
                methods = str((result.get("extraction") or {}).get("methods") or "")
                chunk_ids = list(dict.fromkeys(
                    marker for marker in re.findall(r"\[([^\[\]\n]+)\]", methods)
                    if marker.casefold().startswith(paper_id.casefold() + ":")
                ))[:6]
                if chunk_ids:
                    method_locations.append({"paperId": paper_id, "chunkIds": chunk_ids})
                    seen_method_papers.add(paper_id.casefold())
            if method_locations:
                section_evidence.append({"field": "阅读阶段定位的方法原文入口（不是事实结论）",
                                         "content": json.dumps(method_locations, ensure_ascii=False)})
            # 中文说明：第十三轮的大纲 task 把 GCN 的线性扩展误写成内存瓶颈，
            # 即使提醒作者“仅作线索”，实际正文仍照抄。大纲只决定章节结构、
            # 标题和字数；正文的事实范围由用户原始要求与可核实原文决定。
            task_text = ("用户原始要求（唯一内容边界）：" + str(request.topic or "")[:1600]
                         + "\n当前小节标题：" + str(section_task.get("section_title") or section_task["section_id"]))
            task_text += "\n写作范围：只回答用户要求中与本小节标题对应的问题。大纲任务细节和前文均未通过原文核查，不得补入未要求的方法痛点、效率、硬件或研究空白。每句论文事实先查原文，证据不支持就删去；已索引原文是否缺少某数字须检查实验章节、表格与附录，未查全不得断言不存在。"
            comparison_title = str(section_task.get("section_title") or "")
            is_comparison = any(word in comparison_title for word in ("比较", "对比", "综合", "差异", "对照"))
            if brief_mechanism:
                # 中文说明：“不同任务不直接排名”是整篇综述的写作边界，不是每篇
                # 原论文自己做出的结论。单论文节只讲指定机制；比较节也不能把
                # 作者的写作选择伪装成有原文引用的科研发现。
                # 中文说明：第二十轮虽给每句真实切片，作者仍在“注意力系数”后
                # 自行补出“增强关键结构特征”，或把一阶谱近似扩写成未经该引句
                # 证明的加权平均。这里要求每个分句都能由短原文直接推出，
                # 不能用真实切片为同句中额外的效果、目的或最优性背书。
                task_text += "\n每篇机制句只陈述原文直接写明的操作；原文只说明注意力系数、求和聚合或一阶近似时，不自行补写该操作提高何种能力、保留何种信息或实现何种加权平均。若确需写效果、目的或理论最优性，必须另找能直接说明这一关系的原文，并单独成句引用；找不到就删去该从句。不要在小节结尾补写‘这一设计保持架构优势、融入归纳偏置、提升能力’之类没有对应原文的概括句；已有四句直接机制证据时可以就此结束。"
                if is_comparison:
                    task_text += "\n本节只比较用户指定机制的设计方式，不把标题里的‘适用边界’扩展成原论文局限清单。不得添写用户未要求的内存、硬件、边特征、有向图、网络深度、复杂度或应用场景。若需交代跨任务限制，只写‘本文不作跨任务性能数值比较’，不要声称原论文证明这些任务绝对不可比较，也不要为此添写数据集清单。每篇原论文各写有直接原文证据的机制句即可；不要再添加‘上述方法均旨在弥补某一共同不足’等统括性开场或收尾，除非每条引用的原文都直接支持这项共同归因。"
                    if named_arxiv_ids:
                        # 中文说明：第十九轮综合节只写了五篇中的前三篇，虽有
                        # 正文却没有回答用户要求。明确点名的原论文必须逐篇覆盖；
                        # 每篇只需一句已核实的机制差异，不靠扩写局限凑字数。
                        task_text += "\n必须覆盖用户点名的每篇原论文，每篇只写一句受原文支持的机制差异，不能漏掉末尾论文：" + "、".join(named_arxiv_ids)
                else:
                    # 中文说明：真实大纲多次把用户“GIN 只讨论求和聚合”
                    # 扩成 WL 测试内部机制，把“GCN 一阶传播”扩成谱半径
                    # 等公式细节。用户在主题里逐篇写明的“只讨论”范围
                    # 比模型大纲标题更权威，单独提取给作者和本地审查。
                    matched_ids = [paper_id for paper_id in named_arxiv_ids if paper_id in evidence_text]
                    if len(matched_ids) == 1:
                        scope_match = re.search(
                            rf"[A-Za-z][A-Za-z0-9-]{{2,}}\s*[（(]\s*arXiv\s*:\s*{re.escape(matched_ids[0])}"
                            rf"\s*[）)]\s*只讨论[^；;。]*",
                            topic_text, re.IGNORECASE,
                        )
                        if scope_match:
                            task_text += "\n本节用户明确限定（优先于大纲标题的扩写）：" + scope_match.group(0)
                    task_text += "\n本单论文小节只解释用户明确限定的机制，大纲标题仅用于定位小节、不能扩大范围；不写实验数据集、每层实验参数、时间或空间复杂度、可扩展性、用户未点名的模型变体或理论例外、其他理论或跨任务比较警示。原文只说每批次复杂度固定时，不推成‘因此能够处理大规模图’等能力或因果结论。若关键计算步骤只能从乱码或损坏公式推测，而当前普通文字未直接说清，就不要写该步骤；可改写为普通文字确实描述、且能逐字定位原文的机制。若原文谈的是相关工作而非本文方法，不得将其操作归给本文方法。整篇综述的比较范围留给综合小节。"
            # 每次写作与审稿都读取小节任务，因此这里同时约束初稿和后续重写的语言。
            task_text += f"\n输出语言：{request.language}。正文使用该语言，保留必要的专有名词与原文引句。"
            if state.get("audit_revision"):
                # 把上一轮独立核查的具体问题交给写作模型，禁止靠改措辞保留错误事实。
                # 中文注释：审计结果里还包含已通过段落和整段原文引句。把它们全部
                # 塞回每次写作请求会挤占小节预算，也容易让模型继续改写没问题的句子。
                # 这里只传本节确实失败的主张与原因；原文仍可由全文工具重新核对。
                failed_units = [
                    {"status": unit.get("status"),
                     "claim": str(unit.get("claim") or "")[:400],
                     "reason": str(unit.get("reason") or "")[:500]}
                    for audit_section in (state.get("citation_audit") or {}).get("sections", [])
                    if audit_section.get("section_id") == section_task["section_id"]
                    for unit in audit_section.get("units", [])
                    if unit.get("status") != "supported"
                ]
                task_text += "\n独立核查要求：删除无法证实的事实或通过全文工具找到直接证据，不得仅降低语气掩盖错误。\n" + json.dumps(failed_units, ensure_ascii=False)
                if prior and (prior.get("review") or {}).get("passed") is False:
                    # 中文说明：独立审计通过也不代表本地逐句引用检查通过。
                    # 重写时带上上一轮的具体建议，让模型知道需要修正哪里。
                    task_text += "\n上一轮写作检查问题：" + json.dumps(prior["review"], ensure_ascii=False)[:6000]
            if prior and section_task["section_id"] in target_ids:
                task_text += "\n上一版正文（仅作为修订草稿，仍须核对证据）：\n" + str(prior.get("source_content") or prior.get("content") or "")[:6000]
                task_text += "\n用户本次修订要求：" + str(state.get("writing_instruction") or "重新生成本小节")[:2000]
            section_result = await agent.async_write_section(
                section_id=str(section_task["section_id"]),
                task=task_text,
                evidence_map=section_evidence,
                previous_sections=previous_sections,
                word_count=int(section_task.get("word_count") or 800),
                read_results=read_results,
                cache_dir=cache_dir,
                session_read_results=session_read_results,
                available_paper_ids=available_paper_ids,
                progress_callback=report_section_phase,
                # 中文说明：五篇论文的比较节在五万 Token 内常耗尽预算，
                # 导致最后两篇没有正文。仅这种明确多论文的简明比较节提高
                # 有界上限，普通单论文节不增加用量。
                token_budget=80000 if brief_mechanism and is_comparison and len(named_arxiv_ids) >= 4 else None,
            )
            new_content = str(section_result.get("content") or "").strip()
            old_content = str((prior or {}).get("source_content") or (prior or {}).get("content") or "").strip()
            if (state.get("audit_revision") and
                    (not new_content or new_content == "本节正文未能生成，需重新生成并核查。") and
                    old_content and old_content != "本节正文未能生成，需重新生成并核查。"):
                # 中文说明：修订失败不能把上一轮真实正文覆盖成占位文字。
                # 旧正文仍作为待核查草稿保存，同时记录本轮失败原因供界面显示。
                reason = str((section_result.get("review") or {}).get("message") or "未返回有效正文")
                section_result = {
                    **prior,
                    "content": old_content,
                    "source_content": old_content,
                    "review": {"passed": False, "status": "unverified",
                               "message": f"本轮修订失败，保留上一轮待核查正文：{reason}"},
                    "generation_failed": False,
                    "tool_results": section_result.get("tool_results") or prior.get("tool_results") or [],
                    "warnings": [*list(prior.get("warnings") or []),
                                 *list(section_result.get("warnings") or []),
                                 "本轮修订未生成正文，已恢复上一轮草稿"],
                }
            section_result.update(
                chapter_key=section_task["chapter_key"],
                section_key=section_task["section_key"],
                chapter_title=section_task.get("chapter_title") or section_task["chapter_key"],
                section_title=section_task.get("section_title") or section_task["section_key"],
                chapter_description=section_task.get("chapter_description") or "",
                ref_sections=list(section_task.get("ref_sections") or []),
            )
            if brief_mechanism and is_comparison and named_arxiv_ids:
                # 中文说明：引用审计只判断“写出的句子”是否有证据，不知道用户
                # 明确要求的五篇是否都被写到。这里只检查覆盖，不替模型补论文
                # 事实；缺少任一指定论文就保持待核查，并在修订时指出编号。
                body = str(section_result.get("source_content") or section_result.get("content") or "")
                missing_ids = [paper_id for paper_id in named_arxiv_ids if f"[{paper_id}]" not in body]
                if missing_ids:
                    old_review = dict(section_result.get("review") or {})
                    message = "综合节未覆盖用户指定原论文：" + "、".join(missing_ids)
                    if old_review.get("message"):
                        message += "；原检查：" + str(old_review["message"])
                    section_result["review"] = {
                        **old_review,
                        "passed": False,
                        "suggestions": [*list(old_review.get("suggestions") or []), message],
                        "message": message,
                    }
                    section_result["warnings"] = [*list(section_result.get("warnings") or []), message]
            written_sections.append(section_result)
            await save_checkpoint({**state, "writing_partial_sections": written_sections}, "run_writing")
            if reporter is not None:
                content = str(section_result.get("content") or "").strip()
                missing_body = not content or content == "本节正文未能生成，需重新生成并核查。"
                review = section_result.get("review") or {}
                review_failed = review.get("passed") is False
                section_message = (
                    "小节正文未生成，已保留待补写提示" if missing_body else
                    "小节正文已保存，但写作检查未通过" if review_failed else
                    "小节正文已完成"
                )
                reporter.progress(
                    section_message,
                    stage="writing_section_done",
                    runtime_status="failed" if missing_body or review_failed else "completed",
                    completed=index,
                    total=len(section_tasks),
                    section_id=section_task["section_id"],
                    cited_paper_count=len(section_result.get("cited_paper_ids") or []),
                    error_message=str(review.get("message") or "") if missing_body or review_failed else "",
                    # 中文说明：完成事件沿用开始时的小节事件键，前端才能更新原卡片。
                    event_key=f"writing_section:{section_task['section_id']}",
                    stage_title=str(section_task.get("section_title") or section_task["section_id"]),
                )

        if reporter is not None:
            reporter.progress("正在根据正文生成摘要", stage="writing_abstract")
        def report_abstract_usage(usage: JsonObject) -> None:
            """把摘要模型的真实 token 用量更新到摘要卡片。"""

            if reporter is not None:
                reporter.progress("摘要模型调用完成", stage="writing_abstract", **usage)

        requested_length = re.search(r"(\d{2,4})\s*字", str(state.get("writing_instruction") or ""))
        abstract_instruction = str(state.get("writing_instruction") or "") if "abstract" in target_ids else ""
        if brief_abstract:
            # 中文说明：多篇简明综述若把五个方法的机制全部塞进摘要首句，
            # 独立审计很难逐项找到足够原文。限定为范围句、一个已核实
            # 的机制句和比较边界句，仍须由后续审计检查具体事实。
            abstract_instruction += "\n严格只写三句。第一句只列本文覆盖的方法名称，不逐项解释各方法机制。第二句只写一个正文已有直接原文支持的具体机制事实，不推断历史演进、研究共识或显著差异。第三句只写本文的比较范围。"
        if brief_abstract and ("跨任务" in topic_text or "不同任务" in topic_text):
            # 中文说明：用户要求避免不同任务的数值排名，是这篇综述的写作
            # 选择，不是原论文证明“五种方法绝对不能比较”。第二十二轮
            # 摘要把选择写成绝对科研结论，独立审计因此正确拦下。
            abstract_instruction += "\n若需说明比较边界，只写‘本文不作跨任务性能数值比较’，不要写‘方法不可直接比较’‘研究显著不同’或暗示原论文证明了绝对不可比。"
        if state.get("audit_revision"):
            # 中文说明：旧版只把独立审计的失败理由交给正文小节，摘要每次都在
            # 不知道自己哪里错的情况下重写，常再次把“不作排名”说成论文结论。
            # 这里只带上摘要确实失败的句子与原因，不把审计模型的判词当证据。
            failed_abstract = [
                {"claim": str(unit.get("claim") or "")[:180],
                 "reason": str(unit.get("reason") or "")[:260]}
                for audit_section in (state.get("citation_audit") or {}).get("sections", [])
                if audit_section.get("section_id") == "abstract"
                for unit in audit_section.get("units", [])
                if unit.get("status") not in {"supported", "not_required"}
            ]
            if failed_abstract:
                if not brief_mechanism:
                    # 中文说明：简明任务会在下方改用已支持原句作摘要输入。
                    # 此时不再转发上轮失败句，以免模型照搬无证据局限。
                    abstract_instruction += "\n上轮摘要独立核查未通过；请删除或收窄以下具体句子，不能凭前文或常识保留原结论：" + json.dumps(failed_abstract[:8], ensure_ascii=False)
        abstract_sections = written_sections
        if brief_abstract and state.get("audit_revision"):
            # 中文说明：第十八轮摘要修订虽看到失败理由，仍照搬上一版未经证实的
            # 五篇局限。修订摘要只能阅读“上一轮独立审计支持、且本轮正文仍原样
            # 保留”的事实句；改写过或审计未过的句子不能借旧版正文混进摘要。
            supported_by_section = {
                str(section.get("section_id") or ""): [
                    str(unit.get("claim") or "") for unit in section.get("units") or []
                    if unit.get("status") == "supported" and str(unit.get("claim") or "").strip()
                ]
                for section in (state.get("citation_audit") or {}).get("sections", [])
            }
            abstract_sections = [
                {**section, "content": " ".join(
                    claim for claim in supported_by_section.get(str(section.get("section_id") or ""), [])
                    if claim in str(section.get("source_content") or section.get("content") or "")
                )}
                for section in written_sections
            ]
            # 中文说明：没有原样保留的已支持句，不等于该论文没有原文。
            # 模型仍可从用户主题得知综述覆盖了哪些方法，但不能因此补写
            # 这轮尚未核查的新事实，也不能把内部取证状态写进摘要。
            abstract_instruction += "\n上轮摘要有证据不足的事实句，不得沿用。只根据下方正文写；其他方法仅列为本文覆盖范围，不介绍细节。摘要不要评论证据状态。"
        prior_writing = state.get("writing_report") or {}
        prior_abstract_audit_passed = any(
            section.get("section_id") == "abstract" and section.get("status") == "passed"
            for section in (state.get("citation_audit") or {}).get("sections", [])
        )
        reuse_abstract = (
            bool(state.get("audit_revision")) and "abstract" not in target_ids
            and prior_abstract_audit_passed
            and bool(str(prior_writing.get("abstract") or "").strip())
            and (prior_writing.get("execution_metadata") or {}).get("abstract_status") == "ok"
        )
        if reuse_abstract:
            # 中文说明：修订只触及未通过的小节时，已由独立模型核查通过的摘要
            # 不应无故重写出新事实。沿用原文仍会在最终审计中重新逐句核查，
            # 不能把上一轮的通过标记直接当作最终通过。
            abstract, abstract_status = str(prior_writing["abstract"]), "ok"
        else:
            abstract, abstract_status = await agent.async_write_abstract(
                topic=request.topic,
                sections=abstract_sections,
                language=request.language,
                # 中文说明：简明综述若没有用户指定摘要字数，短摘要即可说明范围；
                # 旧的 300 字目标迫使模型逐篇扩写未充分取证的技术细节。
                word_count=int(requested_length.group(1)) if requested_length else (120 if brief_abstract else 300),
                instruction=abstract_instruction,
                usage_callback=report_abstract_usage,
            )

        # 中文说明：引用顺序以正文里 paperId 第一次出现的位置为准，
        # 这样最后的参考文献编号和正文阅读顺序一致。
        candidate_paper_ids = _extract_paper_ids_from_sections(written_sections)
        paper_metadata = _collect_paper_metadata(
            candidate_paper_ids,
            search_results=search_results,
            read_results=read_results,
            session_read_results=session_read_results,
            cache_dir=Path(str(cache_dir)),
        )
        valid_paper_id_keys = {
            paper_id
            for paper_id, metadata in paper_metadata.items()
            if _has_reference_metadata(metadata)
        }
        unknown_paper_ids = [
            paper_id
            for paper_id in candidate_paper_ids
            if paper_id.lower() not in valid_paper_id_keys
        ]
        # 中文说明：模型偶尔会把 P1 这类临时编号当成真实论文编号。
        # 这些编号在检索和阅读结果里找不到题名、作者等资料，不能出现在报告中，
        # 否则会被误排成一条看似正规的参考文献。
        written_sections = _remove_unknown_paper_citations(written_sections, unknown_paper_ids)
        abstract = _remove_unknown_citation_markers(abstract, unknown_paper_ids)
        cited_paper_ids = _extract_paper_ids_from_sections(written_sections)
        references = _build_references(cited_paper_ids, paper_metadata)
        if reporter is not None:
            # 中文说明：摘要生成结束后发送完成状态，避免摘要卡一直停留在“处理中”。
            reporter.progress(
                "摘要写作已完成",
                stage="writing_abstract",
                runtime_status="completed",
                abstract_status=abstract_status,
            )
            reporter.progress(
                "正在整理参考文献",
                stage="writing_references",
                cited_paper_count=len(cited_paper_ids),
            )

        writing_report = _build_writing_report(
            topic=request.topic,
            outline=outline,
            sections=written_sections,
            abstract=abstract,
            abstract_status=abstract_status,
            references=references,
            model_used=llm.model if isinstance(llm, ProviderSnapshot) else "unavailable",
        )

        if reporter is not None:
            # 中文说明：参考文献已经在内存中整理完毕，文件保存是后续独立步骤。
            reporter.progress(
                "参考文献已生成",
                stage="writing_references",
                runtime_status="completed",
                cited_paper_count=len(cited_paper_ids),
            )

        artifact_refs = list(state.get("writing_artifact_refs") or [])
        persisted = await _persist_writing_if_possible(state, writing_report)
        if persisted:
            artifact_refs.append(persisted)
            if reporter is not None:
                reporter.artifact(persisted, stage="writing_artifact_ready")

        diagnostics = dict(state.get("diagnostics") or {})
        missing_section_count = sum(
            not str(section.get("content") or "").strip()
            or str(section.get("content") or "").strip() == "本节正文未能生成，需重新生成并核查。"
            for section in written_sections
        )
        review_failed_section_count = sum(
            (section.get("review") or {}).get("passed") is False
            and bool(str(section.get("content") or "").strip())
            and str(section.get("content") or "").strip() != "本节正文未能生成，需重新生成并核查。"
            for section in written_sections
        )
        diagnostics["writing"] = {
            "status": "needs_review" if abstract_status != "ok" or missing_section_count or review_failed_section_count else "ok",
            "section_count": len(written_sections),
            "missing_section_count": missing_section_count,
            "review_failed_section_count": review_failed_section_count,
            "abstract_status": abstract_status,
            "reference_count": len(references),
            "ignored_unknown_paper_ids": unknown_paper_ids,
            "used_llm": isinstance(llm, ProviderSnapshot),
            "message": "写作产物已保存，是否可交付以独立引用核查为准",
        }

        if reporter is not None:
            reporter.completed(
                (f"写作产物已保存；{missing_section_count} 节缺少正文，{review_failed_section_count} 节写作检查未通过"
                 if missing_section_count or review_failed_section_count else "正文、摘要和参考文献已完成"),
                stage="writing_done",
                section_count=len(written_sections),
                missing_section_count=missing_section_count,
                review_failed_section_count=review_failed_section_count,
                reference_count=len(references),
            )

        updated = dict(state)
        updated.update(
            # 中文说明：报告生成后 section 内容已经换成参考文献序号，状态里也保存同一份内容，
            # 避免前端读取 writing_sections 时又看到旧的 [paperId] 引用。
            writing_sections=list(writing_report.get("sections") or written_sections),
            writing_partial_sections=[],
            writing_target_ids=[],
            writing_instruction="",
            writing_report=writing_report,
            writing_artifact_refs=artifact_refs,
            diagnostics=diagnostics,
            current_step="write",
        )
        return cast(State, updated)

    return _node


def _load_session_read_results(state: State) -> list[JsonObject]:
    """读取当前会话所有轮次保存的论文阅读笔记。

    中文说明：会话产物记录里既有阅读汇总，也有每篇论文自己的 note.json。
    这里只读取每篇论文的 note.json，因为它同时保存了 paperId、论文信息和阅读笔记，
    内容完整且不会把同一轮汇总数据重复传给写作 Agent。
    """

    repository = state.get("session_repo")
    session_key = str(state.get("session_key") or "").strip()
    if repository is None or not session_key:
        return []

    try:
        session = repository.get(session_key)
    except Exception:
        # 会话刚被删除或产物暂时不可读时，继续使用 State 和 paper_cache。
        return []

    results: list[JsonObject] = []
    for artifact in list(getattr(session, "artifacts", []) or []):
        if not isinstance(artifact, dict) or artifact.get("artifact_type") != "paper_read_note":
            continue
        path = Path(str(artifact.get("path") or ""))
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            results.append(payload)
    return results


def _flatten_outline(outline: JsonObject) -> list[JsonObject]:
    """把章节大纲拍平成按顺序执行的小节任务列表。"""

    tasks: list[JsonObject] = []
    for chapter_key, chapter in outline.items():
        if not isinstance(chapter, dict):
            continue
        sections = chapter.get("Sections")
        if not isinstance(sections, dict):
            continue
        for section_key, section in sections.items():
            if not isinstance(section, dict):
                continue
            section_id = f"{chapter_key}.{section_key}"
            tasks.append(
                {
                    "section_id": section_id,
                    "chapter_key": chapter_key,
                    "section_key": section_key,
                    "chapter_title": str(chapter.get("title") or chapter_key),
                    "section_title": str(section.get("title") or section_key),
                    "chapter_description": str(chapter.get("description") or ""),
                    "task": str(section.get("task") or ""),
                    "evidence_map": list(section.get("evidence-map") or []),
                    "ref_sections": list(section.get("ref-sections") or []),
                    "word_count": int(section.get("word-count") or 800),
                }
            )
    return tasks


def _resolve_previous_sections(*, requested_refs: list[Any], written_sections: list[JsonObject]) -> list[JsonObject]:
    """根据大纲里的 ref-sections 找出已写小节，但不转发未经审计的正文。

    中文注释：
    ref-sections 可能写成 Chapter1.section2，也可能只写 Chapter1。
    这里做最简单的匹配：完整小节编号精确匹配，章节编号匹配该章节下所有已写小节。
    旧正文可能含有未核实的判断，只保留编号帮助作者知道哪些小节已经完成。
    """

    if not requested_refs:
        return []
    resolved: list[JsonObject] = []
    seen: set[str] = set()
    ref_texts = [str(ref or "").strip() for ref in requested_refs if str(ref or "").strip()]
    for section in written_sections:
        section_id = str(section.get("section_id") or "").strip()
        chapter_key = str(section.get("chapter_key") or "").strip()
        if not section_id:
            continue
        matched = section_id in ref_texts or chapter_key in ref_texts
        if matched and section_id not in seen:
            resolved.append(
                {
                    "section_id": section_id,
                    "content": "",
                }
            )
            seen.add(section_id)
    return resolved


def _build_writing_report(
    *,
    topic: str,
    outline: JsonObject,
    sections: list[JsonObject],
    abstract: str,
    abstract_status: str,
    references: list[JsonObject],
    model_used: str,
) -> JsonObject:
    """整理正文、摘要和参考文献组成的完整写作产物。"""

    citation_index_by_paper_id = {
        str(item.get("paperId") or "").strip().lower(): str(item.get("index"))
        for item in references
        if str(item.get("paperId") or "").strip() and item.get("index") is not None
    }
    # 中文说明：参考文献编号只有在所有论文都整理完之后才确定，
    # 因此正文先保留 paperId，最后在这里一次性替换成 [1]、[2] 这样的序号。
    numbered_sections = _replace_section_citations(sections, citation_index_by_paper_id)
    numbered_abstract = _replace_citation_numbers(abstract, citation_index_by_paper_id)

    return {
        "writing_version": WRITING_VERSION,
        "topic": topic,
        "writing_outline": outline,
        "sections": numbered_sections,
        "abstract": numbered_abstract,
        "references": references,
        "references_markdown": _references_markdown(references),
        "content_markdown": _compose_content_markdown(numbered_abstract, numbered_sections, references),
        "cited_paper_ids": [str(item.get("paperId") or "") for item in references if str(item.get("paperId") or "").strip()],
        "execution_metadata": {
            "model_used": model_used,
            "section_count": len(sections),
            "abstract_status": abstract_status,
            "reference_count": len(references),
            "created_at": utc_now(),
        },
    }


def _replace_section_citations(
    sections: list[JsonObject],
    citation_index_by_paper_id: dict[str, str],
) -> list[JsonObject]:
    """把正文小节中的 `[paperId]` 替换为参考文献序号。"""

    if not citation_index_by_paper_id:
        return [dict(section) for section in sections]

    replaced: list[JsonObject] = []
    for section in sections:
        item = dict(section)
        item["content"] = _replace_citation_numbers(str(item.get("content") or ""), citation_index_by_paper_id)
        replaced.append(item)
    return replaced


def _replace_citation_numbers(content: str, citation_index_by_paper_id: dict[str, str]) -> str:
    """替换一段文本中的论文编号，普通 Markdown 方括号保持不变。"""

    citation_pattern = re.compile(r"\[([^\[\]\r\n]+)\]")

    def replace(match: re.Match[str]) -> str:
        candidate = match.group(1).strip().strip('"').strip("'").strip()
        index = citation_index_by_paper_id.get(candidate.lower())
        return f"[{index}]" if index else match.group(0)

    return citation_pattern.sub(replace, content)


def _extract_paper_ids_from_sections(sections: list[JsonObject]) -> list[str]:
    """从小节正文的方括号引用中提取 paperId，并按首次出现顺序去重。"""

    # 中文说明：正文约定使用 [paperId] 标记引用。先收集 Agent 返回的 paperId，
    # 用它过滤普通的 Markdown 方括号文字；随后再保留带数字、冒号或斜杠的未知编号。
    declared_ids = _collect_cited_paper_ids(sections)
    declared_by_key = {paper_id.lower(): paper_id for paper_id in declared_ids}
    found: list[str] = []
    seen: set[str] = set()
    citation_pattern = re.compile(r"\[([^\[\]\r\n]+)\]")
    for section in sections:
        content = str(section.get("content") or "")
        for match in citation_pattern.finditer(content):
            candidate = match.group(1).strip().strip('"').strip("'")
            if not candidate or any(character.isspace() for character in candidate):
                continue
            # 中文说明：即使模型漏掉了前面的归一化，也不能把切片编号直接生成参考文献。
            if _is_chunk_id(candidate):
                continue
            paper_id = declared_by_key.get(candidate.lower(), candidate)
            if candidate.lower() not in declared_by_key and not re.search(r"\d|[:/.]", candidate):
                continue
            key = paper_id.lower()
            if key in seen:
                continue
            seen.add(key)
            found.append(paper_id)

        # 中文说明：如果模型把引用列在结构化字段里但正文没有重复写出，仍保留该引用，
        # 避免正文内容和写作 Agent 的引用记录不一致。
        for paper_id in list(section.get("cited_paper_ids") or []):
            text = str(paper_id or "").strip()
            if _is_chunk_id(text):
                continue
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                found.append(text)
    return found


def _is_chunk_id(value: str) -> bool:
    """判断一个候选编号是否符合全文切片的页码或分段编号格式。"""

    return bool(re.search(r":(?:p|c)\d{4}(?::s\d{4})?$", str(value or "").strip(), flags=re.IGNORECASE))


def _collect_paper_metadata(
    paper_ids: list[str],
    *,
    search_results: list[Any],
    read_results: list[JsonObject],
    session_read_results: list[JsonObject],
    cache_dir: Path,
) -> dict[str, JsonObject]:
    """从当前状态、会话阅读产物和本地缓存汇总论文元数据。"""

    metadata_by_id: dict[str, JsonObject] = {}

    def add_payload(payload: Any) -> None:
        if isinstance(payload, PaperDocument):
            paper = payload.to_dict()
        elif isinstance(payload, dict):
            paper = dict(payload.get("paper") or payload)
        else:
            return
        paper_id = str(paper.get("paperId") or paper.get("id") or "").strip()
        if not paper_id:
            return
        metadata_by_id.setdefault(paper_id.lower(), paper)

    for paper in search_results:
        add_payload(paper)
    for result in [*read_results, *session_read_results]:
        add_payload(result.get("paper") if isinstance(result, dict) else None)

    for paper_id in paper_ids:
        key = paper_id.lower()
        if key in metadata_by_id:
            continue
        for directory in _paper_metadata_dirs(cache_dir, paper_id):
            metadata_path = directory / "metadata.json"
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            paper = dict(payload.get("paper") or payload) if isinstance(payload, dict) else {}
            cached_ids = {
                str(payload.get("paperId") or "").strip().lower() if isinstance(payload, dict) else "",
                str(paper.get("paperId") or "").strip().lower(),
                str(paper.get("id") or "").strip().lower(),
            }
            if key not in cached_ids:
                continue
            add_payload(payload)
            if key in metadata_by_id:
                break
    return metadata_by_id


def _collect_available_paper_ids(
    *,
    read_results: list[JsonObject],
    session_read_results: list[JsonObject],
) -> list[str]:
    """只收集本会话已经建立全文切片的论文编号。"""

    # 中文说明：检索到题录不等于读到了全文。旧逻辑把下载失败的论文也列进
    # “允许引用”的名单，模型写完后必然找不到原文切片。这里只从阅读记录取
    # 已索引且确有切片的论文；失败题录仍留在检索记录，不冒充全文依据。
    paper_ids: list[str] = []
    for payload in [*read_results, *session_read_results]:
        if not isinstance(payload, dict):
            continue
        full_text = payload.get("full_text") or {}
        if full_text.get("status") not in {"indexed", "chunks_saved"} or int(full_text.get("chunk_count") or 0) < 1:
            continue
        paper = dict(payload.get("paper") or {})
        paper_id = str(paper.get("paperId") or paper.get("id") or "").strip()
        if paper_id:
            paper_ids.append(paper_id)
    seen: set[str] = set()
    unique_paper_ids: list[str] = []
    for paper_id in paper_ids:
        key = paper_id.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_paper_ids.append(paper_id)
    return unique_paper_ids


def _has_reference_metadata(paper: JsonObject) -> bool:
    """判断论文资料是否至少包含可展示的题名。"""

    # 中文说明：只有编号而没有题名时，旧逻辑会把编号本身当成题名，
    # 生成 P1[J] 这样的错误条目。因此题名是生成参考文献的最低条件。
    return bool(str(paper.get("title") or "").strip())


def _remove_unknown_paper_citations(
    sections: list[JsonObject],
    unknown_paper_ids: list[str],
) -> list[JsonObject]:
    """删除小节中没有真实论文资料支撑的引用标记。"""

    if not unknown_paper_ids:
        return [dict(section) for section in sections]

    unknown_keys = {paper_id.lower() for paper_id in unknown_paper_ids}
    cleaned_sections: list[JsonObject] = []
    for section in sections:
        cleaned = dict(section)
        cleaned["content"] = _remove_unknown_citation_markers(
            str(cleaned.get("content") or ""),
            unknown_paper_ids,
        )
        # 中文说明：正文和 cited_paper_ids 必须同步清理。
        # 只清正文会让参考文献仍然收集到错误编号；只清列表又会留下错误标记。
        cleaned["cited_paper_ids"] = [
            paper_id
            for paper_id in list(cleaned.get("cited_paper_ids") or [])
            if str(paper_id or "").strip().lower() not in unknown_keys
        ]
        cleaned_sections.append(cleaned)
    return cleaned_sections


def _remove_unknown_citation_markers(content: str, unknown_paper_ids: list[str]) -> str:
    """从一段文字中删除形如 [P1] 的未知引用标记。"""

    if not content or not unknown_paper_ids:
        return content

    unknown_keys = {paper_id.lower() for paper_id in unknown_paper_ids}
    citation_pattern = re.compile(r"\[([^\[\]\r\n]+)\]")

    def replace(match: re.Match[str]) -> str:
        paper_id = match.group(1).strip().strip('"').strip("'")
        return "" if paper_id.lower() in unknown_keys else match.group(0)

    return citation_pattern.sub(replace, content)


def _paper_metadata_dirs(cache_dir: Path, paper_id: str) -> list[Path]:
    """定位某个 paperId 可能对应的缓存目录。"""

    if not cache_dir.is_dir():
        return []
    direct = cache_dir / safe_cache_name(paper_id)
    directories = [direct] if direct.is_dir() else []
    for candidate in cache_dir.iterdir():
        if candidate.is_dir() and candidate not in directories:
            directories.append(candidate)
    return directories


def _build_references(paper_ids: list[str], metadata_by_id: dict[str, JsonObject]) -> list[JsonObject]:
    """按正文首次引用顺序生成 GB/T 7714 参考文献条目。"""

    references: list[JsonObject] = []
    for paper_id in paper_ids:
        metadata = dict(metadata_by_id.get(paper_id.lower()) or {})
        # 中文说明：这里再检查一次，防止以后其他调用方漏掉前面的清理步骤。
        # 没有题名的资料不生成参考文献，不能用 paperId 代替论文题名。
        if not _has_reference_metadata(metadata):
            continue
        references.append(
            {
                "index": len(references) + 1,
                "paperId": paper_id,
                "citation": _format_gbt7714_reference(paper_id, metadata),
                "metadata": metadata,
            }
        )
    return references


def _format_gbt7714_reference(paper_id: str, paper: JsonObject) -> str:
    """使用论文元数据生成常见的 GB/T 7714 顺序编码制格式。"""

    title = str(paper.get("title") or paper_id).strip()
    extra_metadata = paper.get("metadata") if isinstance(paper.get("metadata"), dict) else {}
    authors = _format_reference_authors(paper.get("authors") or paper.get("author"))
    resource_type = _reference_resource_type(paper)
    container = str(
        paper.get("journal_conference")
        or paper.get("journal/conference")
        or paper.get("journal")
        or paper.get("venue")
        or extra_metadata.get("journal")
        or ""
    ).strip()
    year = str(paper.get("year") or paper.get("publication_date") or "").strip()[:4]
    volume = str(paper.get("volume") or "").strip()
    issue = str(paper.get("issue") or "").strip()
    pages = str(
        paper.get("pages")
        or paper.get("page_range")
        or extra_metadata.get("pages")
        or (
            f"{paper.get('page_start')}-{paper.get('page_end')}"
            if paper.get("page_start") is not None and paper.get("page_end") is not None
            else ""
        )
    ).strip()
    doi = str(paper.get("doi") or "").strip()
    url = str(paper.get("url") or "").strip()

    citation = f"{authors + '. ' if authors else ''}{title}[{resource_type}]"
    if container:
        citation += f". {container}"
    if year:
        citation += f", {year}"
    if volume:
        citation += f", {volume}"
        if issue:
            citation += f"({issue})"
    elif issue:
        citation += f", ({issue})"
    if pages:
        citation += f": {pages}"
    citation += "."
    if doi:
        citation += f" DOI: {doi}."
    elif url:
        citation += f" {url}."
    return citation


def _format_reference_authors(value: Any) -> str:
    """整理作者字段，超过三位时按 GB/T 7714 习惯使用 et al.。"""

    if isinstance(value, str):
        authors = [item.strip() for item in re.split(r"[,;，；]", value) if item.strip()]
    elif isinstance(value, list):
        authors = []
        for item in value:
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("author") or "").strip()
            else:
                name = str(item or "").strip()
            if name:
                authors.append(name)
    else:
        authors = []
    if len(authors) > 3:
        suffix = "等" if any("\u4e00" <= character <= "\u9fff" for character in authors[0]) else "et al"
        return ", ".join(authors[:3]) + ("，" if suffix == "等" else ", ") + suffix
    return ", ".join(authors)


def _reference_resource_type(paper: JsonObject) -> str:
    """根据元数据推断参考文献类型，缺少信息时按期刊论文处理。"""

    type_text = " ".join(
        str(paper.get(key) or "")
        for key in ("type", "document_type", "publication_type", "source")
    ).lower()
    if "conference" in type_text or "proceedings" in type_text:
        return "C"
    if "thesis" in type_text or "dissertation" in type_text:
        return "D"
    if "book" in type_text:
        return "M"
    if "arxiv" in type_text or "preprint" in type_text:
        return "EB/OL"
    return "J"


def _compose_content_markdown(
    abstract: str,
    sections: list[JsonObject],
    references: list[JsonObject],
) -> str:
    """按摘要、正文、参考文献顺序拼接最终 Markdown。"""

    blocks: list[str] = []
    if abstract.strip():
        blocks.append(f"# 摘要\n\n{abstract.strip()}")
    body = _sections_to_markdown(sections)
    if body:
        blocks.append(body)
    if references:
        reference_text = _references_markdown(references)
        if reference_text:
            blocks.append(f"# 参考文献\n\n{reference_text}")
    return "\n\n".join(blocks).strip()


def _references_markdown(references: list[JsonObject]) -> str:
    """把结构化参考文献条目拼成带编号的 Markdown 文本。"""

    return "\n".join(
        f"[{item.get('index')}] {item.get('citation')}"
        for item in references
        if str(item.get("citation") or "").strip()
    )


def _sections_to_markdown(sections: list[JsonObject]) -> str:
    """把所有小节正文拼成一份 Markdown，方便用户直接预览。"""

    blocks: list[str] = []
    current_chapter = ""
    for section in sections:
        chapter_key = str(section.get("chapter_key") or "")
        if chapter_key and chapter_key != current_chapter:
            blocks.append(f"# {section.get('chapter_title') or chapter_key}")
            current_chapter = chapter_key
        section_id = str(section.get("section_id") or "")
        section_title = str(section.get("section_title") or section_id).strip()
        content = str(section.get("content") or "").strip()
        blocks.append(f"## {section_title}\n\n{content}")
    return "\n\n".join(blocks).strip()


def _collect_cited_paper_ids(sections: list[JsonObject]) -> list[str]:
    """汇总所有小节实际引用到的 paperId。"""

    seen: set[str] = set()
    result: list[str] = []
    for section in sections:
        for paper_id in list(section.get("cited_paper_ids") or []):
            text = str(paper_id or "").strip()
            if _is_chunk_id(text):
                continue
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            result.append(text)
    return result


def _resolve_llm(state: State) -> ProviderSnapshot | None:
    """优先使用外部注入的正文写作模型，没有注入时读取默认配置。"""

    # 显式 None 表示关闭；只有省略字段或使用 auto 才读取本地配置。
    injected = state.get("writing_node_llm", "auto")
    if isinstance(injected, ProviderSnapshot):
        return injected
    if injected == "auto":
        snapshot = load_writing_agent_llm()
        # 中文注释：本节点自己创建的模型连接，在任务结束时连同备用模型一起关闭。
        resources = getattr(state.get("runtime_context"), "resources", None)
        if isinstance(resources, WorkflowRuntimeResources):
            resources.track_model_snapshot(snapshot)
        return snapshot
    return None


def _resolve_reporter(state: State):
    """从运行上下文里取出正文写作节点的进度上报器。"""

    runtime = cast(WorkflowRuntimeContext | None, state.get("runtime_context"))
    if runtime is None or runtime.sync_port is None:
        return None
    revision = int(state.get("audit_revision") or 0)
    return runtime.sync_port.for_node(f"write_revision_{revision}" if revision else "write", "正文修订" if revision else "正文写作")


async def _persist_writing_if_possible(state: State, report: JsonObject) -> JsonObject | None:
    """如果当前有会话仓库，就把正文写作结果保存成 JSON 产物。"""

    repo = cast(SessionRepository | None, state.get("session_repo"))
    session_key = _optional_text(state.get("session_key"))
    turn_id = _optional_text(state.get("turn_id"))
    if repo is None or not session_key or not turn_id:
        return None
    try:
        previous = (next((item for item in reversed(repo.get(session_key).artifacts)
                          if item.get("artifact_type") == "writing"), None)
                    if state.get("writing_target_ids") or state.get("audit_revision") else None)
        if state.get("writing_target_ids"):
            # 中文说明：每次人工纠错生成新文件，并在文件内部留下来源版本、
            # 修改范围和用户原话，方便之后比较两版报告。
            report.setdefault("execution_metadata", {})["user_revision"] = {
                "previous_version_artifact_id": (previous or {}).get("id"),
                "modified_section_ids": list(state.get("writing_target_ids") or []),
                "instruction": str(state.get("writing_instruction") or "")[:2000],
            }
        record = await asyncio.to_thread(
            repo.write_artifact,
            session_key,
            "writing",
            "writing.json",
            json.dumps(report, ensure_ascii=False, indent=2),
            relative_path=f"artifacts/writing/{turn_id}/revision_{int(state.get('audit_revision') or 0)}/writing.json",
            metadata={"turn_id": turn_id, "format": "json", "writing_version": WRITING_VERSION,
                      "audit_revision": int(state.get("audit_revision") or 0),
                      "previous_version_artifact_id": (previous or {}).get("id"),
                      "modified_section_ids": list(state.get("writing_target_ids") or []),
                      "user_instruction": str(state.get("writing_instruction") or "")[:2000]},
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
