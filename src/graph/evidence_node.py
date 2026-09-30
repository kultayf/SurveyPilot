"""实证矩阵、有限补搜和独立核查的流程衔接。"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import re
from pathlib import Path

from src.agents.base import AgentContext
from src.agents.analyseAgent import load_analyse_agent_llm
from src.agents.evidenceMatrixAgent import DIMENSIONS, EvidenceMatrixAgent
from src.agents.citationAuditAgent import CitationAuditAgent
from src.graph.state_models import State
from src.graph.writing_node import _load_session_read_results
from src.llm import ProviderSnapshot, SystemConfig
from src.paper_retrieval import PaperSearchService
from src.retrieval.hybrid import load_scoped_chunks, paper_scope
from src.utils.read_utils.chunkers import chunks_content_hash


def _reporter(state: State, name: str, title: str):
    runtime = state.get("runtime_context")
    revision = int(state.get("audit_revision") or 0) if name == "citation_audit" else int(state.get("research_round") or 0)
    # 每轮使用独立卡片，补搜与修订的实际用量不会覆盖第一轮的数字。
    key = f"{name}_{revision}" if revision else name
    return runtime.sync_port.for_node(key, title) if runtime and runtime.sync_port else None


def _check_cancel(state: State):
    """每处理一篇或一节前响应用户停止，不必等整个批次结束。"""
    cancellation = getattr(state.get("runtime_context"), "cancellation", None)
    if cancellation:
        cancellation.raise_if_requested()


def _usage_callback(reporter, stage: str):
    """累计供应商返回的真实用量，不能只显示最后一次调用的数值。"""
    totals = {"input_tokens": 0, "output_tokens": 0}
    def report(value):
        for key in totals:
            totals[key] += int(value.get(key) or 0)
        if reporter:
            reporter.progress("模型调用完成", stage=stage, **totals)
    return report


def _model(state: State, key: str, profile: str):
    """显式注入 None 表示关闭模型；本节点创建的连接在 finally 中关闭。"""
    if key in state and state[key] != "auto":
        value = state[key]
        return (value if isinstance(value, ProviderSnapshot) else None), False
    return load_analyse_agent_llm(agent_name=profile), True


async def _save(state: State, kind: str, name: str, content: str, reporter, *, revision: int = 0):
    """每轮报告单独保存，保留修订前后的检查记录；失败明确抛出而不伪装保存成功。"""
    repo, key, turn = state.get("session_repo"), state.get("session_key"), state.get("turn_id")
    if repo is None or not key or not turn:
        return None
    record = await asyncio.to_thread(repo.write_artifact, key, kind, name, content,
        relative_path=f"artifacts/{kind}/{turn}/{revision}/{name}",
        metadata={"turn_id": turn, "revision": revision, "schema_version": 1})
    ref = {**record, "artifact_id": record["id"]}
    if reporter:
        reporter.artifact(ref, stage="evidence_artifact_ready")
    return ref


def _matrix_exports(matrix: dict) -> tuple[str, str]:
    """CSV 防止原文中的等号被表格软件执行；Markdown 保留逐格来源脚注。"""
    columns = ["论文", "年份", *DIMENSIONS.values()]
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(columns)
    markdown = ["# 实证对比矩阵", "", "原文已定位不等于独立核查通过；未找到仅指本次选取的原文范围。", "",
                "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    footnotes = []
    def csv_safe(value):
        text = str(value or "")
        return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text
    def md_safe(value):
        return str(value or "").replace("|", "\\|").replace("\n", "<br>")
    for row_index, row in enumerate(matrix["rows"], start=1):
        values = [row["title"], row.get("year") or ""]
        markdown_values = [md_safe(value) for value in values]
        for dimension, cell in row["cells"].items():
            value = cell["value"] or ("未验证" if row["status"] == "unverified" else "未找到")
            values.append(value)
            label = f"r{row_index}-{dimension}"
            markdown_values.append(md_safe(value) + (f"[^{label}]" if cell["evidence"] else ""))
            for evidence in cell["evidence"]:
                footnotes.append(f"[^{label}]: {md_safe(evidence['paperId'])} / {md_safe(evidence['chunkId'])}，"
                                 f"页码 {evidence.get('page_start') or '?'}–{evidence.get('page_end') or '?'}")
        writer.writerow([csv_safe(value) for value in values])
        markdown.append("| " + " | ".join(markdown_values) + " |")
    return "\ufeff" + output.getvalue(), "\n".join(markdown + ["", *footnotes]) + "\n"


async def run_matrix_node(state: State) -> State:
    """按论文提取 12 个科学维度，补搜返回时复用原文未变的矩阵行。"""
    reporter = _reporter(state, "matrix", "实证矩阵")
    if reporter:
        reporter.started("正在逐格定位论文实证信息", stage="matrix_start")
    llm, owned = _model(state, "matrix_node_llm", "default_agent")
    usage = _usage_callback(reporter, "matrix_extract")
    agent = EvidenceMatrixAgent(AgentContext(llm=llm, usage_callback=usage))
    all_reads = list(state.get("read_results") or [])
    # 中文说明：矩阵需要全文原句。只处理本轮真正被选中精读的论文；
    # 摘要筛选未入选的论文即使碰巧有旧缓存，也不能绕过深读数量限制再次调用模型。
    selected_reads = [item for item in all_reads if (item.get("relevance") or {}).get("status") == "selected_for_deep_read"]
    if not selected_reads and not any((item.get("relevance") or {}).get("status") for item in all_reads):
        selected_reads = all_reads
    # 中文说明：没有拿到全文的论文仍保留摘要笔记，但不能把它的 12 个全文字段
    # 全部算作“未找到”。否则下载站点拒绝访问也会拉低事实矩阵覆盖率并误触发补搜。
    indexed_reads = [item for item in selected_reads if (item.get("full_text") or {}).get("status") in {"indexed", "chunks_saved"}]
    chunks = await asyncio.to_thread(load_scoped_chunks, Path(SystemConfig.load().read.paper_cache_dir), indexed_reads)
    reads = []
    skipped_abstract_only = []
    for item in selected_reads:
        paper = item.get("paper") or {}
        paper_id = str(paper.get("paperId") or paper.get("id") or "")
        indexed = (item.get("full_text") or {}).get("status") in {"indexed", "chunks_saved"}
        allowed = {value.casefold() for value in paper_scope([item])} if indexed else set()
        has_chunks = indexed and any(chunk.paperId.casefold() in allowed for chunk in chunks)
        if has_chunks:
            reads.append(item)
        else:
            # 中文说明：状态写着已索引，但切片文件丢失时，同样不能编造全文矩阵行。
            reason = (item.get("full_text") or {}).get("reason") or (
                "全文切片丢失或不可读取" if indexed else (item.get("full_text") or {}).get("status") or "全文未建立索引"
            )
            skipped_abstract_only.append({"paperId": paper_id, "title": paper.get("title") or "", "reason": reason})
    old_rows = {row["paperId"]: row for row in (state.get("evidence_matrix") or {}).get("rows", [])}
    rows = []
    try:
        for result in reads:
            _check_cancel(state)
            paper = result.get("paper") or {}
            paper_id = str(paper.get("paperId") or paper.get("id") or "")
            allowed = {value.casefold() for value in paper_scope([result])}
            paper_chunks = [chunk for chunk in chunks if chunk.paperId.casefold() in allowed]
            source_hash = chunks_content_hash(paper_chunks)
            previous = old_rows.get(paper_id)
            if previous and previous.get("source_hash") == source_hash and previous.get("status") == "extracted":
                row = previous
            else:
                row = await agent.extract(paper, paper_chunks)
                row["source_hash"] = source_hash
            rows.append(row)
            if reporter:
                reporter.progress(f"已整理 {len(rows)}/{len(reads)} 篇论文", stage="matrix_extract", completed=len(rows), total=len(reads))
    finally:
        if owned and llm:
            await llm.aclose()
    located = sum(cell["status"] == "source_located" for row in rows for cell in row["cells"].values())
    matrix = {"schema_version": 1, "dimensions": DIMENSIONS, "rows": rows,
              "skipped_abstract_only_papers": skipped_abstract_only,
              "located_cells": located, "total_cells": len(rows) * len(DIMENSIONS),
              "coverage": located / (len(rows) * len(DIMENSIONS)) if rows else 0,
              "round": int(state.get("research_round") or 0)}
    csv_content, markdown = _matrix_exports(matrix)
    refs = list(state.get("research_artifact_refs") or [])
    for name, content in (("evidence_matrix.json", json.dumps(matrix, ensure_ascii=False, indent=2)),
                          ("evidence_matrix.csv", csv_content), ("evidence_matrix.md", markdown)):
        ref = await _save(state, "evidence_matrix", name, content, reporter, revision=matrix["round"])
        if ref:
            refs.append(ref)
    if reporter:
        skipped_notice = f"；另有 {len(skipped_abstract_only)} 篇无可用全文，未计入矩阵" if skipped_abstract_only else ""
        reporter.completed(f"已定位 {located}/{matrix['total_cells']} 个字段，缺失项保持空白{skipped_notice}", stage="matrix_done")
    return {"evidence_matrix": matrix, "research_artifact_refs": refs, "current_step": "matrix"}


def route_after_matrix(state: State) -> str:
    """只有提取成功而证据确实不足才补搜；模型失效不靠扩大检索掩盖。"""
    matrix = state.get("evidence_matrix") or {}
    rows = matrix.get("rows") or []
    enabled = SystemConfig.load().research.supplemental_search_enabled
    constraints = state["request"].constraints
    max_results = constraints.get("max_results")
    if max_results is not None:
        try:
            enabled = enabled and len(state.get("search_results") or []) < int(max_results)
        except (ValueError, TypeError):
            pass
    limit = constraints.get("deep_read_limit", constraints.get("max_deep_read"))
    selected = sum((row.get("relevance") or {}).get("status") == "selected_for_deep_read" for row in state.get("read_results") or [])
    if limit is not None:
        try:
            enabled = enabled and selected < int(limit)
        except (ValueError, TypeError):
            pass
    return "supplement" if (enabled and int(state.get("research_round") or 0) < 1 and rows
        and any(row.get("status") == "extracted" for row in rows) and matrix.get("coverage", 0) < 0.5) else "analyse"


def _paper_keys(paper) -> set[str]:
    """用 DOI、来源编号和规范化题名去重，避免多来源返回同一篇论文。"""
    keys = {str(value).strip().casefold().removeprefix("https://doi.org/").removeprefix("doi:")
            for value in (paper.id, paper.paperId, paper.doi) if value}
    title = re.sub(r"\W+", "", paper.title).casefold()
    if title:
        keys.add("title:" + title)
    return keys


async def run_supplement_node(state: State) -> State:
    """仅一轮、最多两条查询、最多三篇新论文；不改写用户原始主题或筛选条件。"""
    if int(state.get("research_round") or 0) >= 1:
        return {}
    reporter = _reporter(state, "supplement", "补充检索")
    if reporter:
        reporter.started("核心实证字段不足，尝试一次补充检索", stage="supplement_start")
    request = state["request"]
    matrix = state.get("evidence_matrix") or {}
    rows = matrix.get("rows") or []
    missing = sorted(DIMENSIONS, key=lambda key: sum(row["cells"][key]["status"] == "source_located" for row in rows))
    queries = [f"{request.topic} {DIMENSIONS[key].split(' / ')[-1]}" for key in missing[:2]]
    service = state.get("search_node_service") or PaperSearchService()
    original = list(state.get("search_results") or [])
    seen = set().union(*(_paper_keys(paper) for paper in original)) if original else set()
    added, errors = [], []
    constraints = request.constraints
    budget = 3
    try:
        if constraints.get("max_results") is not None:
            budget = min(3, max(0, int(constraints["max_results"]) - len(original)))
    except (ValueError, TypeError):
        pass
    intent = state.get("search_summary") or {}
    # 搜索节点保存的 search_summary.sources 是实际采用的来源；恢复后也继续遵守它。
    sources = constraints.get("sources") or (state.get("search_summary") or {}).get("sources")
    if isinstance(sources, str):
        sources = [sources]
    for query in queries:
        _check_cancel(state)
        if len(added) >= budget:
            break
        try:
            response = await asyncio.wait_for(service.async_search(query=query, limit=6, sources=sources,
                year_from=constraints.get("year_from", intent.get("year_from")),
                year_to=constraints.get("year_to", intent.get("year_to")),
                excluded_terms=constraints.get("excluded_terms", intent.get("excluded_terms", [])),
                runtime_resources=getattr(state.get("runtime_context"), "resources", None)), timeout=60)
            errors.extend(str(value) for value in response.errors.values())
            for paper in response.papers:
                # 与初次检索相同，缺少摘要或可靠编号的记录不交给阅读模型。
                if not (paper.abstract or "").strip() or not (paper.paperId or "").strip():
                    continue
                keys = _paper_keys(paper)
                if seen.intersection(keys):
                    continue
                seen.update(keys)
                added.append(paper)
                if len(added) >= budget:
                    break
        except Exception as exc:
            errors.append(type(exc).__name__)
    report = {"round": 1, "queries": queries, "added_paper_ids": [paper.id for paper in added],
              "errors": errors, "stop_reason": "补检索次数上限为 1，新增论文上限为 3"}
    refs = list(state.get("research_artifact_refs") or [])
    ref = await _save(state, "supplemental_search", "supplemental_search.json", json.dumps(report, ensure_ascii=False, indent=2), reporter)
    if ref:
        refs.append(ref)
    if reporter:
        reporter.completed(f"补充检索新增 {len(added)} 篇论文", stage="supplement_done")
    return {"search_results": original + added, "research_round": 1,
            "supplemental_paper_ids": report["added_paper_ids"], "supplemental_search": report,
            "research_artifact_refs": refs, "current_step": "supplement"}


def route_after_supplement(state: State) -> str:
    return "read" if state.get("supplemental_paper_ids") else "analyse"


async def run_audit_node(state: State) -> State:
    """正文与摘要都纳入核查；达到修订上限后仍失败，只生成明确标识的待核查草稿。"""
    reporter = _reporter(state, "citation_audit", "独立引用核查")
    if reporter:
        reporter.started("正在核对正文、摘要和原文证据", stage="audit_start")
    llm, owned = _model(state, "audit_node_llm", "solar_agent")
    usage = _usage_callback(reporter, "audit_check")
    agent = CitationAuditAgent(AgentContext(llm=llm, usage_callback=usage))
    reads = list(state.get("read_results") or []) + await asyncio.to_thread(_load_session_read_results, state)
    chunks = await asyncio.to_thread(load_scoped_chunks, Path(SystemConfig.load().read.paper_cache_dir), reads)
    aliases = {}
    source_hints = {}
    for result in reads:
        paper = result.get("paper") or {}
        canonical = str(paper.get("paperId") or paper.get("id") or "").casefold()
        for alias in paper_scope([result]):
            aliases[alias.casefold()] = canonical
        # 中文说明：这里只取阅读阶段的英文方法说明作检索词，帮助独立审计
        # 在同篇原文中优先找到机制定义。审计模型实际看到的仍是原文切片；
        # 即便这份说明写错，也不能靠它直接判定正文得到支持。
        extraction = result.get("extraction") or {}
        if canonical and isinstance(extraction, dict) and str(extraction.get("methods") or "").strip():
            source_hints[canonical] = str(extraction["methods"])
    writing = dict(state.get("writing_report") or {})
    sections = list(writing.get("sections") or state.get("writing_sections") or [])
    abstract_paper_terms = {}
    # 中文说明：第 24 轮 APPNP 小节标题以中文“基于个性化…”开头，
    # 旧规则未建立 APPNP→论文映射；摘要只点名 APPNP 时仍先塞五篇
    # 各一段，幂迭代方法原文被挤掉。用户题目若明确写出
    # “方法名（arXiv:编号）”，只用这组已在参考文献中的对应关系
    # 缩小候选范围，真正的事实判断仍由独立模型读原文完成。
    referenced_ids = {str(ref.get("paperId") or "").casefold() for ref in writing.get("references") or []}
    topic = str(getattr(state.get("request"), "topic", "") or "")
    for match in re.finditer(r"([A-Za-z][A-Za-z0-9-]{2,})\s*[（(]\s*arXiv\s*:\s*(\d{4}\.\d{4,5})\s*[）)]",
                             topic, re.IGNORECASE):
        paper_id = aliases.get(match.group(2).casefold(), "")
        if paper_id and paper_id in referenced_ids:
            abstract_paper_terms[match.group(1).casefold()] = paper_id
    for section in sections:
        # 中文说明：摘要没有正式引文，但单论文小节标题常写明 GCN、GAT 等
        # 方法名。只在该小节确实引用一篇论文时建立“方法名→论文”对应；
        # 摘要句若明确点名这些方法，可少给无关论文的切片，不改变事实判断。
        cited_ids = list(section.get("cited_paper_ids") or [])
        title_match = re.match(r"\s*([A-Za-z][A-Za-z0-9-]{2,})", str(section.get("section_title") or ""))
        if len(cited_ids) == 1 and title_match:
            abstract_paper_terms.setdefault(title_match.group(1).casefold(), str(cited_ids[0]))
    reports = []
    generation_errors = []
    review_errors = []
    try:
        for section in sections:
            _check_cancel(state)
            section_id = str(section.get("section_id") or "")
            content = str(section.get("content") or "").strip()
            if not content or content == "本节正文未能生成，需重新生成并核查。":
                # 中文说明：只跳过真正缺正文的小节。旧数据里已有正文却误带
                # generation_failed 标记，仍须逐句审计，不能丢掉真实正文。
                generation_errors.append(section_id)
                reports.append({"section_id": section.get("section_id"), "status": "needs_review",
                                "reason": "小节缺少正文，未进行事实核查", "units": []})
                continue
            if (section.get("review") or {}).get("passed") is False:
                review_errors.append(section_id)
            reports.append(await agent.audit(section, chunks, aliases, writing.get("references") or [],
                                             source_hints=source_hints))
        # 摘要不能因为没有正式引用标记而跳过事实核查。
        _check_cancel(state)
        reports.append(await agent.audit({"section_id": "abstract", "content": writing.get("abstract") or ""},
                                        chunks, aliases, writing.get("references") or [], abstract=True,
                                        source_hints=source_hints, abstract_paper_terms=abstract_paper_terms))
    finally:
        if owned and llm:
            await llm.aclose()
    supported_count = sum(unit["status"] == "supported" for report in reports for unit in report["units"])
    if (writing.get("execution_metadata") or {}).get("abstract_status") not in {None, "ok"}:
        generation_errors.append("abstract")
    passed = (not generation_errors and not review_errors and bool(sections)
              and supported_count > 0 and all(report["status"] == "passed" for report in reports))
    report = {"schema_version": 1, "status": "passed" if passed else "needs_review", "sections": reports, "supported_units": supported_count,
              "generation_errors": generation_errors,
              "review_errors": review_errors,
              "revision": int(state.get("audit_revision") or 0),
              "scope": "全部正文段落与摘要；supported 为独立模型判断，仍需科研人员复核",
              "model": llm.model if llm else "unavailable"}
    writing["citation_audit_status"] = report["status"]
    refs = list(state.get("research_artifact_refs") or [])
    ref = await _save(state, "citation_audit", "citation_audit.json", json.dumps(report, ensure_ascii=False, indent=2), reporter,
                      revision=report["revision"])
    if ref:
        refs.append(ref)
    if reporter:
        # 中文说明：把缺正文、写作检查失败和证据问题分开告诉前端，
        # 用户不必打开 JSON 文件才能知道这轮为何没有通过。
        failed_audits = sum(item["status"] != "passed" and item.get("section_id") not in generation_errors
                            for item in reports)
        audit_message = (
            "独立核查通过" if passed else
            f"独立核查未通过：{len(generation_errors)} 处正文缺失或摘要生成异常，"
            f"{len(review_errors)} 节写作检查未通过，{failed_audits} 处证据核查未通过"
        )
        reporter.completed(audit_message, stage="audit_done",
                           generation_error_count=len(generation_errors),
                           review_error_count=len(review_errors),
                           audit_issue_count=failed_audits)
    return {"citation_audit": report, "writing_report": writing, "research_artifact_refs": refs, "current_step": "audit"}


def route_after_audit(state: State) -> str:
    """模型不可用时直接交付待核查草稿；存在具体问题时至多修订一次。"""
    audit = state.get("citation_audit") or {}
    repairable = any(unit["status"] in {"insufficient", "contradicted", "invalid_citation"}
                     for section in audit.get("sections", []) for unit in section.get("units", []))
    # 中文说明：没有正文或本地写作检查失败的小节也需要一次补写机会，
    # 即使独立审计没有产生可修正的证据单元。
    repairable = repairable or bool(audit.get("generation_errors") or audit.get("review_errors"))
    return "revise" if repairable and audit.get("model") not in {None, "unavailable"} and int(state.get("audit_revision") or 0) < 1 else "reply"


async def run_revision_node(state: State) -> State:
    """修订次数写入共享状态，避免图反向跳转时重新归零。"""
    return {"audit_revision": int(state.get("audit_revision") or 0) + 1}
