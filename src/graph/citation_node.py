"""可选引文展开在初次阅读之前执行，新增论文仍经过原有相关性筛选。"""
from __future__ import annotations

import json

from src.agents.base import AgentContext
from src.agents.citationSnowballAgent import CitationSnowballAgent, parse_seed_identifiers
from src.graph.evidence_node import _paper_keys, _reporter, _save
from src.graph.state_models import State
from src.llm import SystemConfig
from src.paper_retrieval.connectors.semantic_scholar import SemanticScholarPaperConnector
from src.paper_retrieval.models import SearchRequest


async def run_citation_node(state: State) -> State:
    """用户明确打开开关才查询引文；最多增加 5 篇，不突破用户指定总篇数。"""
    if state["request"].constraints.get("citation_snowball") is not True:
        return {}
    reporter = _reporter(state, "citation_discovery", "引文扩展")
    if reporter:
        reporter.started("按种子、层数和页数限额查询引文关系", stage="citation_start")
    papers = list(state.get("search_results") or [])
    constraints = dict(state["request"].constraints)
    summary = state.get("search_summary") or {}
    for key in ("year_from", "year_to", "excluded_terms"):
        if key not in constraints and key in summary:
            constraints[key] = summary[key]
    runtime = state.get("runtime_context")
    seeds = parse_seed_identifiers(constraints["citation_seed_ids"]) if constraints.get("citation_seed_ids") else papers[:3]
    result = await CitationSnowballAgent(AgentContext()).expand(seeds, constraints=constraints,
        api_key=SystemConfig.load().paper_retrieval.semantic_scholar_api_key,
        client=getattr(getattr(runtime, "resources", None), "http_client", None),
        cancellation=getattr(runtime, "cancellation", None),
        progress=lambda message: reporter.progress(message, stage="citation_expand") if reporter else None)
    node_connections = {}
    for link in result["links"]:
        node_connections.setdefault(link["discovered_paper_id"], set()).add(link["seed"])
    resolved_seeds = result.pop("seed_papers")
    candidates = sorted(result.pop("papers"), key=lambda paper: (-len(node_connections.get(paper.paperId, [])), paper.paperId or paper.id))
    seen = set().union(*(_paper_keys(paper) for paper in papers)) if papers else set()
    budget = 5
    if constraints.get("max_results") is not None:
        try:
            budget = min(5, max(0, int(constraints["max_results"]) - len(papers)))
        except (TypeError, ValueError):
            pass
    added = []
    selection = SearchRequest(year_from=constraints.get("year_from"), year_to=constraints.get("year_to"),
                              excluded_terms=constraints.get("excluded_terms") or [])
    # 指定种子先占用新增名额，仍遵守用户设置的总篇数。
    for paper in (resolved_seeds if constraints.get("citation_seed_ids") else []) + candidates:
        if len(added) >= budget:
            break
        # 指定种子同样遵守年份与排除词；图中可保留定位记录，但不能越过阅读约束。
        if (not SemanticScholarPaperConnector._within_year_range(paper, selection)
            or SemanticScholarPaperConnector._contains_excluded_terms(paper, selection.excluded_terms)):
            continue
        if not paper.abstract or seen.intersection(_paper_keys(paper)):
            continue
        seen.update(_paper_keys(paper))
        paper.metadata["citation_discovery"] = {"connected_expansion_node_count": len(node_connections.get(paper.paperId, []))}
        added.append(paper)
    report = {**result, "schema_version": 1, "seed_paper_ids": [paper.paperId or paper.id for paper in resolved_seeds],
              "seed_papers": [paper.to_dict() for paper in resolved_seeds],
              "papers": [paper.to_dict() for paper in candidates], "added_paper_ids": [paper.id for paper in added],
              "connected_expansion_node_counts": {key: len(value) for key, value in node_connections.items()},
              "limits": {"seeds": 3, "depth": result["depth"], "pages_per_direction": result["pages_per_direction"], "per_page": 20, "max_requests": 24, "max_new_papers": budget},
              "scope": "限额引文发现，不保证找全奠基文献；共被引计数仅覆盖本次实际取得的关系，不是全局影响力。"}
    refs = list(state.get("research_artifact_refs") or [])
    ref = await _save(state, "citation_graph", "citation_graph.json", json.dumps(report, ensure_ascii=False, indent=2), reporter)
    if ref:
        refs.append(ref)
    if reporter:
        reporter.completed(f"发现 {len(candidates)} 篇候选，新增 {len(added)} 篇进入阅读", stage="citation_done")
    return {"search_results": papers + added, "citation_graph": report, "research_artifact_refs": refs}
