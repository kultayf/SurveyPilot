"""在写作前保存争议线索，无法检查时明确保留未验证状态。"""
import asyncio
import json

from src.agents.base import AgentContext
from src.agents.conflictAgent import ConflictAgent
from src.graph.evidence_node import _model, _reporter, _save, _usage_callback
from src.services.matrix_reviews import matrix_chunks


async def run_conflict_node(state):
    reporter = _reporter(state, 'conflict', '跨文献比较')
    if reporter:
        reporter.started('正在核对跨论文比较条件与原文依据', stage='conflict_start')
    llm, owned = _model(state, 'conflict_node_llm', 'solar_agent')
    try:
        matrix = state.get('evidence_matrix') or {'rows': []}
        chunks = await asyncio.to_thread(matrix_chunks, matrix)
        report = await ConflictAgent(AgentContext(llm=llm, usage_callback=_usage_callback(reporter, 'conflict_check'))).inspect(matrix, chunks)
    finally:
        if owned and llm:
            await llm.aclose()
    refs = list(state.get('research_artifact_refs') or [])
    ref = await _save(state, 'conflict_report', 'conflict_report.json', json.dumps(report, ensure_ascii=False, indent=2), reporter)
    if ref:
        refs.append(ref)
    if reporter:
        reporter.completed('比较线索已保存，结论仍需人工复核', stage='conflict_done', findings=len(report['findings']))
    return {'conflict_report': report, 'research_artifact_refs': refs}
