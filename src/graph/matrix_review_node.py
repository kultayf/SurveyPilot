"""用户可选择在分析和写作之前核对矩阵，确认后的恢复不重复下载论文。"""
from __future__ import annotations

import asyncio
import json

from src.graph.evidence_node import _reporter, _save, _matrix_exports
from src.graph.state_models import State
from src.services.matrix_reviews import latest_matrix, matrix_chunks, matrix_ready


class MatrixConfirmationRequired(RuntimeError):
    """矩阵需要人工确认；恢复现场已单独保存在当前会话。"""


async def run_matrix_review_node(state: State) -> State:
    if state['request'].constraints.get('confirm_matrix_before_writing') is not True:
        return {}
    repo, key = state.get('session_repo'), state.get('session_key')
    if repo is None or not key:
        raise ValueError('写作前人工确认需要在有持久化存储的会话中运行')
    checkpoint = state.get('read_resume_checkpoint') or {}
    base_id = checkpoint.get('matrix_artifact_id') or next((ref.get('artifact_id') or ref.get('id')
        for ref in reversed(state.get('research_artifact_refs') or []) if ref.get('name') == 'evidence_matrix.json'), None)
    if not base_id:
        raise ValueError('没有已保存的矩阵，无法进行人工确认')
    _, artifact, _, matrix = latest_matrix(repo, key, base_id)
    chunks = await asyncio.to_thread(matrix_chunks, matrix)
    if matrix_ready(matrix, chunks):
        # 恢复属于新一轮执行，重新保存已确认的快照，前端才能把它与本轮核查对应。
        matrix['confirmed_from_artifact_id'] = artifact['id']
        csv, markdown = _matrix_exports(matrix)
        refs = list(state.get('research_artifact_refs') or [])
        for name, content in [('evidence_matrix.csv', csv), ('evidence_matrix.md', markdown),
                              ('evidence_matrix.json', json.dumps(matrix, ensure_ascii=False, indent=2))]:
            ref = await _save(state, 'evidence_matrix', name, content, None)
            if ref:
                refs.append(ref)
        return {'evidence_matrix': matrix, 'research_artifact_refs': refs,
                'read_resume_checkpoint': None, 'current_step': 'matrix_confirmed'}
    # 只保存可以写成 JSON 的资料，模型连接、密钥和仓储对象不能进入恢复文件。
    fields = ('read_results', 'read_summary', 'read_artifact_refs', 'search_summary', 'search_output',
              'search_artifact_refs', 'research_round', 'supplemental_paper_ids', 'supplemental_search',
              'research_artifact_refs', 'citation_graph', 'evidence_matrix')
    checkpoint = {field: state[field] for field in fields if field in state}
    checkpoint.update(resume_stage='matrix_review', matrix_artifact_id=artifact['id'],
        recovery_status='waiting_matrix_confirmation', current_step='matrix_waiting_confirmation',
        request={'topic': state['request'].topic, 'constraints': state['request'].constraints, 'language': state['request'].language},
        search_results=[paper.to_dict() for paper in state.get('search_results') or []])
    reporter = _reporter(state, 'matrix_confirmation', '人工确认矩阵')
    await _save(state, 'matrix_checkpoint', 'matrix_checkpoint.json', json.dumps(checkpoint, ensure_ascii=False, indent=2), reporter)
    message = '请在矩阵面板完成编辑、确认整张矩阵并重新核查，再点击继续执行。'
    if reporter:
        reporter.failed(message, stage='matrix_waiting_confirmation', checkpoint=checkpoint,
                        recovery_status='waiting_matrix_confirmation')
    raise MatrixConfirmationRequired(message)
