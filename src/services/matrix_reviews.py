"""保存人工矩阵修订并再次核查，原始矩阵和已生成综述始终保留。"""
from __future__ import annotations

import asyncio
import copy
import json
import uuid
from collections import defaultdict
from pathlib import Path

from src.agents.base import AgentContext
from src.agents.analyseAgent import load_analyse_agent_llm
from src.agents.citationAuditAgent import CitationAuditAgent
from src.llm import SystemConfig
from src.models.sessions import utc_now
from src.retrieval.hybrid import load_scoped_chunks
from src.services.sessions import SessionError
from src.utils.read_utils.chunkers import chunks_content_hash


def latest_matrix(repo, session_key: str, base_id: str):
    """限定当前会话及原始矩阵，旧浏览器标签不能覆盖较新的修订。"""
    record = repo.get(session_key)
    base = next((a for a in record.artifacts if a['id'] == base_id and a['name'] == 'evidence_matrix.json'
                 and a['artifact_type'] in {'evidence_matrix', 'matrix_review'}), None)
    if base is None:
        raise SessionError('找不到此会话的矩阵', 404)
    root = (base.get('metadata') or {}).get('root_artifact_id') or base['id']
    versions = [a for a in record.artifacts if a['name'] == 'evidence_matrix.json'
                and (a['id'] == root or (a.get('metadata') or {}).get('root_artifact_id') == root)]
    latest = versions[-1]
    path = repo.read_artifact_path(session_key, latest['id'])
    if path is None:
        raise SessionError('矩阵文件不可用', 404)
    return record, latest, root, json.loads(path.read_text(encoding='utf-8'))


def matrix_chunks(matrix: dict):
    """原文从会话论文对应的缓存读取，客户端不能提交任意文件路径或伪造正文。"""
    reads = [{'paper': {'paperId': row['paperId']}} for row in matrix['rows']]
    return load_scoped_chunks(Path(SystemConfig.load().read.paper_cache_dir), reads)


def matrix_ready(matrix: dict, chunks) -> bool:
    """人工确认和模型核查是两项记录；源文件变化后即使旧记录通过也需要重新核查。"""
    review = matrix.get('review') or {}
    audit = review.get('audit') or {}
    return (review.get('confirmed') is True and audit.get('status') == 'passed'
            and audit.get('source_hash') == chunks_content_hash(chunks))


class MatrixReviewService:
    """单机服务按会话串行修改，并检查版本号，防止两个页面互相覆盖。"""

    def __init__(self, repo):
        self.repo = repo
        self.locks = defaultdict(asyncio.Lock)

    async def update(self, session_key: str, body: dict, *, llm='auto') -> dict:
        async with self.locks[session_key]:
            record, artifact, root, matrix = latest_matrix(self.repo, session_key, str(body.get('base_artifact_id') or ''))
            if record.run_started_at or record.status in {'running', 'cancel_requested'}:
                raise SessionError('流程运行期间不能修改矩阵，请等待矩阵确认阶段或任务结束', 409)
            if artifact['id'] != body.get('base_artifact_id'):
                raise SessionError('矩阵已有新版本，请刷新后再编辑', 409)
            action = body.get('action')
            if not isinstance(action, str) or action not in {'edit', 'confirm', 'recheck'}:
                raise ValueError('action 必须是 edit、confirm 或 recheck')
            reviewer = str(body.get('reviewer') or '').strip()
            if action != 'recheck' and not 1 <= len(reviewer) <= 80:
                raise ValueError('请填写本地审阅者姓名或标识（1–80 字）')
            chunks = await asyncio.to_thread(matrix_chunks, matrix)
            by_id = {chunk.chunk_id: chunk for chunk in chunks}
            updated = copy.deepcopy(matrix)
            review = dict(updated.get('review') or {})
            if action == 'edit':
                row = next((row for row in updated['rows'] if row['paperId'] == body.get('paperId')), None)
                dimension = body.get('dimension')
                if row is None or not isinstance(dimension, str) or dimension not in row['cells']:
                    raise ValueError('论文或矩阵维度不存在')
                value, note = body.get('value'), body.get('note', '')
                if not isinstance(value, str) or len(value) > 8000 or not isinstance(note, str) or len(note) > 2000:
                    raise ValueError('单元格需为不超过 8000 字的文字，说明不超过 2000 字')
                sources = body.get('evidence', [])
                if not isinstance(sources, list) or len(sources) > 8:
                    raise ValueError('每格最多 8 条原文证据')
                evidence = []
                for source in sources:
                    chunk = by_id.get(str(source.get('chunkId') or '')) if isinstance(source, dict) else None
                    quote = source.get('quote') if isinstance(source, dict) else None
                    if (chunk is None or chunk.paperId.casefold() != row['paperId'].casefold()
                            or not isinstance(quote, str) or len(quote.strip()) < 4 or quote.strip() not in chunk.content):
                        raise ValueError('证据必须是当前论文切片中的连续原文；来源改变后请重新阅读论文')
                    evidence.append({'chunkId': chunk.chunk_id, 'paperId': chunk.paperId, 'quote': quote.strip(),
                                     'page_start': chunk.page_start, 'page_end': chunk.page_end, 'section': chunk.section})
                if value.strip() and not evidence:
                    raise ValueError('新增或修改事实必须保留至少一条原文证据；缺失项请留空')
                if not value.strip():
                    evidence = []  # 清空内容即恢复缺失状态，不保留会抬高覆盖率的旧引句。
                row['cells'][dimension] = {'value': value.strip(), 'status': 'human_edited' if value.strip() else 'not_found',
                    'evidence': evidence, 'edited_by': reviewer, 'edited_at': utc_now(), 'review_note': note.strip()}
                review.update(confirmed=False, audit={'status': 'unverified', 'reason': '矩阵已修改，需要重新核查'})
            elif action == 'confirm':
                # 确认是用户对整张矩阵的明确操作，不冒充模型判断或经过身份认证的签名。
                review.update(confirmed=True, confirmed_by=reviewer, confirmed_at=utc_now())
            else:
                snapshot = load_analyse_agent_llm(agent_name='solar_agent') if llm == 'auto' else llm
                owns = llm == 'auto'
                agent = CitationAuditAgent(AgentContext(llm=snapshot))
                rows = []
                try:
                    for row in updated['rows']:
                        claims, bindings, dimensions = [], [], []
                        for dimension, cell in row['cells'].items():
                            if not cell['value']:
                                continue
                            claim = f"{updated['dimensions'][dimension]}：{cell['value']} [{row['paperId']}]"
                            claims.append(claim); dimensions.append(dimension)
                            bindings.append({'claim': claim, 'chunks': cell.get('evidence') or []})
                        if not claims:
                            continue
                        paper_chunks = [chunk for chunk in chunks if chunk.paperId.casefold() == row['paperId'].casefold()]
                        result = await agent.audit({'section_id': row['paperId'], 'source_content': '\n\n'.join(claims),
                            'citation_evidence': bindings}, paper_chunks, {row['paperId'].casefold(): row['paperId'].casefold()}, [])
                        result['dimensions'] = dimensions
                        rows.append(result)
                finally:
                    if owns and snapshot:
                        await snapshot.aclose()
                review['audit'] = {'status': 'passed' if rows and all(row['status'] == 'passed' for row in rows) else 'needs_review',
                    'rows': rows, 'source_hash': chunks_content_hash(chunks), 'checked_at': utc_now(),
                    'model': snapshot.model if snapshot else 'unavailable'}
            review.update(revision=int(review.get('revision') or 0) + 1, action=action, updated_at=utc_now(),
                          root_artifact_id=root, parent_artifact_id=artifact['id'])
            updated['review'] = review
            updated['located_cells'] = sum(bool(cell.get('evidence')) for row in updated['rows'] for cell in row['cells'].values())
            updated['coverage'] = updated['located_cells'] / updated['total_cells'] if updated['total_cells'] else 0
            # 在模型核查期间如果任务被另一个入口启动，不能再把修订写入运行中的会话。
            current = self.repo.get(session_key)
            if current.run_started_at or current.status in {'running', 'cancel_requested'}:
                raise SessionError('任务已开始运行，本次修订未保存', 409)
            if latest_matrix(self.repo, session_key, root)[1]['id'] != artifact['id']:
                raise SessionError('矩阵已变更，请刷新后重试', 409)
            return await asyncio.to_thread(self._save, session_key, artifact, root, updated)

    def _save(self, session_key, original, root, matrix):
        """先写导出再写作为版本入口的 JSON，失败的半份导出不会成为最新矩阵。"""
        from src.graph.evidence_node import _matrix_exports
        csv, markdown = _matrix_exports(matrix)
        version = uuid.uuid4().hex
        metadata = {'turn_id': (original.get('metadata') or {}).get('turn_id'), 'root_artifact_id': root,
                    'revision': matrix['review']['revision'], 'parent_artifact_id': original['id']}
        artifacts = []
        for name, content in [('evidence_matrix.csv', csv), ('evidence_matrix.md', markdown),
                              ('evidence_matrix.json', json.dumps(matrix, ensure_ascii=False, indent=2))]:
            artifacts.append(self.repo.write_artifact(session_key, 'matrix_review', name, content,
                relative_path=f'artifacts/matrix_review/{root}/{version}/{name}', metadata=metadata))
        return {'matrix': matrix, 'artifact': artifacts[-1], 'artifacts': artifacts}
