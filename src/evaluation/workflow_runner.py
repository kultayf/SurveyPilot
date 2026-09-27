"""顺序运行题库并逐题保存结果；费用与缓存命中缺少可靠账单时不猜测。"""
import asyncio
import copy
from dataclasses import asdict
import json
import time
from pathlib import Path

from .benchmark import fingerprint


async def run_cases(cases, profile, output, storage_root, limit):
    from src.agents.contracts import ReviewRequest
    from src.graph.graph import run_graph
    from src.llm.base import usage_observer
    from src.llm import SystemConfig
    from .usage import usage_record, summarize_usage
    from src.repositories.sessions.sqlite import SQLiteSessionRepository
    from src.services.matrix_reviews import matrix_chunks
    if not isinstance(profile, dict) or not profile.get('model') or not profile.get('prompt_version'):
        raise ValueError('profile 必须记录实际模型与 prompt_version；可同时记录 constraints 和检索参数')
    if limit < 1:
        raise ValueError('limit 必须大于零')
    profile = copy.deepcopy(profile)
    # 保存实际系统参数的摘要，原始模型配置中的认证信息不进入评测文件。
    profile['system_config_hash'] = fingerprint(asdict(SystemConfig.load()))
    constraints = dict(profile.get('constraints') or {})
    if constraints.get('confirm_matrix_before_writing'):
        raise ValueError('批量评测不能使用交互式矩阵确认，请在工作台单独完成此流程')
    if Path(output).exists():
        raise ValueError('输出文件已存在，请为新实验选择新文件名以保留历史结果')
    repo = SQLiteSessionRepository(storage_root=storage_root)
    result = {'profile': profile, 'dataset_hash': fingerprint(cases), 'predictions': []}
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    for case in cases[:limit]:
        session = repo.create('评测 ' + case['id'])
        started = time.monotonic()
        prediction = {'case_id': case['id'], 'session_key': session.key, 'response': '', 'status': 'failed',
                      'retrieved_paper_ids': [], 'retrieved_contexts': [], 'usage': None}
        records = []
        token = usage_observer.set(lambda model, operation, raw, ok: records.append(usage_record(model, operation, raw, ok)))
        try:
            # 题目若指定论文，必须把目标题名交给检索，不能只搜索脱离文献背景的问题。
            sources = case.get('seed_sources') or []
            topic = case['question'] + ('\n目标文献：' + '；'.join(str(s.get('title') or '') + ' ' + str(s.get('url') or '') for s in sources) if sources else '')
            graph = await run_graph(ReviewRequest(topic=topic, constraints=constraints),
                                    session_repo=repo, session_key=session.key, turn_id='benchmark')
            state = graph.state
            files = state.get('final_artifact_refs') or []
            review = next((ref for ref in reversed(files) if ref.get('artifact_type') in {'final_review', 'draft_review'}), None)
            artifact_path = repo.read_artifact_path(session.key, review.get('artifact_id') or review.get('id')) if review else None
            prediction['response'] = artifact_path.read_text(encoding='utf-8') if artifact_path else str(state.get('assistant_message') or '')
            prediction['retrieved_paper_ids'] = [paper.paperId for paper in graph.papers if paper.paperId]
            # 仅保存写作工具确实绑定的片段；不把整个文献缓存冒充“本次取回的上下文”。
            used_ids = {item.get('chunkId') for section in state.get('writing_sections') or []
                        for binding in section.get('citation_evidence') or [] for item in binding.get('chunks') or []}
            chunks = await asyncio.to_thread(matrix_chunks, state.get('evidence_matrix') or {'rows': []})
            prediction['retrieved_contexts'] = [chunk.content for chunk in chunks if chunk.chunk_id in used_ids]
            prediction['status'] = 'completed'
            prediction['audit_status'] = (state.get('citation_audit') or {}).get('status', 'unverified')
        except Exception as exc:
            # 错误类型足以定位失败；原始异常可能包含服务商地址或认证信息，不写入公共报告。
            prediction['error_type'] = type(exc).__name__
        finally:
            usage_observer.reset(token)
        prediction['model_calls'] = records
        prediction['usage'] = summarize_usage(records)
        prediction['latency_seconds'] = time.monotonic() - started
        result['predictions'].append(prediction)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(path)
        print(f"{case['id']}: {prediction['status']}，已保存 {len(result['predictions'])} 题", flush=True)
    return result
