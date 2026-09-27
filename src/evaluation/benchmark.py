"""离线指标不调用模型；缺少人工标注的质量项返回 null，不拿模型自评冒充标准答案。"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean


METRICS = ('retrieval_precision_at_k', 'retrieval_recall_at_k', 'chunk_precision_at_k', 'chunk_recall_at_k', 'citation_precision',
           'citation_recall', 'faithfulness', 'latency_seconds', 'input_tokens', 'output_tokens',
           'cached_input_tokens', 'cost_usd', 'ragas_faithfulness', 'ragas_answer_relevancy')

CHUNK_CITATION = re.compile(r'\[[^\[\]\s:]+:h\d+-[0-9a-f]+:s\d+-[0-9a-f]+\]')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_cases(path):
    """题库使用逐行 JSON，空行可保留；重复编号或没有研究问题时拒绝运行。"""
    cases = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    ids = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get('id'), str) or not case['id'].strip() or case['id'] in ids:
            raise ValueError('每题必须具有唯一的非空 id')
        if not isinstance(case.get('question'), str) or not case['question'].strip():
            raise ValueError(f"{case['id']} 缺少研究问题")
        ids.add(case['id'])
        relevant = case.get('relevant_paper_ids', [])
        chunk_ids = case.get('relevant_chunk_ids', [])
        if not isinstance(chunk_ids, list) or any(not isinstance(v, str) or not v.strip() for v in chunk_ids) or len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError('relevant_chunk_ids 必须为无重复的非空字符串列表')
        if not isinstance(relevant, list) or any(not isinstance(v, str) or not v.strip() for v in relevant) or len(set(relevant)) != len(relevant):
            raise ValueError('relevant_paper_ids 必须为无重复的非空字符串列表')
        if case.get('annotation_status') == 'verified':
            if not case.get('reviewer') or not case.get('reference_answer') or not case.get('relevant_paper_ids'):
                raise ValueError(f"{case['id']} 标注为 verified 时必须有审阅者、参考答案和相关论文编号")
    if not cases:
        raise ValueError('题库为空')
    return cases


def number(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('用量、费用和耗时必须为有限非负数字或 null')
    return value


def score(cases, run, k=5, annotations=None):
    """预测、人工判断分开存放；判断绑定回答摘要，修改回答后旧判断失效。"""
    if type(k) is not int or k < 1:
        raise ValueError('K 必须是正整数')
    if not isinstance(run.get('profile'), dict) or not run['profile'].get('model') or not run['profile'].get('prompt_version'):
        raise ValueError('run.profile 必须记录模型、提示词版本和检索配置')
    if run.get('dataset_hash') and run['dataset_hash'] != fingerprint(cases):
        raise ValueError('预测记录来自不同题库，不能与当前标准答案混用')
    predictions = run.get('predictions')
    if not isinstance(predictions, list):
        raise ValueError('run.predictions 必须为列表')
    by_id = {}
    case_ids = {c['id'] for c in cases}
    for prediction in predictions:
        key = prediction.get('case_id')
        if key not in case_ids or key in by_id:
            raise ValueError('预测包含未知或重复 case_id')
        if not isinstance(prediction.get('response'), str):
            raise ValueError('每份预测必须包含 response 文本')
        by_id[key] = prediction
    judgments = {}
    for annotation in annotations or []:
        key = annotation.get('case_id')
        if key not in by_id or key in judgments or not annotation.get('reviewer'):
            raise ValueError('人工判断需唯一、具有审阅者，并对应本次预测')
        if annotation.get('response_hash') != fingerprint(by_id[key]['response']):
            raise ValueError(f'{key} 回答已改变，人工判断失效')
        # 人工分母必须给出实际逐条判断，不能直接输入希望得到的总分。
        for field in ('claims', 'citations'):
            values = annotation.get(field)
            if values is not None and (not isinstance(values, list) or any(not isinstance(v, dict) or type(v.get('supported')) is not bool or not v.get('text') for v in values)):
                raise ValueError(f'{field} 必须逐条填写 text 和 supported 布尔值')
        citations = annotation.get('citations')
        response_citations = CHUNK_CITATION.findall(by_id[key]['response'])
        if citations and (response_citations or any(CHUNK_CITATION.search(c['text']) for c in citations)):
            labeled = [CHUNK_CITATION.findall(c['text']) for c in citations]
            if any(len(ids) != 1 for ids in labeled) or Counter(ids[0] for ids in labeled) != Counter(response_citations):
                raise ValueError(f'{key} 引用判断必须逐次覆盖回答中的引用编号，且每条判断只对应一个编号')
        judgments[key] = annotation
    rows = []
    for case in cases:
        prediction = by_id.get(case['id'])
        row = {'case_id': case['id'], 'status': 'missing' if prediction is None else 'scored', **dict.fromkeys(METRICS)}
        if prediction is None:
            rows.append(row)
            continue
        # 只接收供应商实际用量；未记录缓存命中时保留 null，不能推算“节约 58%”。
        row['latency_seconds'] = number(prediction.get('latency_seconds'))
        usage = prediction.get('usage') or {}
        for metric in ('input_tokens', 'output_tokens', 'cached_input_tokens', 'cost_usd'):
            row[metric] = number(usage.get(metric))
        if row['cached_input_tokens'] is not None and row['input_tokens'] is not None and row['cached_input_tokens'] > row['input_tokens']:
            raise ValueError('缓存输入 token 不能超过总输入 token')
        # 检索命中和回答生成是两个阶段。即使回答被引用门禁拒绝，已实际返回的
        # 检索片段仍应参与 P@K/R@K；否则会把生成失败误报为检索失败。
        retrieved = prediction.get('retrieved_paper_ids', [])
        if not isinstance(retrieved, list) or any(not isinstance(value, str) for value in retrieved):
            raise ValueError('retrieved_paper_ids 必须为字符串列表')
        ranked = list(dict.fromkeys(retrieved))[:k]
        if case.get('annotation_status') == 'verified':
            relevant = set(case['relevant_paper_ids'])
            hits = len(set(ranked) & relevant)
            # Precision@K 使用固定 K 分母，返回不足 K 篇不会把一条命中夸成满分。
            row['retrieval_precision_at_k'] = hits / k
            row['retrieval_recall_at_k'] = hits / len(relevant)
            # 只按人工列出的精确片段编号计分；没有片段标注时保留空值。
            # 相邻重叠片段不自动视为等价，报告须说明人工证据集合可能并不穷尽。
            gold_chunks = set(case.get('relevant_chunk_ids') or [])
            retrieved_chunks = prediction.get('retrieved_chunks')
            if gold_chunks and retrieved_chunks is not None:
                if not isinstance(retrieved_chunks, list) or any(not isinstance(c, dict) or not isinstance(c.get('chunkId'), str) or not c['chunkId'].strip() for c in retrieved_chunks):
                    raise ValueError('retrieved_chunks 必须包含有效 chunkId')
                ranked_chunks = list(dict.fromkeys(c['chunkId'] for c in retrieved_chunks))[:k]
                chunk_hits = len(set(ranked_chunks) & gold_chunks)
                row['chunk_precision_at_k'] = chunk_hits / k
                row['chunk_recall_at_k'] = chunk_hits / len(gold_chunks)
        if prediction.get('status') == 'failed':
            row['status'] = 'failed'
            rows.append(row)
            continue
        row['response_hash'] = fingerprint(prediction['response'])
        judgment = judgments.get(case['id'], {})
        claims, citations = judgment.get('claims'), judgment.get('citations')
        if claims:
            row['faithfulness'] = sum(c['supported'] for c in claims) / len(claims)
        if citations:
            row['citation_precision'] = sum(c['supported'] for c in citations) / len(citations)
        if claims and all(type(c.get('citation_supported')) is bool for c in claims):
            row['citation_recall'] = sum(c['citation_supported'] for c in claims) / len(claims)
        if prediction.get('ragas') and not run['profile'].get('evaluator'):
            raise ValueError('Ragas 分数必须记录 evaluator 模型、版本和提示词身份')
        for metric, value in (prediction.get('ragas') or {}).items():
            field = 'ragas_' + metric
            if field in METRICS:
                value = number(value)
                if value is not None and value > 1:
                    raise ValueError('Ragas 指标应位于 0–1')
                row[field] = value
        rows.append(row)
    summary = {}
    for metric in METRICS:
        values = [row[metric] for row in rows if row[metric] is not None]
        summary[metric] = {'mean': mean(values) if values else None, 'count': len(values), 'total': len(rows)}
    return {'schema_version': 2, 'dataset_hash': fingerprint(cases), 'run_hash': fingerprint(run),
            'profile': run['profile'], 'k': k, 'rows': rows, 'summary': summary,
            'note': '人工指标与 Ragas 模型判断分开显示；null 表示缺少数据，不是零分。'}


def compare(before, after):
    """逐题配对比较，不能拿不同题库或两组不同的有效样本均值直接相减。"""
    if before.get('schema_version') != 2 or after.get('schema_version') != 2:
        raise ValueError('请使用当前评分器重新生成两份评分结果')
    if before['dataset_hash'] != after['dataset_hash'] or before['k'] != after['k']:
        raise ValueError('比较要求相同题库摘要和 K')
    if before['profile'].get('evaluator') != after['profile'].get('evaluator'):
        raise ValueError('比较要求相同评测模型与提示词设置')
    left, right = ({r['case_id']: r for r in report['rows']} for report in (before, after))
    if left.keys() != right.keys():
        raise ValueError('比较要求相同题目编号')
    result = {}
    for metric in METRICS:
        pairs = [(left[key][metric], right[key][metric]) for key in left if left[key][metric] is not None and right[key][metric] is not None]
        result[metric] = {'paired_count': len(pairs), 'total': len(left),
            'before': mean(a for a, _ in pairs) if pairs else None,
            'after': mean(b for _, b in pairs) if pairs else None,
            'delta': mean(b - a for a, b in pairs) if pairs else None}
    return {'dataset_hash': before['dataset_hash'], 'before_profile': before['profile'],
            'after_profile': after['profile'], 'metrics': result,
            'note': '仅比较两次都有数据的同一题；不据此自动宣称改进具有统计显著性。'}
