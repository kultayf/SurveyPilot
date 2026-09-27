"""生成回答级人工复核材料；只整理实际输出，不自动判断事实或引文是否正确。"""
from __future__ import annotations

import json
from pathlib import Path

from .benchmark import fingerprint


def _fenced(value: str) -> str:
    """避免回答本身含有 Markdown 围栏时破坏复核材料布局。"""
    delimiter = '````' if '```' in value else '```'
    return f'{delimiter}\n{value}\n{delimiter}'


def prepare_review_packet(cases, run, packet_path, annotations_path):
    """为一次已冻结的运行生成 Markdown 阅读材料和空白人工判断表。"""
    packet_path, annotations_path = Path(packet_path), Path(annotations_path)
    if packet_path.exists() or annotations_path.exists():
        raise ValueError('人工复核材料或判断模板已存在，请使用新的文件名')
    if run.get('dataset_hash') and run['dataset_hash'] != fingerprint(cases):
        raise ValueError('运行记录来自不同题库，不能生成本次复核材料')
    predictions = run.get('predictions')
    if not isinstance(predictions, list):
        raise ValueError('run.predictions 必须为列表')
    case_by_id = {case['id']: case for case in cases}
    by_id = {}
    for prediction in predictions:
        case_id = prediction.get('case_id')
        if case_id not in case_by_id or case_id in by_id or not isinstance(prediction.get('response'), str):
            raise ValueError('预测必须有唯一的已知 case_id 与 response 文本')
        by_id[case_id] = prediction

    profile = run.get('profile')
    if not isinstance(profile, dict) or not profile.get('model') or not profile.get('prompt_version'):
        raise ValueError('运行记录缺少实际模型或提示词版本')
    lines = [
        '# 模型回答的人工复核材料',
        '',
        '本文件只整理一次实际运行的回答、人工冻结的题库参考答案和模型实际取回的原文。',
        '请逐条核对回答中的全部事实主张及全部引用；不得把参考答案、模型自述或引用格式通过当作人工判断。',
        '被门禁拒绝的回答保留用于审计，但不进入下方的人工判断模板，也不能在评分中转为成功。',
        '',
        '## 运行身份',
        '',
        f'- 模型：`{profile["model"]}`',
        f'- 提示词摘要：`{profile["prompt_version"]}`',
        f'- 检索模式：`{profile.get("retrieval_mode", "未记录")}`',
        f'- 题库摘要：`{fingerprint(cases)}`',
        '',
        '## 填写方法',
        '',
        '1. 打开配套 JSON 模板。每个非失败回答都已绑定本次 `response_hash`。',
        '2. 将回答拆成全部可核实的事实主张，逐条填入 `claims`，并填写 `supported` 与 `citation_supported`。',
        '3. 将每一个引用及其对应主张填入 `citations`，逐条填写 `supported`。',
        '4. 将 `reviewer` 改为实际复核人标识。空数组和空 reviewer 只是待填写模板，不能直接传给 `score --annotations`。',
        '',
    ]
    annotations = []
    for index, case in enumerate(cases, 1):
        prediction = by_id.get(case['id'])
        if prediction is None:
            lines.extend([f'## {index}. {case["id"]}', '', '本次运行缺少该题预测。', ''])
            continue
        response = prediction['response']
        status = prediction.get('status', '未知')
        chunks = prediction.get('retrieved_chunks') or []
        contexts = prediction.get('retrieved_contexts') or []
        evidence = [
            (chunk, contexts[position] if position < len(contexts) else '')
            for position, chunk in enumerate(chunks)
            if isinstance(chunk, dict) and isinstance(chunk.get('chunkId'), str)
            and chunk['chunkId'] in response
        ]
        lines.extend([
            f'## {index}. {case["id"]}',
            '',
            f'**问题**：{case["question"]}',
            '',
            f'**运行状态**：`{status}`' + (f'；错误：`{prediction["error_type"]}`' if prediction.get('error_type') else ''),
            '',
            '### 本次模型回答',
            '',
            _fenced(response),
            '',
        ])
        if status != 'failed':
            response_hash = fingerprint(response)
            lines.extend([f'**本次回答摘要**：`{response_hash}`', ''])
            annotations.append({
                'case_id': case['id'], 'response_hash': response_hash, 'reviewer': '',
                'claims': [], 'citations': [],
            })
        else:
            lines.extend(['该回答已被运行门禁拒绝；以下材料只用于定位问题，不能填写为通过。', ''])
        lines.extend(['### 冻结题库参考答案（复核辅助，不等同于本回答金标）', '', _fenced(case['reference_answer']), ''])
        lines.extend(['### 模型取回的片段', ''])
        if not chunks:
            lines.extend(['本次没有记录可用片段。', ''])
        else:
            lines.extend(['| chunkId | 页码 | 是否在回答中逐字引用 |', '| --- | --- | --- |'])
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                chunk_id = str(chunk.get('chunkId', '缺失'))
                pages = f'{chunk.get("page_start", "?")}–{chunk.get("page_end", "?")}'
                lines.append(f'| `{chunk_id}` | {pages} | {"是" if chunk_id in response else "否"} |')
            lines.append('')
        if evidence:
            lines.extend(['### 回答实际引用的原文', ''])
            for chunk, content in evidence:
                lines.extend([
                    f'#### `{chunk["chunkId"]}`（第 {chunk.get("page_start", "?")}–{chunk.get("page_end", "?")} 页）',
                    '', _fenced(content), '',
                ])
        lines.extend(['---', ''])

    packet_path.parent.mkdir(parents=True, exist_ok=True)
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text('\n'.join(lines), encoding='utf-8')
    annotations_path.write_text(json.dumps(annotations, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return {
        'packet_path': str(packet_path), 'annotations_path': str(annotations_path),
        'included_predictions': len(by_id), 'annotation_rows': len(annotations),
        'failed_predictions': sum(prediction.get('status') == 'failed' for prediction in by_id.values()),
    }
