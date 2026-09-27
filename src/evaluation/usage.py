"""收集供应商实际返回的用量，字段缺失和未知账单不能按零处理。"""
from .benchmark import number


def usage_record(model, operation, raw, ok):
    raw = raw if isinstance(raw, dict) else {}
    def count(*keys):
        value = next((raw[key] for key in keys if raw.get(key) is not None), None)
        try:
            return number(value)
        except ValueError:
            return None
    incoming = count('input_tokens', 'prompt_tokens')
    output = count('output_tokens', 'completion_tokens')
    details = raw.get('prompt_tokens_details') or raw.get('input_tokens_details') or {}
    cached = details.get('cached_tokens') if isinstance(details, dict) else None
    if 'cache_read_input_tokens' in raw or 'cache_creation_input_tokens' in raw:
        cached = count('cache_read_input_tokens')
        creation = count('cache_creation_input_tokens')
        # Anthropic 输入分成普通、写入缓存和读取缓存三项；少一项就保留未知。
        incoming = incoming + cached + creation if all(v is not None for v in (incoming, cached, creation)) else None
    if 'embed' in operation:
        output = 0  # 向量接口没有生成文本 token。
    try:
        cached = number(cached)
    except ValueError:
        cached = None
    return {'model': model, 'operation': operation, 'ok': ok,
            'input_tokens': incoming, 'output_tokens': output, 'cached_input_tokens': cached}


def summarize_usage(records):
    totals = {}
    for key in ('input_tokens', 'output_tokens', 'cached_input_tokens'):
        values = [record[key] for record in records]
        totals[key] = sum(values) if values and all(value is not None for value in values) else None
    totals['cost_usd'] = None
    return totals
