"""可选 Ragas 评测；只有显式运行 ragas 命令时才读取模型环境变量并产生调用费用。"""
from __future__ import annotations

import copy
import math
import os
from importlib.metadata import version
from urllib.parse import urlsplit, urlunsplit

from .benchmark import fingerprint, score


def _single_response_llm(chat_model):
    """每次只请求一个回答，避免兼容节点忽略 n，以及并发指标改写同一个客户端参数。"""
    from ragas.llms import LangchainLLMWrapper
    from langchain_core.outputs import LLMResult

    class SingleResponseLLM(LangchainLLMWrapper):
        def _client(self, n):
            if type(n) is not int or not 1 <= n <= 3:
                raise ValueError('当前评测每个提示词支持 1–3 次独立回答')
            # 浅复制只隔离请求参数，HTTP 连接仍复用；不会让两个指标互相改变 n 或温度。
            return self.langchain_llm.model_copy(update={'n': 1, 'temperature': 0})

        @staticmethod
        def _one(result):
            if len(result.generations) != 1 or len(result.generations[0]) != 1:
                raise ValueError('评测接口没有返回恰好一个回答')
            return result.generations[0][0]

        def generate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
            client = self._client(n)
            generations = [self._one(client.generate_prompt([prompt], stop=stop, callbacks=callbacks)) for _ in range(n)]
            return LLMResult(generations=[generations])

        async def agenerate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
            client = self._client(n)
            # 同一指标的多次回答顺序请求，总并发仍由外层两题的上限控制。
            generations = []
            for _ in range(n):
                generations.append(self._one(await client.agenerate_prompt([prompt], stop=stop, callbacks=callbacks)))
            return LLMResult(generations=[generations])

    return SingleResponseLLM(chat_model)


def evaluate_run(cases, run, model, embedding_model=None, *, faithfulness_only=False, max_tokens=8192, reasoning_effort=None):
    # 固定接口版本，防止依赖升级后同名指标的实现悄悄变化。
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy
    from ragas.run_config import RunConfig
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    if version('ragas') != '0.3.3':
        raise ValueError('请使用 doc/evaluation/requirements.txt 固定的 Ragas 0.3.3 环境')
    if not os.environ.get('OPENAI_API_KEY'):
        raise ValueError('Ragas 需要显式配置 OPENAI_API_KEY；兼容服务可设置 OPENAI_BASE_URL')
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError('max_tokens 必须为正整数')
    score(cases, run)  # 先校验编号与预测格式，避免花费费用后才发现输入不合法。
    questions = {case['id']: case['question'] for case in cases}
    result = copy.deepcopy(run)
    dataset, selected = [], []
    for prediction in result['predictions']:
        if prediction.get('status') == 'failed':
            continue
        contexts = prediction.get('retrieved_contexts')
        if not isinstance(contexts, list) or not contexts or any(not isinstance(c, str) or not c.strip() for c in contexts):
            raise ValueError(f"{prediction['case_id']} 需要本次实际检索到的非空原文 retrieved_contexts")
        dataset.append({'user_input': questions[prediction['case_id']], 'response': prediction['response'], 'retrieved_contexts': contexts})
        selected.append(prediction)
    if not dataset:
        raise ValueError('没有可评测的预测')
    model_options = {'reasoning_effort': reasoning_effort} if reasoning_effort is not None else {}
    llm = _single_response_llm(ChatOpenAI(model=model, temperature=0, timeout=120, max_retries=1, max_tokens=max_tokens, **model_options))
    metrics = [Faithfulness(llm=llm)]
    if not faithfulness_only:
        if not embedding_model:
            raise ValueError('Answer Relevancy 需要 --embedding-model；只核查忠实度可用 --faithfulness-only')
        # 生成与向量模型可来自不同服务；文本按原样发送，避免兼容接口收到 OpenAI token 编号。
        embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
            model=embedding_model, max_retries=1, chunk_size=8, check_embedding_ctx_length=False,
            api_key=os.environ.get('EMBEDDING_API_KEY') or os.environ['OPENAI_API_KEY'],
            base_url=os.environ.get('EMBEDDING_BASE_URL') or os.environ.get('OPENAI_BASE_URL'),
        ))
        metrics.append(ResponseRelevancy(llm=llm, embeddings=embeddings))
    endpoint = urlsplit(os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1'))
    safe_endpoint = urlunsplit((endpoint.scheme, endpoint.netloc.rsplit('@', 1)[-1], endpoint.path, '', ''))
    evaluator = {'framework': 'ragas', 'version': version('ragas'), 'model': model,
                 'embedding_model': embedding_model, 'temperature': 0,
                 'max_tokens': max_tokens, 'reasoning_effort': reasoning_effort,
                 'completion_strategy': 'independent_single_response_requests',
                 'answer_relevancy_strictness': None if faithfulness_only else 3,
                 'endpoint': safe_endpoint,
                 'prompt_hash': fingerprint([{name: prompt.to_string() for name, prompt in metric.get_prompts().items()} for metric in metrics])}
    if not faithfulness_only:
        embedding_endpoint = urlsplit(os.environ.get('EMBEDDING_BASE_URL') or os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1'))
        evaluator['embedding_endpoint'] = urlunsplit((embedding_endpoint.scheme, embedding_endpoint.netloc.rsplit('@', 1)[-1], embedding_endpoint.path, '', ''))
    evaluated = evaluate(EvaluationDataset.from_list(dataset), metrics=metrics,
                         # 一个指标包含提取事实及逐条判断两次调用，总限时应覆盖两次请求。
                         run_config=RunConfig(timeout=300, max_retries=1, max_workers=2),
                         raise_exceptions=False, show_progress=True)
    for prediction, values in zip(selected, evaluated.scores, strict=True):
        prediction['ragas'] = {key: float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None
                               for key, value in values.items()}
    result['profile']['evaluator'] = evaluator
    result['evaluation_note'] = 'Ragas 分数是模型判断；运行费用不含在原工作流 usage 中。失败指标保留 null。'
    return result
