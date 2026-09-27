"""以少量种子论文限额展开引文关系，保存发现路径，不宣称找全领域文献。"""
from __future__ import annotations

import asyncio
import re
from urllib.parse import quote

import httpx

from src.agents.base import AgentContext, AgentSpec, BaseAgent
from src.paper_retrieval.connectors.semantic_scholar import SemanticScholarPaperConnector, wait_for_semantic_scholar_slot
from src.paper_retrieval.models import PaperDocument, SearchRequest


def semantic_seed_id(paper: PaperDocument) -> str | None:
    """优先使用来源明确的 S2 编号，其次 DOI/arXiv，不把其他来源编号冒充 S2。"""
    semantic_id = str(paper.metadata.get("semantic_scholar_id") or "").strip()
    if semantic_id and re.fullmatch(r"[a-fA-F0-9]{40}", semantic_id):
        return semantic_id
    doi = (paper.doi or "").strip()
    if doi:
        doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", "", doi, flags=re.I)
        return "DOI:" + doi
    arxiv = str(paper.metadata.get("arxiv_id") or "").strip()
    if not arxiv and paper.source == "arxiv":
        arxiv = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", str(paper.paperId or paper.id))
    if arxiv:
        return "ARXIV:" + re.sub(r"v\d+$", "", arxiv.removesuffix(".pdf"))
    return None


class CitationSnowballAgent(BaseAgent):
    spec = AgentSpec(name="citation_snowball_agent", role="search", description="限额发现种子论文引用及被引关系。")

    def __init__(self, context: AgentContext):
        context.spec = self.spec
        super().__init__(context)

    def _run(self, state):
        raise NotImplementedError("请使用 expand 异步展开引文")

    async def expand(self, seeds: list[PaperDocument], *, constraints: dict, api_key: str | None = None,
                     client: httpx.AsyncClient | None = None, cancellation=None, progress=None) -> dict:
        """最多两层、每方向三页及 24 次请求；达到上限明确标记截断。"""
        depth_limit = citation_limit(constraints, 'citation_depth', 1, 2)
        page_limit = citation_limit(constraints, 'citation_pages', 1, 3)
        connector = SemanticScholarPaperConnector(api_key=api_key)
        session = client or httpx.AsyncClient(timeout=20)
        papers, links, errors, skipped, resolved = {}, [], [], [], []
        calls, truncated = 0, False
        request = SearchRequest(year_from=constraints.get('year_from'), year_to=constraints.get('year_to'),
                                excluded_terms=constraints.get('excluded_terms') or [])
        async def fetch(identifier, suffix='', offset=0):
            nonlocal calls
            params = {'fields': connector._fields}
            if suffix:
                params.update(limit=20, offset=offset)
            for attempt in range(2):
                if cancellation:
                    cancellation.raise_if_requested()
                await asyncio.to_thread(wait_for_semantic_scholar_slot)
                calls += 1
                response = await session.get(f"https://api.semanticscholar.org/graph/v1/paper/{quote(identifier, safe='')}{suffix}",
                                             params=params, headers=connector.headers, timeout=20)
                if response.status_code == 429 and attempt == 0 and calls < 24:
                    try:
                        retry_after = float(response.headers.get('Retry-After') or 2)
                    except ValueError:
                        retry_after = 2
                    # 只等待服务端可接受的短退避；长配额限制直接向上报告 429。
                    if retry_after <= 30:
                        await asyncio.sleep(max(1.1, retry_after))
                        continue
                response.raise_for_status()
                return response.json()
            raise RuntimeError('引文请求重试未返回响应')
        def snapshot():
            # 共被引计数只来自已取回的边：同一篇论文同时引用两个目标，算一次共同引用。
            from collections import defaultdict
            from itertools import combinations
            targets = defaultdict(set)
            for edge in links:
                targets[edge['source']].add(edge['target'])
            pairs = defaultdict(set)
            for citing, cited in targets.items():
                for pair in combinations(sorted(cited), 2):
                    pairs[pair].add(citing)
            co_citations = [{'paper_ids': list(pair), 'citing_paper_count': len(citing), 'citing_paper_ids': sorted(citing)}
                            for pair, citing in sorted(pairs.items(), key=lambda item: (-len(item[1]), item[0]))[:20]]
            return {'papers': list(papers.values()), 'seed_papers': resolved, 'links': links, 'errors': errors,
                    'skipped': skipped, 'calls': calls, 'truncated': truncated, 'co_citations': co_citations,
                    'depth': depth_limit, 'pages_per_direction': page_limit, 'request_limit': 24}
        try:
            queue = [(seed, 1) for seed in seeds[:3]]
            visited, queued, edge_keys = set(), set(), set()
            while queue:
                seed, depth = queue.pop(0)
                identifier = semantic_seed_id(seed)
                if not identifier:
                    skipped.append({'paperId': seed.paperId or seed.id, 'reason': '种子没有可识别的编号'})
                    continue
                if identifier.casefold() in visited:
                    continue
                if calls >= 24:
                    truncated = True
                    break
                visited.add(identifier.casefold())
                direction = 'metadata'
                try:
                    # 中文说明：用户已提供 DOI、arXiv 或 S2 标识时，该标识本身足以作为
                    # 引文边端点。先请求一次元数据既不增加可审计信息，也会额外消耗 S2 的
                    # 共享 1 req/s 配额；直接从 references/citations 两个端点展开即可。
                    if depth == 1:
                        resolved.append(seed)
                    discovered = []
                    for direction, key in (('references', 'citedPaper'), ('citations', 'citingPaper')):
                        offset, offsets = 0, set()
                        for page in range(page_limit):
                            if calls >= 24:
                                truncated = True
                                break
                            offsets.add(offset)
                            payload = await fetch(identifier, '/' + direction, offset)
                            if not isinstance(payload, dict) or not isinstance(payload.get('data'), list):
                                raise ValueError('引文接口没有返回 data 列表')
                            for item in payload['data'][:20]:
                                candidate = connector.normalize_paper(item.get(key)) if isinstance(item, dict) else None
                                if candidate is None or not candidate.paperId:
                                    continue
                                if not connector._within_year_range(candidate, request) or connector._contains_excluded_terms(candidate, request.excluded_terms):
                                    continue
                                paper_id = candidate.paperId
                                papers.setdefault(paper_id, candidate)
                                discovered.append(candidate)
                                source, target = ((seed.paperId or seed.id, paper_id) if direction == 'references' else (paper_id, seed.paperId or seed.id))
                                if source == target or (source, target) in edge_keys:
                                    continue
                                edge_keys.add((source, target))
                                links.append({'source': source, 'target': target, 'seed': seed.paperId or seed.id,
                                              'direction': direction, 'depth': depth, 'discovered_paper_id': paper_id})
                            if progress:
                                progress(f'已完成 {calls}/24 次引文查询')
                            next_offset = payload.get('next')
                            if next_offset is None:
                                break
                            if page + 1 == page_limit or type(next_offset) is not int or next_offset <= offset or next_offset in offsets:
                                truncated = True
                                break
                            offset = next_offset
                    if depth < depth_limit:
                        # 每个展开节点最多再选择前三篇，避免二层扩展出现指数增长。
                        # 同一篇论文可能同时出现在两个方向，先去重再分配三篇名额。
                        candidates_by_id = {}
                        for paper in discovered:
                            identity = semantic_seed_id(paper)
                            if identity and identity.casefold() not in visited | queued:
                                candidates_by_id.setdefault(identity.casefold(), paper)
                        candidates = list(candidates_by_id.values())
                        truncated = truncated or len(candidates) > 3
                        for candidate in candidates[:3]:
                            queued.add(semantic_seed_id(candidate).casefold())
                            queue.append((candidate, depth + 1))
                except httpx.HTTPStatusError as exc:
                    errors.append({'seed': seed.paperId or seed.id, 'direction': direction, 'status': exc.response.status_code})
                    truncated = True
                    if exc.response.status_code in {401, 403, 429}:
                        break
                except (httpx.HTTPError, ValueError, TypeError) as exc:
                    errors.append({'seed': seed.paperId or seed.id, 'direction': direction, 'reason': type(exc).__name__})
                    truncated = True
        finally:
            connector.client.close()
            if client is None:
                await session.aclose()
        return snapshot()


def citation_limit(constraints, key, default, maximum):
    value = constraints.get(key, default)
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f'{key} 必须是 1–{maximum} 的整数')
    return value


def parse_seed_identifiers(values):
    """只接受学术编号，网址仅提取 DOI/arXiv 编号，不访问用户提交的任意网址。"""
    if not isinstance(values, list) or len(values) > 3:
        raise ValueError('请提供最多 3 个种子编号')
    result, seen = [], set()
    for value in values:
        if not isinstance(value, str):
            raise ValueError('种子编号必须为文字')
        value = value.strip()
        value = re.sub(r'^https?://(?:dx\.)?doi\.org/', 'DOI:', value, flags=re.I)
        value = re.sub(r'^https?://arxiv\.org/(?:abs|pdf)/', 'ARXIV:', value, flags=re.I).removesuffix('.pdf')
        doi = re.sub(r'^doi:', '', value, flags=re.I)
        arxiv = re.sub(r'^arxiv:', '', value, flags=re.I)
        meta = {'user_supplied_seed': True}
        if re.fullmatch(r'10\.\d{4,9}/\S+', doi):
            paper = PaperDocument(id='DOI:' + doi, title='DOI:' + doi, doi=doi, metadata=meta)
        elif re.fullmatch(r'(?:\d{4}\.\d{4,5}|[a-z-]+/\d{7})(?:v\d+)?', arxiv, flags=re.I):
            paper = PaperDocument(id='ARXIV:' + arxiv, title='ARXIV:' + arxiv, metadata={**meta, 'arxiv_id': arxiv})
        elif re.fullmatch(r'[a-fA-F0-9]{40}', value):
            paper = PaperDocument(id=value, title=value, metadata={**meta, 'semantic_scholar_id': value})
        else:
            raise ValueError('种子需要 DOI、arXiv 编号或 40 位 Semantic Scholar 编号')
        identity = semantic_seed_id(paper).casefold()
        if identity not in seen:
            seen.add(identity); result.append(paper)
    return result
