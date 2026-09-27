from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.llm import ModelConfig, SystemConfig, make_provider
from src.llm.base import normalize_token_usage
from src.repositories.chroma.read_vector_store import EmbeddingConnection, embedding_collection_name, embedding_identity
from src.repositories.chroma.vector_store import make_chroma_store
from src.utils.read_utils.cache import safe_cache_name
from src.utils.read_utils.chunkers import TextChunk, load_chunks_file


JsonObject = dict[str, Any]


def paper_scope(read_results: list[JsonObject], paper_ids: list[str] | None = None) -> set[str]:
    """从会话阅读记录确定可查论文，再与模型请求的论文编号取交集。"""

    requested = {str(value).strip().casefold() for value in paper_ids or [] if str(value).strip()}
    allowed: set[str] = set()
    for result in read_results:
        paper = result.get("paper")
        if not isinstance(paper, dict):
            continue
        aliases = {str(paper.get(key) or "").strip() for key in ("id", "paperId", "doi")}
        aliases.discard("")
        if requested and not requested.intersection(value.casefold() for value in aliases):
            continue
        allowed.update(aliases)
    return allowed


def load_scoped_chunks(
    cache_dir: Path,
    read_results: list[JsonObject],
    paper_ids: list[str] | None = None,
) -> list[TextChunk]:
    """只读会话内论文的缓存；缓存里有其他论文，也不会被加入候选资料。"""

    allowed = paper_scope(read_results, paper_ids)
    if not allowed or not cache_dir.is_dir():
        return []
    allowed_keys = {value.casefold() for value in allowed}
    directories = {cache_dir / safe_cache_name(value) for value in allowed}
    # 中文说明：缓存目录可能按来源内部编号命名，而会话保存的是 DOI。
    # 此时只查 metadata.json 的编号对应关系，匹配以后才读正文。
    for directory in cache_dir.iterdir():
        if not directory.is_dir() or directory in directories:
            continue
        try:
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            # 中文说明：某个旧缓存的元数据可能已损坏。它无法用于匹配当前论文，
            # 直接跳过即可，不能让一份无关缓存中断整场事实矩阵和后续写作。
            continue
        if not isinstance(metadata, dict):
            continue
        paper = metadata.get("paper") if isinstance(metadata.get("paper"), dict) else {}
        aliases = [metadata.get("paperId"), paper.get("id"), paper.get("paperId"), paper.get("doi")]
        if any(str(value or "").casefold() in allowed_keys for value in aliases):
            directories.add(directory)
    chunks: dict[str, TextChunk] = {}
    for directory in sorted(directories):
        # 中文说明：只接受该论文自身的切片，避免目录名与文件内容不一致时串入其他论文。
        for chunk in load_chunks_file(directory / "chunk.json"):
            if chunk.paperId.casefold() in allowed_keys:
                chunks[chunk.chunk_id] = chunk
    return list(chunks.values())


def _tokens(text: str) -> list[str]:
    """保留算法名、版本号；中文以相邻字组匹配，不额外下载分词模型。"""

    tokens: list[str] = []
    for term in re.findall(r"[a-z0-9]+(?:[.\-+_][a-z0-9]+)*|[\u3400-\u9fff]+", text.casefold()):
        if re.fullmatch(r"[\u3400-\u9fff]+", term):
            tokens.extend(term[index : index + 2] for index in range(max(1, len(term) - 1)))
        else:
            tokens.append(term)
            if re.search(r"[.\-+_]", term):
                tokens.extend(part for part in re.split(r"[.\-+_]", term) if part)
    return tokens


def bm25_rank(query: str, chunks: list[TextChunk], limit: int) -> list[tuple[str, float]]:
    """按 BM25 计算关键词相关性；没有共同关键词的片段不会冒充命中。"""

    terms = set(_tokens(query))
    documents = [Counter(_tokens(chunk.content)) for chunk in chunks]
    if not terms or not documents:
        return []
    lengths = [sum(document.values()) for document in documents]
    average_length = sum(lengths) / len(lengths) or 1.0
    document_frequency = {term: sum(term in document for document in documents) for term in terms}
    ranked: list[tuple[str, float]] = []
    for chunk, counts, length in zip(chunks, documents, lengths, strict=True):
        score = 0.0
        for term in terms:
            frequency = counts[term]
            if not frequency:
                continue
            inverse_frequency = math.log(1 + (len(documents) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            score += inverse_frequency * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * length / average_length))
        if score > 0:
            ranked.append((chunk.chunk_id, score))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))[:limit]


def reciprocal_rank_fusion(rankings: list[list[tuple[str, float]]], rank_constant: int) -> list[tuple[str, float]]:
    """只合并各路的名次，避免把关键词分数和向量距离直接相加。"""

    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (chunk_id, _) in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (rank_constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


async def async_search_chunks(
    query: str,
    *,
    read_results: list[JsonObject],
    cache_dir: str | Path,
    paper_ids: list[str] | None = None,
    top_k: int | None = None,
    system_config: SystemConfig | None = None,
    model_config_path: str | Path = "config/model.json",
) -> JsonObject:
    """查询当前会话的真实正文，返回原文、父级上下文和实际执行情况。"""

    config = system_config or SystemConfig.load()
    options = config.retrieval
    limit = min(options.top_k, max(1, top_k)) if isinstance(top_k, int) else options.top_k
    query = query.strip()
    diagnostics: JsonObject = {
        "bm25": {"status": "not_run"},
        "dense": {"status": "not_run"},
        "reranker": {"status": "disabled" if not options.reranker_enabled else "not_run"},
        "fusion": "reciprocal_rank_fusion",
    }
    if not query:
        return {"status": "invalid_query", "query": query, "chunks": [], "diagnostics": diagnostics}
    allowed = paper_scope(read_results, paper_ids)
    diagnostics["allowed_paper_count"] = len({str(item.get("paper", {}).get("paperId") or item.get("paper", {}).get("id") or "") for item in read_results if isinstance(item.get("paper"), dict)})
    if not allowed:
        return {"status": "no_allowed_papers", "query": query, "chunks": [], "diagnostics": diagnostics}
    chunks = await asyncio.to_thread(load_scoped_chunks, Path(cache_dir), read_results, paper_ids)
    diagnostics["corpus_chunk_count"] = len(chunks)
    if not chunks:
        return {"status": "no_full_text", "query": query, "chunks": [], "diagnostics": diagnostics}
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    sparse = await asyncio.to_thread(bm25_rank, query, chunks, options.candidate_k)
    diagnostics["bm25"] = {"status": "executed", "hit_count": len(sparse)}
    dense, dense_diagnostics = await _dense_rank(query, chunks, config, Path(model_config_path))
    diagnostics["dense"] = dense_diagnostics
    fused = reciprocal_rank_fusion([sparse, dense], options.rrf_k)[: options.candidate_k]
    rerank_scores: dict[str, float] = {}
    if options.reranker_enabled and fused:
        try:
            scores = await asyncio.to_thread(
                _cross_encoder_scores, options.reranker_model, options.reranker_local_files_only,
                query, [by_id[chunk_id].content for chunk_id, _ in fused],
            )
            rerank_scores = dict(zip((chunk_id for chunk_id, _ in fused), scores, strict=True))
            fused.sort(key=lambda item: (-rerank_scores[item[0]], -item[1], item[0]))
            diagnostics["reranker"] = {"status": "executed", "model": options.reranker_model, "candidate_count": len(scores)}
        except Exception as exc:
            diagnostics["reranker"] = {"status": "unavailable", "reason": str(exc)}
    sparse_scores, dense_scores = dict(sparse), dict(dense)
    matches: list[JsonObject] = []
    for chunk_id, score in fused[:limit]:
        chunk = by_id[chunk_id]
        matches.append({
            **chunk.to_dict(),
            "score": score,
            "scores": {"bm25": sparse_scores.get(chunk_id), "dense_distance": dense_scores.get(chunk_id), "rrf": score, "reranker": rerank_scores.get(chunk_id)},
            "matched_by": [name for name, scores in (("bm25", sparse_scores), ("dense", dense_scores)) if chunk_id in scores],
        })
    diagnostics["mode"] = "hybrid" if dense_diagnostics["status"] == "executed" else "bm25_only"
    return {"status": "ok" if matches else "no_matches", "query": query, "chunks": matches, "diagnostics": diagnostics}


async def _dense_rank(query: str, chunks: list[TextChunk], config: SystemConfig, model_path: Path) -> tuple[list[tuple[str, float]], JsonObject]:
    """只在确实读到同模型索引并调用 embedding 后，才报告向量检索已执行。"""

    if not config.retrieval.dense_enabled:
        return [], {"status": "disabled", "reason": "配置已关闭向量检索"}
    snapshot = None
    try:
        model_config = ModelConfig.from_dict(json.loads(model_path.read_text(encoding="utf-8")), config)
        profile = model_config.resolve_embedding_profile()
        snapshot = make_provider(model_config, embedding_profile_name=model_config.default_embedding_profile)
        connection = EmbeddingConnection(snapshot.provider, profile.model_name, profile.dimensions, int(profile.batch_size or 32))
        collection = embedding_collection_name(config.read.vector_store_collection, connection)
        where = {"$and": [{"paperId": {"$in": sorted({chunk.paperId for chunk in chunks})}}, {"embedding_identity": embedding_identity(connection)}]}
        indexed_ids = await asyncio.to_thread(_available_index_ids, config.read.vector_store_path, collection, where, chunks)
        if not indexed_ids:
            return [], {"status": "unavailable", "reason": "当前会话没有与现用模型及正文版本匹配的向量索引，需要重新阅读建立索引"}
        response = await asyncio.wait_for(snapshot.provider.embed([query], dimensions=profile.dimensions), timeout=max(1, config.read.download_timeout_seconds))
        usage = normalize_token_usage(response.usage)
        if not response.ok:
            return [], {"status": "unavailable", "reason": response.content or response.error_type or "查询向量生成失败", "usage": usage}
        if len(response.embeddings) != 1 or not response.embeddings[0] or not all(
            isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
            for value in response.embeddings[0]
        ):
            raise ValueError("查询 embedding 没有返回一个有效向量")
        vector = response.embeddings[0]
        if profile.dimensions is not None and len(vector) != profile.dimensions:
            raise ValueError("查询向量维度与配置不一致")
        results = await asyncio.to_thread(_query_index, config.read.vector_store_path, collection, where, vector, config.retrieval.candidate_k, indexed_ids)
        return results, {"status": "executed", "model": profile.model_name, "hit_count": len(results), "usage": usage}
    except Exception as exc:
        return [], {"status": "unavailable", "reason": str(exc) or type(exc).__name__}
    finally:
        if snapshot is not None:
            await snapshot.aclose()


def _available_index_ids(path: str, collection: str, where: JsonObject, chunks: list[TextChunk]) -> set[str]:
    """向量对应的正文必须和当前缓存一致；过期片段不能用来支撑新报告。"""

    if not Path(path).is_dir():
        return set()
    store = make_chroma_store(path, collection)
    try:
        hashes = {chunk.chunk_id: hashlib.sha256(chunk.content.encode("utf-8")).hexdigest() for chunk in chunks}
        return {item.id for item in store.get_by_filter(where) if item.id in hashes and item.metadata.get("content_hash") == hashes[item.id]}
    finally:
        store.close()


def _query_index(path: str, collection: str, where: JsonObject, vector: list[float], limit: int, allowed_ids: set[str]) -> list[tuple[str, float]]:
    """先限定论文范围及有效切片编号，再从这些证据中选择最相关的内容。"""

    store = make_chroma_store(path, collection)
    try:
        results = store.query_by_embedding(vector, top_k=limit, where=where, ids=sorted(allowed_ids))
        return [(item.id, float(item.distance)) for item in results if item.id in allowed_ids and item.distance is not None and math.isfinite(item.distance)]
    finally:
        store.close()


@lru_cache(maxsize=1)
def _load_cross_encoder(model_name: str, local_files_only: bool):
    """首次明确开启重排时才加载模型，默认只使用机器上已有的模型文件。"""

    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, local_files_only=local_files_only, trust_remote_code=False)


def _cross_encoder_scores(model_name: str, local_files_only: bool, query: str, contents: list[str]) -> list[float]:
    """用真实的查询和原文成对评分；加载失败时由上层明确报告未完成重排。"""

    model = _load_cross_encoder(model_name, local_files_only)
    scores = model.predict([(query, content) for content in contents], batch_size=8, show_progress_bar=False)
    values = [float(score) for score in scores]
    if len(values) != len(contents) or not all(math.isfinite(value) for value in values):
        raise ValueError("重排模型必须为每个候选片段返回一个有限数值")
    return values
