from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
from datetime import datetime
from urllib.error import HTTPError
from urllib.parse import unquote, urlencode, urlsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

import httpx

from src.utils import get_logger

from ..models import PaperDocument, SearchRequest
from .base import PaperSearchConnector


logger = get_logger(__name__)


class ArxivPaperConnector(PaperSearchConnector):
    """arXiv connector。

    这里负责把结构化意图拼成 arXiv Atom API 可接受的查询表达式，
    具体的检索语句组合规则不再暴露给上层 Agent。
    """

    source_name = "arxiv"
    _endpoint = "https://export.arxiv.org/api/query"
    _atom_ns = {"atom": "http://www.w3.org/2005/Atom"}

    def __init__(self, client: httpx.Client | None = None):
        """初始化 HTTP 客户端。"""

        self.headers = {
            "User-Agent": "survey-pilot/0.1 paper-retrieval",
            "Accept": "application/atom+xml, application/xml;q=0.9, */*;q=0.8",
        }
        self.client = client or httpx.Client(
            timeout=20.0,
            headers=self.headers,
        )

    def search(self, request: SearchRequest) -> list[PaperDocument]:
        """执行 arXiv 检索，并在 connector 内完成查询拼装。"""

        params = self._build_request_params(request)
        try:
            response = self.client.get(self._endpoint, params=params)
            response.raise_for_status()
            response_text = response.text
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 406:
                raise
            # 中文说明：2026-09-21 的真实验收中，同一 search_query 经 httpx 会被
            # arXiv/CDN 返回 406，而 id_list 和 Python 标准库请求都能成功。这里仅对
            # 这个已复现的状态重试一次；401、403、429 等状态必须原样交给上层处理。
            logger.warning("arXiv 关键词请求返回 406，改用标准库重试一次", extra={"http_status": 406})
            response_text = self._request_after_406(params)
        except httpx.ConnectError:
            # 中文说明：少数运行环境中 httpx 会在 DNS 连接阶段失败，而同一个公开 Atom
            # 地址可由标准库正常访问。这里只重放一次同一个 URL；鉴权和限流错误不走这里。
            logger.warning("arXiv httpx 连接失败，改用标准库重试一次", extra={"error_kind": "connect_error"})
            response_text = self._request_with_stdlib(params)
        return self._parse_response_text(response_text, request)

    async def async_search(
        self,
        request: SearchRequest,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> list[PaperDocument]:
        """异步执行 arXiv 检索，避免在异步编排里阻塞事件循环。"""

        params = self._build_request_params(request)
        resolved_client = client or httpx.AsyncClient(timeout=20.0)
        owns_client = client is None
        try:
            try:
                response = await resolved_client.get(
                    self._endpoint,
                    params=params,
                    headers=self.headers,
                    timeout=20.0,
                )
            except httpx.ConnectError:
                # 中文说明：连接还没有建立就失败时没有响应对象可检查状态码，直接把
                # 同一个固定 URL 交给标准库重放一次，后续解析规则保持不变。
                logger.warning("arXiv httpx 连接失败，改用标准库重试一次", extra={"error_kind": "connect_error"})
                response_text = await asyncio.to_thread(self._request_with_stdlib, params)
            else:
                try:
                    response.raise_for_status()
                    response_text = response.text
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code != 406:
                        raise
                    logger.warning("arXiv 关键词请求返回 406，改用标准库重试一次", extra={"http_status": 406})
                    # 标准库请求会阻塞当前线程，所以异步入口必须把它放到工作线程中执行。
                    response_text = await asyncio.to_thread(self._request_after_406, params)
        finally:
            if owns_client:
                await resolved_client.aclose()
        return self._parse_response_text(response_text, request)

    def _build_request_params(self, request: SearchRequest) -> dict[str, str | int]:
        """集中生成请求参数，确保正常请求与 406 回退使用完全相同的查询。"""

        arxiv_ids = self._explicit_arxiv_ids(request)
        if arxiv_ids:
            # 中文说明：用户明确给出 arXiv 编号时，使用 Atom API 的 id_list 精确读取。
            # 多个编号也按 API 原生逗号列表一次读取，避免重复关键词请求与 406。
            return {
                "id_list": ",".join(arxiv_ids[: request.limit]),
                "start": 0,
                "max_results": min(max(1, request.limit), len(arxiv_ids)),
            }
        return {
            "search_query": self._build_query(request),
            "start": 0,
            "max_results": max(1, request.limit),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

    def _explicit_arxiv_ids(self, request: SearchRequest) -> list[str]:
        """优先读取当前子主题的显式编号，再读取用户主题中的编号列表。"""

        values = (request.keyword_expression, *request.keywords, request.topic, request.query)
        prefixed = re.compile(r"arxiv(?:\.org/(?:abs|pdf)/)?\s*:?\s*(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)
        bare = re.compile(r"\d{4}\.\d{4,5}(?:v\d+)?", re.IGNORECASE)
        for value in values:
            text = str(value or "")
            matches = prefixed.findall(text)
            if matches:
                return list(dict.fromkeys(matches))
            if bare.fullmatch(text.strip()):
                return [text.strip()]
        return []

    def _request_with_stdlib(self, params: dict[str, str | int]) -> str:
        """使用 Python 标准库重放一次固定的 arXiv GET 请求。"""

        url = f"{self._endpoint}?{urlencode(params)}"
        fallback_request = Request(url, headers=self.headers, method="GET")
        # 中文说明：地址固定为当前 connector 的 arXiv 端点，参数、请求头和 20 秒超时
        # 都与主请求一致；这里只返回 XML 文本，不读取或记录失败响应的正文。
        with urlopen(fallback_request, timeout=20.0) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")

    def _request_after_406(self, params: dict[str, str | int]) -> str:
        """标准库同样被 406 拒绝时，再用固定端点做一次受限请求。"""

        try:
            return self._request_with_stdlib(params)
        except HTTPError as exc:
            if exc.code != 406:
                raise
            # 中文说明：真实请求中，同一 Atom URL 的 httpx/标准库都返回空体 406，
            # 而 curl 返回 200。只针对这个明确状态尝试一次；429、鉴权错误和其他
            # 连接问题不走 curl，失败后仍向上层保留原始 406。
            logger.warning("arXiv 标准库请求仍返回 406，使用受限 curl 重试一次", extra={"http_status": 406})
            try:
                return self._request_with_curl(params)
            except Exception as fallback_exc:
                logger.warning("arXiv curl 回退失败", extra={"error_kind": type(fallback_exc).__name__})
                raise exc from fallback_exc

    def _request_with_curl(self, params: dict[str, str | int]) -> str:
        """无 shell 调用系统 curl；端点、协议、时间和响应大小均受限。"""

        executable = shutil.which("curl")
        if executable is None:
            raise FileNotFoundError("curl unavailable")
        url = f"{self._endpoint}?{urlencode(params)}"
        result = subprocess.run(
            [
                executable, "--disable", "--silent", "--show-error", "--fail",
                "--max-time", "20", "--max-filesize", "5000000",
                "--proto", "=https", "--url", url,
                "--header", f"User-Agent: {self.headers['User-Agent']}",
                "--header", f"Accept: {self.headers['Accept']}",
            ],
            capture_output=True,
            check=True,
            timeout=25,
        )
        return result.stdout.decode("utf-8", errors="replace")

    def _parse_response_text(self, text: str, request: SearchRequest) -> list[PaperDocument]:
        """把 arXiv XML 响应解析成论文列表，同步和异步入口共用。"""

        root = ET.fromstring(text)
        papers: list[PaperDocument] = []
        for entry in root.findall("atom:entry", self._atom_ns):
            paper = self.normalize_paper(entry)
            if paper is None:
                continue
            if not self._within_year_range(paper, request):
                continue
            if self._contains_excluded_terms(paper, request.excluded_terms):
                continue
            papers.append(paper)
        return papers[: request.limit]

    def _build_query(self, request: SearchRequest) -> str:
        """把 topic / keywords / query 组合成 arXiv 的查询串。"""

        query_text = self._choose_query_text(request)
        if not query_text:
            return "all:*"
        return f"all:{self._escape_query(query_text)}"

    def _choose_query_text(self, request: SearchRequest) -> str:
        """优先使用上层原始 query，没有时再用 topic 和 keywords 兜底。"""

        if request.keyword_expression.strip():
            return request.keyword_expression.strip()
        if request.query.strip():
            return request.query.strip()
        parts: list[str] = []
        if request.topic.strip():
            parts.append(request.topic.strip())
        if request.keywords:
            parts.extend(request.keywords[:5])
        return " ".join(parts).strip()

    def _escape_query(self, query: str) -> str:
        """对 arXiv 查询串做最小化清理，避免空白字符导致语义不稳定。"""

        return " ".join(query.split())

    def normalize_paper(self, raw: object) -> PaperDocument | None:
        """把单个 Atom entry 解析成统一论文对象。"""

        if not isinstance(raw, ET.Element):
            return None
        entry = raw
        title = self._text(entry, "atom:title")
        if not title:
            return None
        raw_paper_id = self._text(entry, "atom:id").rsplit("/", 1)[-1]
        # 中文说明：Atom 链接常带 v1、v2 等版本尾巴。检索、缓存和人工标注都应使用
        # 稳定的论文编号，否则同一篇论文更新版本后会被误认成不同来源。
        paper_id = re.sub(r"v\d+$", "", raw_paper_id)
        authors = [author_name.text.strip() for author_name in entry.findall("atom:author/atom:name", self._atom_ns) if author_name.text]
        summary = self._text(entry, "atom:summary")
        published_text = self._text(entry, "atom:published")
        published_year = self._parse_year(published_text)
        pdf_url = ""
        doi = ""
        for link in entry.findall("atom:link", self._atom_ns):
            href = (link.attrib.get("href") or "").strip()
            title_attr = (link.attrib.get("title") or "").strip().lower()
            link_type = (link.attrib.get("type") or "").strip().lower()
            if link_type == "application/pdf" and href:
                pdf_url = href
            if title_attr == "doi" and href:
                # 中文说明：DOI 自身包含斜杠。只取 URL 最后一段会把
                # 10.1088/1742-5468/ac9830 错写成 ac9830，造成不可定位引用。
                candidate = unquote(urlsplit(href).path.lstrip("/"))
                if re.match(r"^10\.\d{4,9}/\S+$", candidate):
                    doi = candidate
        unique_id = doi or paper_id
        return PaperDocument(
            id=unique_id or title,
            paperId=unique_id,
            title=title,
            authors=authors,
            abstract=summary,
            year=published_year,
            venue="arXiv",
            url=self._text(entry, "atom:id"),
            pdf_url=pdf_url or None,
            doi=doi or None,
            source=self.source_name,
            publication_date=published_text,
            journal_conference="arXiv",
            language="en",
            metadata={"published": published_text, "arxiv_id": paper_id, "arxiv_versioned_id": raw_paper_id},
        )

    def _text(self, entry: ET.Element, path: str) -> str:
        """安全读取 XML 文本节点。"""

        node = entry.find(path, self._atom_ns)
        return node.text.strip() if node is not None and node.text else ""

    def _parse_year(self, value: str) -> int | None:
        """从发布时间中提取年份。"""

        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).year
        except ValueError:
            return None

    def _within_year_range(self, paper: PaperDocument, request: SearchRequest) -> bool:
        """按年份范围过滤结果。"""

        if paper.year is None:
            return True
        if request.year_from is not None and paper.year < request.year_from:
            return False
        if request.year_to is not None and paper.year > request.year_to:
            return False
        return True

    def _contains_excluded_terms(self, paper: PaperDocument, excluded_terms: list[str]) -> bool:
        """对标题和摘要做排除词过滤。"""

        haystack = f"{paper.title} {paper.abstract or ''}".lower()
        return any(term.strip().lower() in haystack for term in excluded_terms if term.strip())
