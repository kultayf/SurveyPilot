import asyncio
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from xml.etree import ElementTree as ET

from src.paper_retrieval.connectors.arxiv import ArxivPaperConnector
from src.paper_retrieval.connectors.base import PaperSearchConnector
from src.paper_retrieval.models import PaperDocument, SearchRequest
from src.paper_retrieval.service import PaperSearchService


class _FakeConnector(PaperSearchConnector):
    """测试用 connector，用来稳定验证编排层行为。"""

    def __init__(self, source_name: str, items: list[PaperDocument]):
        self.source_name = source_name
        self.items = items
        self.seen_requests: list[SearchRequest] = []

    def search(self, request: SearchRequest) -> list[PaperDocument]:
        """记录请求并返回预设结果，避免测试依赖真实外部网络。"""

        self.seen_requests.append(request)
        return list(self.items)


class _AsyncFakeConnector(_FakeConnector):
    """带异步入口的测试 connector，用来验证新的异步搜索主链路。"""

    async def async_search(self, request: SearchRequest, *, client=None) -> list[PaperDocument]:
        """异步入口里继续记录请求，确保服务层真的走到了 async 接口。"""

        self.seen_requests.append(request)
        await asyncio.sleep(0)
        return list(self.items)


class PaperSearchServiceTest(unittest.TestCase):
    def test_single_source_search_returns_standardized_response(self):
        service = PaperSearchService(
            connectors={
                "openalex": _FakeConnector(
                    "openalex",
                    [
                        PaperDocument(
                            id="oa-1",
                            title="Graph Neural Networks",
                            authors=["Alice"],
                            year=2024,
                            source="openalex",
                        )
                    ],
                )
            }
        )

        response = service.search("graph neural networks", source="openalex", limit=5)

        self.assertEqual(response.sources_used, ["openalex"])
        self.assertEqual(response.source_results["openalex"], 1)
        self.assertEqual(response.total, 1)
        self.assertEqual(response.papers[0].title, "Graph Neural Networks")

    def test_multi_source_search_deduplicates_by_doi(self):
        duplicate_a = PaperDocument(
            id="a1",
            title="Shared Paper",
            authors=["Alice"],
            doi="10.1000/shared",
            source="openalex",
        )
        duplicate_b = PaperDocument(
            id="b1",
            title="Shared Paper",
            authors=["Bob"],
            doi="10.1000/shared",
            source="semantic_scholar",
        )
        unique = PaperDocument(
            id="c1",
            title="Unique Paper",
            authors=["Carol"],
            source="arxiv",
        )
        service = PaperSearchService(
            connectors={
                "openalex": _FakeConnector("openalex", [duplicate_a]),
                "semantic_scholar": _FakeConnector("semantic_scholar", [duplicate_b]),
                "arxiv": _FakeConnector("arxiv", [unique]),
            }
        )

        response = service.search("shared query", limit=5)

        self.assertEqual(response.total, 2)
        self.assertEqual(sorted(response.source_results.keys()), ["arxiv", "openalex", "semantic_scholar"])

    def test_multi_source_search_uses_full_limit_for_each_connector(self):
        service = PaperSearchService(
            connectors={
                "openalex": _FakeConnector("openalex", []),
                "arxiv": _FakeConnector("arxiv", []),
            }
        )

        response = service.search("shared query", limit=5, truncate=False)

        self.assertEqual(response.total, 0)
        self.assertEqual(service._connectors["openalex"].seen_requests[0].limit, 5)
        self.assertEqual(service._connectors["arxiv"].seen_requests[0].limit, 5)

    def test_invalid_source_returns_error(self):
        service = PaperSearchService(connectors={})

        response = service.search("anything", source="missing", limit=3)

        self.assertIn("sources", response.errors)
        self.assertEqual(response.total, 0)

    def test_async_multi_source_search_uses_requested_sources(self):
        service = PaperSearchService(
            connectors={
                "openalex": _AsyncFakeConnector(
                    "openalex",
                    [PaperDocument(id="oa-1", title="OpenAlex Paper", authors=["Alice"], source="openalex")],
                ),
                "arxiv": _AsyncFakeConnector(
                    "arxiv",
                    [PaperDocument(id="ax-1", title="arXiv Paper", authors=["Bob"], source="arxiv")],
                ),
                "semantic_scholar": _AsyncFakeConnector(
                    "semantic_scholar",
                    [PaperDocument(id="ss-1", title="Semantic Paper", authors=["Carol"], source="semantic_scholar")],
                ),
            }
        )

        response = asyncio.run(
            service.async_search(
                topic="agent search",
                keywords=["multi-agent"],
                sources=["openalex", "arxiv"],
                limit=5,
                truncate=False,
            )
        )

        self.assertEqual(response.sources_used, ["openalex", "arxiv"])
        self.assertEqual(response.total, 2)
        self.assertEqual(service._connectors["openalex"].seen_requests[0].limit, 5)
        self.assertEqual(service._connectors["arxiv"].seen_requests[0].limit, 5)
        self.assertEqual(service._connectors["semantic_scholar"].seen_requests, [])


class ArxivConnectorTest(unittest.TestCase):
    def test_doi_link_keeps_the_full_identifier(self):
        """Atom DOI 链接不能只取最后一个路径片段。"""

        entry = ET.fromstring(
            '<entry xmlns="http://www.w3.org/2005/Atom">'
            '<id>https://arxiv.org/abs/2103.10697v2</id>'
            '<title>ConViT: Improving Vision Transformers</title>'
            '<summary>Vision transformer study.</summary>'
            '<link title="doi" href="https://doi.org/10.1088/1742-5468/ac9830" />'
            '<link title="pdf" type="application/pdf" href="https://arxiv.org/pdf/2103.10697v2" />'
            '</entry>'
        )
        connector = ArxivPaperConnector()
        try:
            paper = connector.normalize_paper(entry)
        finally:
            connector.client.close()

        self.assertIsNotNone(paper)
        self.assertEqual(paper.paperId, "10.1088/1742-5468/ac9830")
        self.assertEqual(paper.metadata["arxiv_id"], "2103.10697")

    def test_curl_fallback_only_handles_second_406(self):
        """两层 Python 请求都遇 406 才尝试受限 curl，429 不绕过。"""

        connector = ArxivPaperConnector()
        params = {"id_list": "2103.10697", "start": 0, "max_results": 1}
        rejected = HTTPError("https://export.arxiv.org/api/query", 406, "Not Acceptable", None, None)
        limited = HTTPError("https://export.arxiv.org/api/query", 429, "Too Many Requests", None, None)
        try:
            with patch.object(connector, "_request_with_stdlib", side_effect=rejected), patch.object(
                connector, "_request_with_curl", return_value="<feed/>"
            ) as curl_request:
                self.assertEqual(connector._request_after_406(params), "<feed/>")
                curl_request.assert_called_once_with(params)
            with patch.object(connector, "_request_with_stdlib", side_effect=limited), patch.object(
                connector, "_request_with_curl"
            ) as curl_request:
                with self.assertRaises(HTTPError):
                    connector._request_after_406(params)
                curl_request.assert_not_called()
        finally:
            connector.client.close()

    def test_explicit_arxiv_id_batch_and_subtopic_priority(self):
        """多个原论文编号可一次精确读取，子主题编号优先于整段主题。"""

        connector = ArxivPaperConnector()
        query = "比较 arXiv:2103.10697 与 arXiv:2103.14030 的原论文"
        try:
            batch = connector._build_request_params(SearchRequest(query=query, limit=5))
            focused = connector._build_request_params(
                SearchRequest(query=query, keyword_expression="arXiv:2103.14030", limit=5)
            )
        finally:
            connector.client.close()

        self.assertEqual(batch["id_list"], "2103.10697,2103.14030")
        self.assertEqual(batch["max_results"], 2)
        self.assertEqual(focused["id_list"], "2103.14030")
        self.assertEqual(focused["max_results"], 1)


if __name__ == "__main__":
    unittest.main()
