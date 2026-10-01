from __future__ import annotations

import asyncio
import html
import json
import re
import threading
from importlib.metadata import version, PackageNotFoundError
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from src.paper_retrieval.models import PaperDocument
from src.utils.read_utils.pdf_parsers import get_pdf_parser
from src.utils.read_utils.cache import text_sha256, write_text_atomic, file_sha256


_UNSUPPORTED_FULLTEXT_WARNING = "暂不支持该全文文件格式"
# 多篇论文可以并发下载，但同一进程里的版面解析按顺序执行，避免原生解析库同时操作。
_PDF_PARSE_LOCK = threading.Lock()


@dataclass(slots=True)
class MarkdownConversion:
    """保存全文转成 Markdown 后的文件位置、页数和提示信息。"""

    markdown_path: Path | None = None
    page_count: int | None = None
    warnings: list[str] = field(default_factory=list)


def convert_fulltext_to_markdown(
    paper: PaperDocument,
    *,
    source_path: Path,
    source_url: str | None,
    parser_name: str = "pymupdf4llm",
    docling_artifacts_path: str | Path | None = None,
) -> MarkdownConversion:
    """兼容旧同步流程的全文解析入口，主逻辑统一交给异步实现。"""

    # 中文注释：模块四整理之后，外层主入口改成了 async。
    # 这里保留一个同步壳，只是为了让还没改成 async 的阅读节点继续可用。
    awaitable = async_convert_fulltext_to_markdown(
        paper,
        source_path=source_path,
        source_url=source_url,
        parser_name=parser_name,
        docling_artifacts_path=docling_artifacts_path,
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # 中文注释：普通同步脚本或旧节点调用这里时，临时开一个事件循环把异步逻辑跑完。
        # 这样全文解析只保留一套真正的主逻辑，后面继续维护时不会出现两份代码越改越不一致。
        with asyncio.Runner() as runner:
            return runner.run(awaitable)
    # 中文注释：如果已经在 async 环境里，就不能再走这个同步兼容壳。
    # 这里直接报错，提醒调用方改成 await 异步入口，避免把事件循环卡死。
    if hasattr(awaitable, "close"):
        awaitable.close()
    raise RuntimeError("同步全文解析兼容接口不能在已有事件循环中调用，请改用 await async_convert_fulltext_to_markdown(...)")


async def async_convert_fulltext_to_markdown(
    paper: PaperDocument,
    *,
    source_path: Path,
    source_url: str | None,
    parser_name: str = "pymupdf4llm",
    docling_artifacts_path: str | Path | None = None,
) -> MarkdownConversion:
    """异步全文转换入口，供后续 async 阅读流程直接调用。"""

    return await asyncio.to_thread(
        _convert_with_cache, paper, source_path, source_url, parser_name, docling_artifacts_path,
    )


def _convert_with_cache(paper: PaperDocument, source_path: Path, source_url: str | None,
                        parser_name: str, docling_artifacts_path: str | Path | None) -> MarkdownConversion:
    """核对源文件、解析器版本和产物内容；改变任一项都重新解析。"""

    markdown_path = _markdown_output_path(source_path)
    manifest_path = source_path.parent / "parse_manifest.json"
    suffix = source_path.suffix.lower()
    try:
        parser_version = version(parser_name) if suffix == ".pdf" else "builtin-html-1"
    except PackageNotFoundError:
        parser_version = "not-installed"
    try:
        source_hash = file_sha256(source_path)
        # 中文说明：PDF 损坏页现在会带同页纯文本补充。旧缓存虽来自同一 PDF，
        # 内容却缺少这份自动补充，所以提高版本号让下次阅读重新解析和切块。
        expected = {"schema_version": 4, "source_hash": source_hash,
                    "parser": parser_name if suffix == ".pdf" else "html", "parser_version": parser_version,
                    "artifacts_path": str(docling_artifacts_path or ""), "paperId": paper.paperId or paper.id,
                    "source_url": source_url, "title": paper.title}
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")
            if (all(cached.get(k) == v for k, v in expected.items()) and markdown.strip()
                    and cached.get("markdown_hash") == text_sha256(markdown)):
                return MarkdownConversion(markdown_path, cached.get("page_count"), list(cached.get("warnings") or []))
        except (OSError, ValueError, AttributeError):
            pass
        if suffix == ".pdf":
            result = _convert_pdf_with_parser(paper, source_path, source_url, markdown_path,
                                              parser_name, docling_artifacts_path)
        elif suffix in {".html", ".htm"}:
            result = _convert_html(paper, source_path, source_url, markdown_path)
        else:
            return MarkdownConversion(warnings=[_UNSUPPORTED_FULLTEXT_WARNING])
        if result.markdown_path:
            payload = {**expected, "markdown_hash": text_sha256(markdown_path.read_text(encoding="utf-8")),
                       "page_count": result.page_count, "warnings": result.warnings}
            write_text_atomic(manifest_path, json.dumps(payload, ensure_ascii=False, indent=2))
        return result
    except Exception as exc:
        return MarkdownConversion(warnings=[f"全文解析不可用：{exc}"])


def _markdown_output_path(source_path: Path) -> Path:
    """统一计算 Markdown 结果文件路径，避免不同入口各自拼路径。"""

    return source_path.parent / "paper.md"


def _convert_html(paper: PaperDocument, source_path: Path, source_url: str | None, markdown_path: Path) -> MarkdownConversion:
    """提取普通 HTML 页面中的标题和段落，生成不含页码的 Markdown 文件。"""

    parser = _ArticleHtmlParser()
    try:
        parser.feed(source_path.read_text(encoding="utf-8", errors="replace"))
        parser.close()
    except OSError as exc:
        return MarkdownConversion(warnings=[f"HTML 正文读取失败：{exc}"])
    article = parser.to_markdown()
    if not article.strip():
        return MarkdownConversion(warnings=["HTML 页面没有可读取的正文"])
    write_text_atomic(markdown_path, _markdown_header(paper, source_url, None) + "\n\n" + article.strip() + "\n")
    return MarkdownConversion(markdown_path=markdown_path)


def _markdown_header(paper: PaperDocument, source_url: str | None, page_count: int | None) -> str:
    """生成 Markdown 开头的论文基本信息，避免正文和来源信息分散保存。"""

    header = {
        "paper_id": paper.id,
        "title": paper.title,
        "doi": paper.doi,
        "source_url": source_url or paper.url,
        "page_count": page_count,
    }
    return "---\n" + json.dumps(header, ensure_ascii=False, indent=2) + "\n---"


def _convert_pdf_with_parser(paper: PaperDocument, source_path: Path, source_url: str | None,
                             markdown_path: Path, parser_name: str,
                             docling_artifacts_path: str | Path | None) -> MarkdownConversion:
    """原样保留结构化 Markdown 和原页码；部分空页的提醒不覆盖其他有效内容。"""

    parser = get_pdf_parser(parser_name, docling_artifacts_path=docling_artifacts_path)
    with _PDF_PARSE_LOCK:
        parsed = parser.parse(source_path)
    if not any(page.text.strip() for page in parsed.pages):
        return MarkdownConversion(warnings=parsed.warnings or ["PDF 中没有可读取的正文，可能需要 OCR"])
    page_count = max(page.page_number for page in parsed.pages)
    body = [_markdown_header(paper, source_url, page_count)]
    for page in parsed.pages:
        body.extend([f"<!-- page: {page.page_number} -->", page.text])
    write_text_atomic(markdown_path, "\n\n".join(body).strip() + "\n")
    return MarkdownConversion(markdown_path, page_count, parsed.warnings)


class _ArticleHtmlParser(HTMLParser):
    """用标准库提取常见 HTML 正文标签，避免额外引入网页解析依赖。"""

    def __init__(self) -> None:
        """初始化标签栈和已经整理出的 Markdown 片段。"""

        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._current_tag: str | None = None
        self._buffer: list[str] = []
        self._blocks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """遇到开始标签时记录正文标签，脚本和样式内容直接忽略。"""

        del attrs
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1
            return
        if self._ignored_depth == 0 and lowered in {"p", "h1", "h2", "h3", "h4", "li", "blockquote"}:
            self._flush_current()
            self._current_tag = lowered

    def handle_endtag(self, tag: str) -> None:
        """遇到结束标签时把已收集的段落写入结果列表。"""

        lowered = tag.lower()
        if lowered in {"script", "style", "noscript", "svg"} and self._ignored_depth > 0:
            self._ignored_depth -= 1
            return
        if self._ignored_depth == 0 and self._current_tag == lowered:
            self._flush_current()

    def handle_data(self, data: str) -> None:
        """只收集正文标签内的文字，避免把导航菜单等内容写入论文正文。"""

        if self._ignored_depth == 0 and self._current_tag is not None:
            self._buffer.append(data)

    def to_markdown(self) -> str:
        """完成最后一个未闭合段落，并返回拼接后的 Markdown 正文。"""

        self._flush_current()
        return "\n\n".join(self._blocks)

    def _flush_current(self) -> None:
        """将当前标签中的文字转换成简单 Markdown 块，并清空临时缓存。"""

        if self._current_tag is None:
            return
        text = " ".join("".join(self._buffer).split())
        tag = self._current_tag
        self._buffer = []
        self._current_tag = None
        if not text:
            return
        if tag.startswith("h") and len(tag) == 2 and tag[1].isdigit():
            self._blocks.append("#" * min(int(tag[1]), 4) + " " + html.unescape(text))
        elif tag == "li":
            self._blocks.append("- " + html.unescape(text))
        elif tag == "blockquote":
            self._blocks.append("> " + html.unescape(text))
        else:
            self._blocks.append(html.unescape(text))
