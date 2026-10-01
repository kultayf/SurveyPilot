from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ParsedPdfPage:
    """保存原始页码及版面解析得到的 Markdown，不提前删除参考文献或附录。"""

    page_number: int
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class PdfParseResult:
    """空页也保留在结果中，让后续页数和引用位置保持准确。"""

    pages: list[ParsedPdfPage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _page_warnings(pages: list[ParsedPdfPage]) -> list[str]:
    """标出确定可见的缺字；没有告警也不代表公式与复杂表格已经人工核对。"""
    warnings = []
    empty = [str(page.page_number) for page in pages if not page.text.strip()]
    damaged = [str(page.page_number) for page in pages if "\ufffd" in page.text]
    if empty:
        warnings.append("以下页面没有可读取文字，可能需要 OCR：" + ", ".join(empty))
    if damaged:
        warnings.append("以下页面含无法识别的字符，公式或表格需对照 PDF 复核：" + ", ".join(damaged))
    return warnings


class PyMuPDF4LLMParser:
    """使用版面顺序提取文字和表格；不把它当成扫描件 OCR 或公式识别保证。"""

    name = "pymupdf4llm"

    def parse(self, source_path: Path) -> PdfParseResult:
        """按页返回 Markdown；不生成图片文件，也不自动下载模型。"""

        import pymupdf
        import pymupdf4llm

        pages = pymupdf4llm.to_markdown(
            str(source_path), page_chunks=True, write_images=False, show_progress=False, use_ocr=False,
        )
        parsed_pages: list[ParsedPdfPage] = []
        with pymupdf.open(source_path) as document:
            for index, page in enumerate(pages, start=1):
                markdown = str(page.get("text") or "").strip()
                # 中文说明：版面转换偶尔把公式中的求和号、上下标变成无法识别
                # 的字符。只在确实出现这种损坏时，从同一份 PDF 的同一页再取
                # 一份普通文字，供后续作者和审计逐字查证；原 Markdown 不删除，
                # 页码不变，警告也继续保留。普通文字可能仍不完美，不能据此
                # 宣称公式已验证，更不能从其他论文或模型记忆补出缺失符号。
                if "\ufffd" in markdown and index <= len(document):
                    plain = document[index - 1].get_text("text").strip()
                    if plain and plain.count("\ufffd") < markdown.count("\ufffd"):
                        markdown += "\n\n### 同页 PDF 纯文本补充（公式仍需核对原页）\n\n" + plain
                parsed_pages.append(ParsedPdfPage(page_number=index, text=markdown,
                                                  metadata={"parser": self.name}))
        result = PdfParseResult(pages=parsed_pages)
        result.warnings = _page_warnings(result.pages)
        return result


class DoclingParser:
    """显式选择并准备本地模型后，使用 Docling 解析复杂版面。"""

    name = "docling"

    def __init__(self, artifacts_path: str | Path | None):
        """本地模型位置必须由配置提供，避免启动流程时意外下载大型文件。"""

        self.artifacts_path = Path(artifacts_path) if artifacts_path else None

    def parse(self, source_path: Path) -> PdfParseResult:
        """关闭 OCR，按原页码导出 Markdown；缺少模型时返回明确错误。"""

        if self.artifacts_path is None or not self.artifacts_path.is_dir():
            raise ValueError("选择 Docling 前需在 read.docling_artifacts_path 配置已准备好的本地模型目录")
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        options = PdfPipelineOptions(artifacts_path=self.artifacts_path, do_ocr=False)
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
        document = converter.convert(str(source_path)).document
        pages = [ParsedPdfPage(page_number=number,
                               text=document.export_to_markdown(page_no=number),
                               metadata={"parser": self.name})
                 for number in sorted(document.pages)]
        return PdfParseResult(pages=pages, warnings=_page_warnings(pages))


def get_pdf_parser(name: str = "pymupdf4llm", *, docling_artifacts_path: str | Path | None = None):
    """只装配用户明确选择的解析器，出错时不会偷偷切换成另一种实现。"""

    if name == "pymupdf4llm":
        return PyMuPDF4LLMParser()
    if name == "docling":
        return DoclingParser(docling_artifacts_path)
    raise ValueError(f"未知 PDF 解析器：{name}")
