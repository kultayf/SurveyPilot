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
    damaged = [str(page.page_number) for page in pages if _has_unreadable_characters(page.text)]
    if empty:
        warnings.append("以下页面没有可读取文字，可能需要 OCR：" + ", ".join(empty))
    if damaged:
        warnings.append("以下页面含无法识别的字符，公式或表格需对照 PDF 复核：" + ", ".join(damaged))
    return warnings


def _has_unreadable_characters(text: str) -> bool:
    """缺字和不可显示字符都需告警；正常换行、制表符不属于缺字。"""
    return "\ufffd" in text or any(
        ord(character) < 32 and character not in "\t\n\r" for character in text
    )


class PyMuPDF4LLMParser:
    """使用版面顺序提取文字和表格；不把它当成扫描件 OCR 或公式识别保证。"""

    name = "pymupdf4llm"

    def parse(self, source_path: Path) -> PdfParseResult:
        """按页返回 Markdown；不生成图片文件，也不自动下载模型。"""

        import pymupdf
        import pymupdf4llm
        from pypdf import PdfReader

        pages = pymupdf4llm.to_markdown(
            str(source_path), page_chunks=True, write_images=False, show_progress=False, use_ocr=False,
        )
        parsed_pages: list[ParsedPdfPage] = []
        font_reader = None
        additional_warnings = []
        with pymupdf.open(source_path) as document:
            for index, page in enumerate(pages, start=1):
                markdown = str(page.get("text") or "").strip()
                # 中文说明：某些数学字体没有 Unicode 映射。版面文字会出现
                # 缺字，普通文字也可能把求和号读成字母、括号读成控制字符。
                # 用同一 PDF 的同一页做字体解码，不从模型记忆补符号；原来的
                # Markdown 保留，新增内容明确标为同页补充，页码不改变。
                if index <= len(document):
                    plain = document[index - 1].get_text("text").strip()
                    if _has_unreadable_characters(markdown) or _has_unreadable_characters(plain):
                        try:
                            if font_reader is None:
                                font_reader = PdfReader(source_path)
                            decoded = str(font_reader.pages[index - 1].extract_text() or "").strip()
                            if decoded and not _has_unreadable_characters(decoded):
                                markdown += "\n\n### 同页 PDF 字体解码文本补充（pypdf，公式仍需核对原页）\n\n" + decoded
                            else:
                                additional_warnings.append(f"第 {index} 页字体解码仍有缺字或空文本，不能据此复原公式")
                        except Exception as exc:
                            # 补充失败不能把已取得的版面文字也丢掉；保存明确警告，
                            # 写作和审计仍须拒绝无法逐字核对的损坏证据。
                            additional_warnings.append(f"第 {index} 页字体解码补充失败：{type(exc).__name__}")
                parsed_pages.append(ParsedPdfPage(page_number=index, text=markdown,
                                                  metadata={"parser": self.name}))
        result = PdfParseResult(pages=parsed_pages)
        result.warnings = [*_page_warnings(result.pages), *additional_warnings]
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
