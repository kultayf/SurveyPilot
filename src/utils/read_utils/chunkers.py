from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.paper_retrieval.models import PaperDocument
from src.utils.read_utils.cache import text_sha256, write_text_atomic


JsonObject = dict[str, Any]
CHUNK_SCHEMA_VERSION = 3
_PAGE_MARKER = re.compile(r"^\s*<!--\s*page:\s*(\d+)\s*-->\s*$")


@dataclass(slots=True)
class TextChunk:
    """保存可直接引用的短片段，以及帮助理解它的较长上下文。

    中文说明：content 才是 chunk_id 指向的原文。parent_content 是它所在章节的
    较长片段，仅供理解上下文；不能把父片段中的另一句话冒充当前短片段的证据。
    """

    chunk_id: str
    paperId: str
    chunk_index: int
    content: str
    page_start: int | None = None
    page_end: int | None = None
    section: str = ""
    parent_chunk_id: str = ""
    parent_content: str = ""
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    metadata: JsonObject = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        """转换成 chunk.json、检索结果共同使用的普通字段。"""

        return {
            "chunkId": self.chunk_id, "paperId": self.paperId,
            "chunk_index": self.chunk_index, "content": self.content,
            "page_start": self.page_start, "page_end": self.page_end,
            "section": self.section, "parent_chunk_id": self.parent_chunk_id,
            "parent_content": self.parent_content,
            "previous_chunk_id": self.previous_chunk_id, "next_chunk_id": self.next_chunk_id,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ChunkBuildResult:
    """保存分块结果和 chunk.json 的位置。"""

    chunks_path: Path
    chunks: list[TextChunk]


class HierarchicalChunker:
    """先按章节整理较长上下文，再切出带重叠的短片段。

    中文说明：长度统一按字符计算，不冒充精确 token 数。英文与中文的 token 比例
    不同，因此调参时必须用实际所选模型检查输入长度。
    """

    name = "hierarchical"

    def __init__(self, *, chunk_size: int = 1200, chunk_overlap: int = 150, parent_chunk_size: int = 5000):
        """检查分块参数，拒绝会产生空块或无法前进的重叠设置。"""

        if chunk_size < 1 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_size 必须为正数，chunk_overlap 必须大于等于 0 且小于 chunk_size")
        if parent_chunk_size < chunk_size:
            raise ValueError("parent_chunk_size 不能小于 chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parent_chunk_size = parent_chunk_size

    @property
    def parameters(self) -> JsonObject:
        """这些参数参与缓存判断，改动任何一个都会重新切分。"""

        return {
            "chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap,
            "parent_chunk_size": self.parent_chunk_size, "length_unit": "characters",
        }

    def chunk(self, paper: PaperDocument, markdown: str) -> list[TextChunk]:
        """在保留页码、章节及原文顺序的前提下生成父子片段。"""

        paper_id = str(paper.paperId or paper.id)
        chunks: list[TextChunk] = []
        parent_index = 0
        for section, text, page_spans in _markdown_sections(preprocess_markdown_body(markdown)):
            for parent_start, parent_end in _split_text_spans(text, self.parent_chunk_size, 0):
                parent_content = text[parent_start:parent_end]
                parent_index += 1
                # 中文说明：编号含内容摘要。原文变了，引用编号随之变化，旧摘要中的
                # 引用就不会悄悄指向另一句话；相同文件及参数再次处理会得到相同编号。
                parent_id = f"{paper_id}:h{parent_index:04d}-{text_sha256(parent_content)[:12]}"
                for child_index, (start, end) in enumerate(
                    _split_text_spans(parent_content, self.chunk_size, self.chunk_overlap), start=1
                ):
                    content = parent_content[start:end]
                    pages = [
                        page for span_start, span_end, page in page_spans
                        if page is not None and span_start < parent_start + end and span_end > parent_start + start
                    ]
                    chunks.append(TextChunk(
                        chunk_id=f"{parent_id}:s{child_index:04d}-{text_sha256(content)[:12]}",
                        paperId=paper_id, chunk_index=len(chunks), content=content,
                        page_start=min(pages) if pages else None, page_end=max(pages) if pages else None,
                        section=section, parent_chunk_id=parent_id, parent_content=parent_content,
                        metadata={"chunker": self.name, "schema_version": CHUNK_SCHEMA_VERSION},
                    ))
        for index, chunk in enumerate(chunks):
            chunk.previous_chunk_id = chunks[index - 1].chunk_id if index else None
            chunk.next_chunk_id = chunks[index + 1].chunk_id if index + 1 < len(chunks) else None
        return chunks


def build_chunks_file(
    paper: PaperDocument,
    *,
    markdown_path: Path,
    chunks_path: Path | None = None,
    chunker: HierarchicalChunker | None = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    parent_chunk_size: int = 5000,
) -> ChunkBuildResult:
    """只在原文、参数及缓存内容都未变化时复用已有分块。"""

    output_path = chunks_path or markdown_path.parent / "chunk.json"
    markdown = markdown_path.read_text(encoding="utf-8")
    resolved = chunker or HierarchicalChunker(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, parent_chunk_size=parent_chunk_size
    )
    expected = {
        "schema_version": CHUNK_SCHEMA_VERSION, "paperId": paper.paperId or paper.id,
        "chunker": resolved.name, "parameters": resolved.parameters,
        "source_hash": text_sha256(markdown),
    }
    payload = _read_chunk_payload(output_path)
    if payload and all(payload.get(key) == value for key, value in expected.items()):
        cached = load_chunks_file(output_path)
        if cached:
            return ChunkBuildResult(chunks_path=output_path, chunks=cached)
    chunks = resolved.chunk(paper, markdown)
    if not chunks:
        raise ValueError("Markdown 中没有可切分的正文")
    payload = {**expected, "chunks_hash": chunks_content_hash(chunks), "chunks": [chunk.to_dict() for chunk in chunks]}
    write_text_atomic(output_path, json.dumps(payload, ensure_ascii=False, indent=2))
    return ChunkBuildResult(chunks_path=output_path, chunks=chunks)


async def async_build_chunks_file(
    paper: PaperDocument,
    *,
    markdown_path: Path,
    chunks_path: Path | None = None,
    chunker: HierarchicalChunker | None = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    parent_chunk_size: int = 5000,
) -> ChunkBuildResult:
    """把本地读取和分块放入线程，避免阻塞其他论文的处理。"""

    return await asyncio.to_thread(
        build_chunks_file, paper, markdown_path=markdown_path, chunks_path=chunks_path,
        chunker=chunker, chunk_size=chunk_size, chunk_overlap=chunk_overlap, parent_chunk_size=parent_chunk_size,
    )


def chunks_content_hash(chunks: list[TextChunk]) -> str:
    """把原文、父上下文、位置和编号共同纳入摘要，供分块及抽取缓存校验。"""

    return text_sha256(json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False, sort_keys=True))


def load_chunks_file(chunks_path: Path) -> list[TextChunk]:
    """读取当前版本的分层片段；旧缓存需要重新处理源文件后才能用于取证。"""

    payload = _read_chunk_payload(chunks_path)
    if payload.get("schema_version") != CHUNK_SCHEMA_VERSION or not isinstance(payload.get("chunks"), list):
        return []
    chunks: list[TextChunk] = []
    try:
        for item in payload["chunks"]:
            if not isinstance(item, dict) or not item.get("content") or not item.get("chunkId"):
                return []
            chunks.append(TextChunk(
                chunk_id=str(item["chunkId"]), paperId=str(item["paperId"]),
                chunk_index=int(item["chunk_index"]), content=str(item["content"]),
                page_start=int(item["page_start"]) if item.get("page_start") is not None else None,
                page_end=int(item["page_end"]) if item.get("page_end") is not None else None,
                section=str(item.get("section") or ""),
                parent_chunk_id=str(item["parent_chunk_id"]), parent_content=str(item["parent_content"]),
                previous_chunk_id=item.get("previous_chunk_id"), next_chunk_id=item.get("next_chunk_id"),
                metadata=dict(item.get("metadata") or {}),
            ))
    except (KeyError, TypeError, ValueError):
        return []
    if payload.get("chunks_hash") != chunks_content_hash(chunks):
        return []
    return chunks


def _read_chunk_payload(path: Path) -> JsonObject:
    """读取本地缓存，不完整或损坏的文件按未命中处理。"""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def preprocess_markdown_body(markdown: str) -> str:
    """只移除文件头元数据和空字符，保留摘要、参考文献以及附录的证据。"""

    text = markdown.replace("\x00", "").strip()
    if text.startswith("---\n"):
        match = re.match(r"\A---\s*\n.*?\n---(?:\n|$)", text, flags=re.DOTALL)
        if match:
            text = text[match.end():]
    return text.strip()


def _markdown_sections(markdown: str) -> list[tuple[str, str, list[tuple[int, int, int | None]]]]:
    """按 Markdown 标题区分章节，同时记住每一行来自哪一页。"""

    sections: list[tuple[str, str, list[tuple[int, int, int | None]]]] = []
    title, page = "正文", None
    lines: list[str] = []
    spans: list[tuple[int, int, int | None]] = []
    offset = 0
    for line in markdown.splitlines(keepends=True):
        marker = _PAGE_MARKER.match(line)
        if marker:
            page = int(marker.group(1))
            continue
        heading = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if heading:
            if lines:
                sections.append((title, "".join(lines), spans))
            title, lines, spans, offset = heading.group(1).strip(), [], [], 0
        lines.append(line)
        spans.append((offset, offset + len(line), page))
        offset += len(line)
    if lines:
        sections.append((title, "".join(lines), spans))
    return sections


def _split_text_spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """优先在段落、句子或空白处切开，保留原文坐标并确保每次向前移动。"""

    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            minimum = start + max(overlap + 1, size // 2)
            window = text[minimum:end]
            for pattern in (r"\n\s*\n", r"(?<=[.!?。！？])\s+", r"\s+"):
                boundaries = list(re.finditer(pattern, window))
                if boundaries:
                    end = minimum + boundaries[-1].end()
                    break
        trimmed_start, trimmed_end = start, end
        while trimmed_start < trimmed_end and text[trimmed_start].isspace():
            trimmed_start += 1
        while trimmed_end > trimmed_start and text[trimmed_end - 1].isspace():
            trimmed_end -= 1
        if trimmed_start < trimmed_end:
            spans.append((trimmed_start, trimmed_end))
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return spans
