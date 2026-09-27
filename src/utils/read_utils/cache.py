from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from src.paper_retrieval.models import PaperDocument


JsonObject = dict[str, Any]


def text_sha256(text: str) -> str:
    """按实际文字内容生成摘要，用于判断缓存是否仍对应当前原文。"""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """分批读取文件计算摘要，避免大 PDF 一次占用太多内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_atomic(path: Path, text: str) -> None:
    """先完整写入临时文件，再替换缓存，避免中断留下半个 JSON 或 Markdown。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as output:
            temporary_path = Path(output.name)
            output.write(text)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def paper_cache_dir(base_dir: str | Path, paper: PaperDocument) -> Path:
    """用论文编号生成可读且不易重名的缓存目录。"""

    paper_id = str(paper.paperId or paper.id or paper.doi or paper.title).strip()
    return Path(base_dir) / safe_cache_name(paper_id)


def safe_cache_name(value: str) -> str:
    """保留可读编号并附加内容摘要，避免斜杠替换或截断后两个编号同名。"""

    cleaned = "".join(character if character.isalnum() or character in {"-", "_", "."} else "_" for character in value)
    return f"{cleaned[:140] or 'paper'}_{text_sha256(value)[:12]}"


def write_metadata(cache_dir: Path, paper: PaperDocument, *, source_url: str | None, content_type: str | None) -> Path:
    """把论文元数据写入缓存目录里的 metadata.json。

    中文注释：只有下载到全文后才会调用这个函数，所以不会给下载失败的论文
    创建空缓存。metadata 里同时放论文原始信息和全文来源，后面分析节点不用
    再回头猜这篇论文是从哪里来的。
    """

    payload: JsonObject = {
        "paperId": paper.paperId or paper.id,
        "paper": paper.to_dict(),
        "fulltext": {
            "source_url": source_url,
            "content_type": content_type,
        },
    }
    path = cache_dir / "metadata.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_cached_source_url(cache_dir: Path) -> str | None:
    """从缓存里的 metadata.json 读回全文来源地址。

    中文注释：旧缓存里可能还有 source.json，所以这里顺手兼容一下。主流程新写入
    的都是 metadata.json。
    """

    for name in ("metadata.json", "source.json"):
        try:
            payload = json.loads((cache_dir / name).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            # 中文说明：损坏的缓存文件按未命中处理，后续流程仍可尝试其他来源。
            continue
        fulltext = payload.get("fulltext") if isinstance(payload, dict) else None
        if isinstance(fulltext, dict) and fulltext.get("source_url"):
            return str(fulltext["source_url"])
        if isinstance(payload, dict) and payload.get("source_url"):
            return str(payload["source_url"])
    return None
