"""从已保存的写作报告导出 BibTeX、RIS 与可独立编译的中文 LaTeX。"""
from __future__ import annotations

import hashlib
import re


def _plain(value) -> str:
    """元数据只允许单行文字，防止标题中的换行被误认成 RIS 字段。"""
    return re.sub(r"[\x00-\x1f\x7f]+", " ", str(value or "")).strip()


def latex_escape(value) -> str:
    """逐字符转义一次；原文中的命令和特殊符号只作为文字，不能成为可执行 TeX。"""
    mapping = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&",
               "#": r"\#", "%": r"\%", "_": r"\_", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(mapping.get(char, char) for char in re.sub(r"[\x00-\x1f\x7f]+", " ", str(value or "")))


def reference_key(paper_id: str) -> str:
    """编号只使用安全 ASCII 字符；同一论文在不同导出格式中的锚点保持一致。"""
    return "paper" + hashlib.sha256(paper_id.casefold().encode("utf-8")).hexdigest()[:20]


def _records(report: dict) -> list[dict]:
    records, seen = [], set()
    for reference in report.get("references") or []:
        paper_id = _plain(reference.get("paperId"))
        if not paper_id or paper_id.casefold() in seen:
            continue
        seen.add(paper_id.casefold())
        metadata = reference.get("metadata") or {}
        raw_authors = metadata.get("authors") or []
        if isinstance(raw_authors, str):
            raw_authors = [raw_authors]
        authors = [_plain(author.get("name")) if isinstance(author, dict) else _plain(author) for author in raw_authors]
        records.append({"paperId": paper_id, "key": reference_key(paper_id), "index": str(reference.get("index") or ""),
                        "title": _plain(metadata.get("title") or reference.get("citation") or paper_id),
                        "authors": [author for author in authors if author],
                        "year": _plain(metadata.get("year") or metadata.get("publication_date"))[:4],
                        "venue": _plain(metadata.get("journal_conference") or metadata.get("journal/conference") or metadata.get("venue")),
                        "doi": re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", "", _plain(metadata.get("doi")), flags=re.I),
                        "url": _plain(metadata.get("url")), "citation": _plain(reference.get("citation"))})
    return records


def export_bibtex(records: list[dict]) -> str:
    """元数据没有可靠出版类型时使用 misc，避免凭空声称期刊或会议论文。"""
    entries = []
    for record in records:
        fields = {"title": record["title"], "author": " and ".join(record["authors"]), "year": record["year"],
                  "howpublished": record["venue"], "doi": record["doi"], "url": record["url"]}
        content = ",\n".join(f"  {key} = {{{latex_escape(value)}}}" for key, value in fields.items() if value)
        entries.append(f"@misc{{{record['key']},\n{content}\n}}")
    return "\n\n".join(entries) + "\n"


def export_ris(records: list[dict]) -> str:
    """RIS 是 Zotero 可导入的文献交换格式；每位作者独占一行。"""
    lines = []
    for record in records:
        lines += ["TY  - GEN", "ID  - " + record["key"], "TI  - " + record["title"]]
        lines += ["AU  - " + author for author in record["authors"]]
        for tag, key in (("PY", "year"), ("T2", "venue"), ("DO", "doi"), ("UR", "url")):
            if record[key]:
                lines.append(f"{tag}  - {record[key]}")
        lines += ["ER  - ", ""]
    return "\n".join(lines)


def _tex_paragraphs(text: str, records: list[dict]) -> str:
    """保留段落及合法引用锚点；Markdown 和公式按普通文本输出，不猜测公式语义。"""
    index_map = {record["index"]: record["key"] for record in records}
    id_map = {record["paperId"].casefold(): record["key"] for record in records}
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text):
        parts, position = [], 0
        for match in re.finditer(r"\[([^\[\]\n]+)\]", paragraph):
            parts.append(latex_escape(paragraph[position:match.start()]))
            raw = match.group(1).strip()
            identifiers = [raw] if raw.casefold() in id_map else re.split(r"[,;，；]\s*", raw)
            keys = [id_map.get(value.casefold()) or index_map.get(value) for value in identifiers]
            if keys and all(keys):
                parts.append(r"\cite{" + ",".join(dict.fromkeys(keys)) + "}")
            else:
                parts.append(latex_escape(match.group(0)))
            position = match.end()
        parts.append(latex_escape(paragraph[position:]))
        if paragraph.strip():
            paragraphs.append("".join(parts))
    return "\n\n".join(paragraphs)


def export_latex(report: dict, topic: str, records: list[dict]) -> str:
    """引用表内嵌在 tex 中；只需 XeLaTeX 两遍，不依赖 BibTeX 或 shell-escape。"""
    blocks = [r"\documentclass[UTF8,fontset=fandol]{ctexart}", r"\usepackage[a4paper,margin=25mm]{geometry}",
              r"\usepackage{hyperref}", r"\hypersetup{hidelinks}", r"\setlength{\emergencystretch}{3em}",
              r"\title{" + latex_escape(topic or report.get("topic") or "文献综述") + "}",
              r"\author{}", r"\date{}", r"\begin{document}", r"\maketitle"]
    status = "独立模型核查通过；科研使用前仍需核对原文。" if report.get("citation_audit_status") == "passed" else "待核查草稿：存在证据不足、引用错误或尚未验证的内容，未达到核查通过稿交付条件。"
    blocks.append(r"\noindent\textbf{" + latex_escape(status) + "}")
    if report.get("abstract"):
        blocks += [r"\begin{abstract}", _tex_paragraphs(report["abstract"], records), r"\end{abstract}"]
    chapter = ""
    for section in report.get("sections") or []:
        if section.get("chapter_key") and section["chapter_key"] != chapter:
            chapter = section["chapter_key"]
            blocks.append(r"\section{" + latex_escape(section.get("chapter_title") or chapter) + "}")
        blocks += [r"\subsection{" + latex_escape(section.get("section_title") or section.get("section_id")) + "}",
                   _tex_paragraphs(str(section.get("content") or ""), records)]
    if records:
        blocks.append(r"\begin{thebibliography}{" + str(len(records)) + "}")
        for record in records:
            blocks.append(r"\bibitem{" + record["key"] + "} " + latex_escape(record["citation"] or record["title"]))
        blocks.append(r"\end{thebibliography}")
    blocks.append(r"\end{document}")
    return "\n\n".join(blocks) + "\n"


def build_research_exports(report: dict, topic: str) -> dict[str, str]:
    """导出始终保留核查状态；文献交换文件只含实际引用的已知元数据。"""
    records = _records(report)
    stem = "literature_review" if report.get("citation_audit_status") == "passed" else "literature_review_draft"
    return {"references.bib": export_bibtex(records), "references.ris": export_ris(records),
            stem + ".tex": export_latex(report, topic, records),
            "EXPORT_README.txt": "LaTeX：需要含 ctex、Fandol 字体的 TeX Live / MiKTeX，使用 XeLaTeX 编译两遍。\n"
            "参考文献已内嵌，无需另跑 BibTeX；references.bib / references.ris 可导入 Zotero。\n"
            "正文中的公式、Markdown 与代码按安全的普通文本导出，未转换为数学排版。\n"
            "核查失败或未完成时文件名带 draft，正文亦保留待核查说明。导出不等于科研内容已获验证。\n"}
