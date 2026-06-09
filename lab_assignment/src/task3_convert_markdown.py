"""
Task 3 - Convert files in data/landing/ to Markdown.

Scans data/landing/legal and data/landing/news, then writes converted Markdown
files to data/standardized/ while preserving the legal/news folder structure.
"""

import json
import re
import zipfile
from html import unescape
from pathlib import Path
from xml.etree import ElementTree

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _safe_text(value) -> str:
    text = str(value)
    return text.encode("ascii", errors="backslashreplace").decode("ascii")


def _print(message: str) -> None:
    print(_safe_text(message))


def _safe_filename(stem: str) -> str:
    filename = re.sub(r'[<>:"/\\|?*]+', "-", stem).strip()
    return filename or "document"


def _markdown_header(title: str, metadata: dict[str, str]) -> str:
    lines = [f"# {title or 'Untitled'}", ""]
    for key, value in metadata.items():
        if value:
            lines.append(f"**{key}:** {value}")
    lines.extend(["", "---", ""])
    return "\n".join(lines)


def _convert_with_markitdown(filepath: Path) -> str:
    try:
        from markitdown import MarkItDown
    except ImportError as exc:
        raise RuntimeError("markitdown is not installed") from exc

    result = MarkItDown().convert(str(filepath))
    return result.text_content


def _convert_docx_fallback(filepath: Path) -> str:
    """
    Extract text from a DOCX file using only the standard library.

    This keeps conversion usable in environments where MarkItDown is not
    installed or cannot parse a specific file.
    """
    with zipfile.ZipFile(filepath) as docx:
        xml_text = docx.read("word/document.xml")

    root = ElementTree.fromstring(xml_text)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []

    for paragraph in root.findall(".//w:p", namespace):
        parts = [
            node.text
            for node in paragraph.findall(".//w:t", namespace)
            if node.text
        ]
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def _convert_legal_file(filepath: Path) -> str:
    if filepath.suffix.lower() == ".docx":
        try:
            return _convert_with_markitdown(filepath)
        except Exception as exc:
            _print(f"  MarkItDown failed, using DOCX fallback: {exc}")
            return _convert_docx_fallback(filepath)

    return _convert_with_markitdown(filepath)


def convert_legal_docs() -> list[Path]:
    """Convert PDF/DOC/DOCX files in data/landing/legal/ to Markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        _print(f"Legal directory not found: {legal_dir}")
        return []

    converted: list[Path] = []
    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in {".pdf", ".docx", ".doc"}:
            continue

        _print(f"Converting: {filepath.name}")
        content = _convert_legal_file(filepath).strip()
        if not content:
            _print("  Skipped: empty converted content")
            continue

        output_path = output_dir / f"{_safe_filename(filepath.stem)}.md"
        markdown = _markdown_header(
            filepath.stem,
            {
                "Source file": filepath.name,
                "Document type": "legal",
            },
        )
        output_path.write_text(markdown + content + "\n", encoding="utf-8")
        converted.append(output_path)
        _print(f"  Saved: {output_path}")

    return converted


def _json_article_to_markdown(data: dict) -> str:
    title = data.get("title") or "Unknown"
    content = data.get("content_markdown") or data.get("content") or ""
    content = unescape(str(content)).strip()

    header = _markdown_header(
        title,
        {
            "Source": data.get("url", "N/A"),
            "Crawled": data.get("date_crawled", "N/A"),
            "Document type": "news",
        },
    )
    return header + content + "\n"


def convert_news_articles() -> list[Path]:
    """Convert JSON crawled articles in data/landing/news/ to Markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        _print(f"News directory not found: {news_dir}")
        return []

    converted: list[Path] = []
    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue

        _print(f"Converting: {filepath.name}")
        data = json.loads(filepath.read_text(encoding="utf-8"))
        markdown = _json_article_to_markdown(data)
        if len(markdown.strip()) <= 200:
            _print("  Skipped: converted content is too short")
            continue

        output_path = output_dir / f"{_safe_filename(filepath.stem)}.md"
        output_path.write_text(markdown, encoding="utf-8")
        converted.append(output_path)
        _print(f"  Saved: {output_path}")

    return converted


def convert_all() -> list[Path]:
    """Convert all supported landing files to Markdown."""
    _print("=" * 50)
    _print("Task 3: Convert to Markdown")
    _print("=" * 50)

    _print("\n--- Legal Documents ---")
    legal_files = convert_legal_docs()

    _print("\n--- News Articles ---")
    news_files = convert_news_articles()

    converted = legal_files + news_files
    _print(f"\nDone. Converted {len(converted)} files to: {OUTPUT_DIR}")
    return converted


if __name__ == "__main__":
    convert_all()
