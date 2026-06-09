"""
Task 2 - Crawl news articles about artists related to drug cases.

Requirements:
    1. Crawl at least 5 Vietnamese news articles.
    2. Save output to data/landing/news/.
    3. Store one JSON file per article with metadata:
       url, title, date_crawled, content_markdown.
"""

import asyncio
import json
import re
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory() -> None:
    """Create data/landing/news/ if it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://cuoi.tuoitre.vn/loat-nghe-si-viet-tieu-tan-su-nghiep-vi-ma-tuy-20241114142620463.htm",
    "https://cuoi.tuoitre.vn/chuyen-gia-tam-ly-nghe-si-bao-dung-ma-tuy-de-sang-tao-la-dang-lua-doi-chinh-minh-20250724191615468.htm",
    "https://tuoitre.vn/bat-ca-si-long-nhat-va-ca-si-son-ngoc-minh-vi-lien-quan-ma-tuy-20260520082138943.htm",
    "https://tuoitre.vn/ca-si-long-nhat-khai-su-dung-ma-tuy-da-cung-quan-ly-20260520132251413.htm",
    "https://tuoitre.vn/vi-sao-g-dragon-lee-sun-kyun-va-nhieu-sao-han-dinh-dam-dinh-ma-tuy-20231031093753313.htm",
]


class _ArticleHTMLParser(HTMLParser):
    """Small stdlib-only HTML extractor for the requests fallback."""

    def __init__(self) -> None:
        super().__init__()
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._tag_stack: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        self._tag_stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        if self._tag_stack:
            self._tag_stack.pop()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return

        text = " ".join(unescape(data).split())
        if not text:
            return

        current_tag = self._tag_stack[-1] if self._tag_stack else ""
        if current_tag == "title":
            self.title_parts.append(text)
        elif current_tag in {"h1", "h2", "p", "li"}:
            self.text_parts.append(text)

    @property
    def title(self) -> str:
        return " ".join(self.title_parts).strip()

    @property
    def content(self) -> str:
        return "\n\n".join(self.text_parts).strip()


def _slugify(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug[:80] or fallback


def _crawl_with_requests(url: str) -> dict:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding

    parser = _ArticleHTMLParser()
    parser.feed(response.text)

    title = parser.title or "Unknown"
    content = parser.content
    if not content:
        content = re.sub(r"<[^>]+>", " ", response.text)
        content = " ".join(unescape(content).split())

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content,
    }


async def crawl_article(url: str) -> dict:
    """
    Crawl one article and return metadata plus markdown content.

    Crawl4AI is used when available. The requests fallback keeps the task usable
    in lighter environments where browser dependencies are not installed.
    """
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            title = getattr(result, "metadata", {}).get("title", "Unknown")
            markdown = getattr(result, "markdown", "") or ""
            return {
                "url": url,
                "title": title,
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": markdown,
            }
    except Exception as exc:
        print(f"  Crawl4AI failed, falling back to requests: {exc}")
        return await asyncio.to_thread(_crawl_with_requests, url)


async def crawl_all() -> None:
    """Crawl all articles in ARTICLE_URLS and save one JSON file per article."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        clean_url = url.strip()
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {clean_url}")
        article = await crawl_article(clean_url)

        slug = _slugify(article.get("title", ""), f"article-{i:02d}")
        filename = f"article_{i:02d}_{slug}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("Please fill ARTICLE_URLS before running.")
    else:
        asyncio.run(crawl_all())
