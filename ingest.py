"""
Step 1 — Ingest: fetch Go official docs, clean, and chunk.
----------------------------------------------------------
RAG 管線的第一步：
    1. 從 golang/website repo 下載官方文件（HTML 與 Markdown 混合）
    2. 清理成純文字（保留段落與程式碼區塊）
    3. 依「標題結構優先、長度上限為輔」的策略切塊（chunking）
    4. 輸出 chunks.json，供下一步 embedding 使用

Usage:
    python ingest.py
"""

import json
import logging
import re
import sys
from dataclasses import dataclass, asdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import urlopen, Request

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data" / "raw"
CHUNKS_PATH = BASE_DIR / "data" / "chunks.json"

RAW_BASE = "https://raw.githubusercontent.com/golang/website/master/_content/doc"

# 語料庫：Go 官方核心文件
# (檔案路徑, 顯示名稱, 官方網址-用於答案的來源標注)
SOURCES = [
    ("effective_go.html", "Effective Go", "https://go.dev/doc/effective_go"),
    ("faq.md", "Go FAQ", "https://go.dev/doc/faq"),
    ("code.html", "How to Write Go Code", "https://go.dev/doc/code"),
    ("modules/gomod-ref.md", "go.mod Reference", "https://go.dev/doc/modules/gomod-ref"),
    ("comment.md", "Go Doc Comments", "https://go.dev/doc/comment"),
]

# 切塊參數
MAX_CHUNK_CHARS = 2000   # 單一 chunk 的長度上限（約 500 tokens）
MIN_CHUNK_CHARS = 150    # 過短的碎片會併入前一個 chunk
OVERLAP_CHARS = 200      # 超長段落強制切分時，前後保留的重疊區

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTML -> 結構化文字
# ---------------------------------------------------------------------------
class GoDocHTMLParser(HTMLParser):
    """把 Go 官方文件的 HTML 轉成帶標題標記的純文字。

    輸出格式：標題行以「## 」開頭（模仿 markdown），
    程式碼區塊用 ``` 包住，讓後續切塊器可以用同一套邏輯處理
    HTML 與 Markdown 兩種來源。
    """

    HEADING_TAGS = {"h1", "h2", "h3", "h4"}
    SKIP_TAGS = {"script", "style", "nav", "header", "footer"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._in_heading = False
        self._heading_level = 2
        self._in_pre = False
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        elif tag in self.HEADING_TAGS:
            self._in_heading = True
            self._heading_level = int(tag[1]) + 1  # h2 -> ###? 統一降一級不必要，直接對應
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "pre":
            self._in_pre = True
            self.parts.append("\n```\n")
        elif tag == "p":
            self.parts.append("\n\n")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in self.HEADING_TAGS:
            self._in_heading = False
            self.parts.append("\n")
        elif tag == "pre":
            self._in_pre = False
            self.parts.append("\n```\n")

    def handle_data(self, data):
        if self._skip_depth > 0:
            return
        if self._in_pre:
            self.parts.append(data)
        else:
            self.parts.append(re.sub(r"\s+", " ", data))

    def get_text(self) -> str:
        text = "".join(self.parts)
        text = re.sub(r"\n{3,}", "\n\n", text)   # 壓縮多餘空行
        return text.strip()


def strip_front_matter(text: str) -> str:
    """移除 golang/website 文件開頭的 JSON front matter 註解。"""
    return re.sub(r"^<!--\{.*?\}-->\s*", "", text, flags=re.DOTALL)


def html_to_text(html: str) -> str:
    parser = GoDocHTMLParser()
    parser.feed(strip_front_matter(html))
    return parser.get_text()


def clean_markdown(md: str) -> str:
    """Markdown 文件僅需移除 front matter，標題結構本身就是我們要的格式。"""
    return strip_front_matter(md).strip()


# ---------------------------------------------------------------------------
# 切塊（chunking）
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    doc_title: str      # 所屬文件（例如 Effective Go）
    doc_url: str        # 官方網址，用於答案的來源標注
    section: str        # 所屬章節標題
    text: str           # chunk 內容
    chunk_id: str       # 唯一識別


def split_by_headings(text: str) -> list[tuple[str, str]]:
    """依標題行切成 (章節標題, 章節內容) 的列表。"""
    sections: list[tuple[str, str]] = []
    current_title = "Introduction"
    current_lines: list[str] = []

    for line in text.splitlines():
        m = re.match(r"^#{1,4}\s+(.+)$", line)
        if m:
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = m.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return [(t, c) for t, c in sections if c]


def split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """章節超過長度上限時，優先在段落邊界切分，並保留重疊區。"""
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n\n")
    pieces: list[str] = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                pieces.append(buf)
            # 段落本身就超長（例如超大程式碼區塊）時硬切
            while len(p) > max_chars:
                pieces.append(p[:max_chars])
                p = p[max_chars - overlap:]
            buf = p
    if buf:
        pieces.append(buf)

    # 為相鄰 pieces 加上重疊區，降低語意被切斷的檢索損失
    overlapped: list[str] = []
    for i, piece in enumerate(pieces):
        if i > 0 and overlap > 0:
            tail = pieces[i - 1][-overlap:]
            piece = f"...{tail}\n\n{piece}"
        overlapped.append(piece)
    return overlapped


def chunk_document(doc_title: str, doc_url: str, text: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    seq = 0  # 文件內全域流水號，避免同名章節導致 chunk_id 重複
    for section_title, section_text in split_by_headings(text):
        for piece in split_long_text(section_text, MAX_CHUNK_CHARS, OVERLAP_CHARS):
            if len(piece) < MIN_CHUNK_CHARS and chunks:
                # 過短碎片併入前一個 chunk，避免產生無語意的向量
                chunks[-1].text += "\n\n" + piece
                continue
            chunk_id = f"{doc_title}::{seq:04d}::{section_title}"
            chunks.append(Chunk(doc_title, doc_url, section_title, piece, chunk_id))
            seq += 1
    return chunks


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def fetch(url: str) -> str:
    req = Request(url, headers={"User-Agent": "go-docs-rag/0.1"})
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks: list[Chunk] = []

    for rel_path, title, url in SOURCES:
        logger.info(f"Fetching {title} ...")
        raw = fetch(f"{RAW_BASE}/{rel_path}")

        # 快取原始檔，避免重複下載
        cache_path = RAW_DIR / rel_path.replace("/", "__")
        cache_path.write_text(raw, encoding="utf-8")

        text = html_to_text(raw) if rel_path.endswith(".html") else clean_markdown(raw)
        chunks = chunk_document(title, url, text)
        logger.info(f"  -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in all_chunks], f, ensure_ascii=False, indent=2)

    sizes = [len(c.text) for c in all_chunks]
    logger.info(f"Total: {len(all_chunks)} chunks -> {CHUNKS_PATH}")
    logger.info(f"Chunk size: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)}")


if __name__ == "__main__":
    main()
