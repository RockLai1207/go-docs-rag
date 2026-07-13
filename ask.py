"""
Step 3 — Ask: retrieve relevant chunks and generate a grounded answer.
----------------------------------------------------------------------
RAG 管線的最後一步：
    1. 使用者用中文（或英文）提問
    2. 問題轉向量，從 Chroma 檢索最相關的 Go 官方文件段落
    3. 把段落作為依據交給 Gemini，生成繁體中文答案並標注來源
    4. 若文件中沒有相關內容，誠實說不知道（拒答機制，抑制幻覺）

Usage:
    python ask.py "goroutine 和 thread 有什麼不同?"
    python ask.py            # 不帶參數則進入互動模式
"""

import logging
import os
import sys
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "data" / "chroma_db"

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
GEN_MODEL = "gemini-2.5-flash"
COLLECTION_NAME = "go_docs"

TOP_K = 4                    # 每次檢索取幾個段落交給生成模型
MAX_DISTANCE = 0.55          # 相關度門檻：全部段落都比這遠就直接拒答，不送生成
MAX_RETRIES = 3
RETRY_BASE_DELAY = 10

logging.basicConfig(level=logging.WARNING)  # 問答模式下保持輸出乾淨
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是 Go 語言官方文件的問答助手。你只能根據提供的「參考段落」回答問題。

規則：
1. 一律使用繁體中文回答（程式碼與專有名詞保持英文）
2. 答案必須完全基於參考段落的內容，不可以添加段落中沒有的資訊
3. 如果參考段落不足以回答問題，直接說「官方文件中沒有找到相關內容」，不要猜測或編造
4. 回答結尾用「來源：」列出你實際引用的段落編號與其文件章節
5. 適合用程式碼說明時，引用參考段落中的程式碼範例
"""


def get_client() -> genai.Client:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("錯誤：GEMINI_API_KEY 未設定。請複製 .env.example 為 .env 並填入你的 key。")
        sys.exit(1)
    return genai.Client(api_key=api_key)


def with_retry(fn):
    """429 退避重試的小包裝。"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except genai_errors.ClientError as e:
            if "429" in str(e) and attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(f"(額度限制，{delay} 秒後重試...)")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("retries exhausted")


def retrieve(client: genai.Client, collection, question: str) -> list[dict]:
    """問題轉向量 -> 檢索 Top K 段落，回傳含 metadata 與距離的結果。"""
    q_vec = with_retry(lambda: client.models.embed_content(
        model=EMBED_MODEL,
        contents=[question],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=EMBED_DIM),
    )).embeddings[0].values

    results = collection.query(query_embeddings=[q_vec], n_results=TOP_K)
    hits = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        hits.append({"text": doc, "meta": meta, "distance": dist})
    return hits


def build_context(hits: list[dict]) -> str:
    """把檢索結果組成給生成模型的參考段落文字。"""
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        blocks.append(
            f"[段落 {i}] 來源：{m['doc_title']} › {m['section']}（{m['doc_url']}）\n{h['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def answer(client: genai.Client, question: str, hits: list[dict]) -> str:
    context = build_context(hits)
    prompt = f"""參考段落：

{context}

---

問題：{question}"""

    response = with_retry(lambda: client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,   # 問答場景要穩定貼合資料，不要創意
        ),
    ))
    return response.text


def ask(client: genai.Client, collection, question: str) -> None:
    hits = retrieve(client, collection, question)

    # 拒答機制第一層：檢索端。全部段落都不夠相關就不送生成，省額度也防幻覺
    if not hits or min(h["distance"] for h in hits) > MAX_DISTANCE:
        print("\n官方文件中沒有找到與這個問題相關的內容。")
        print("（本系統的知識範圍：Effective Go、Go FAQ、How to Write Go Code、go.mod Reference、Go Doc Comments）")
        return

    print("\n" + answer(client, question, hits))

    print("\n--- 檢索到的段落（依相關度排序）---")
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        print(f"  [{i}] {m['doc_title']} › {m['section']}  (distance={h['distance']:.3f})")
        print(f"      {m['doc_url']}")


def main() -> None:
    if not DB_DIR.exists():
        print(f"找不到向量資料庫 {DB_DIR} — 請先執行 python embed_index.py 建立索引。")
        sys.exit(1)

    client = get_client()
    db = chromadb.PersistentClient(path=str(DB_DIR))
    collection = db.get_collection(COLLECTION_NAME)

    if len(sys.argv) > 1:
        ask(client, collection, " ".join(sys.argv[1:]))
        return

    # 互動模式
    print("Go 官方文件問答（輸入 exit 離開）")
    while True:
        try:
            question = input("\n問題> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("exit", "quit"):
            break
        ask(client, collection, question)


if __name__ == "__main__":
    main()
