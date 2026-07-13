"""
Step 2 — Index: embed chunks and store them in a vector database.
-----------------------------------------------------------------
RAG 管線的第二步：
    1. 讀取 Step 1 產出的 data/chunks.json（247 個知識塊）
    2. 用 Gemini embedding API 把每塊文字轉成語意向量
    3. 連同 metadata（文件名、章節、網址）存入 Chroma 向量資料庫
    4. 建完索引後跑一次冒煙測試，驗證檢索功能正常

設計重點：
    - 批次處理：一次送多塊給 API，減少請求數（免費額度友善）
    - 斷點續跑：已入庫的 chunk 自動跳過，中斷後重跑不會浪費額度
    - 429 重試：免費額度撞到 rate limit 時自動等待重試

Usage:
    python embed_index.py
"""

import json
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
CHUNKS_PATH = BASE_DIR / "data" / "chunks.json"
DB_DIR = BASE_DIR / "data" / "chroma_db"

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768              # 向量維度：768 在品質與儲存成本間是好平衡
COLLECTION_NAME = "go_docs"
BATCH_SIZE = 20              # 每次 API 呼叫送幾塊
MAX_RETRIES = 5
RETRY_BASE_DELAY = 10        # 429 重試等待秒數（每次翻倍）

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def get_client() -> genai.Client:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)
    return genai.Client(api_key=api_key)


def embed_batch(client: genai.Client, texts: list[str], task_type: str) -> list[list[float]]:
    """呼叫 Gemini embedding API，含 429 重試。

    task_type 很重要：
      - RETRIEVAL_DOCUMENT：建索引時用（告訴模型這是「被搜尋的資料」）
      - RETRIEVAL_QUERY：查詢時用（告訴模型這是「搜尋的問題」）
    兩者用不同方式編碼，是 Gemini embedding 提升檢索品質的設計。
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=EMBED_DIM,
                ),
            )
            return [e.values for e in result.embeddings]
        except genai_errors.ClientError as e:
            if "429" in str(e) and attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(f"Rate limited (attempt {attempt}/{MAX_RETRIES}), waiting {delay}s...")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("embed_batch exhausted retries")


def build_index() -> None:
    if not CHUNKS_PATH.exists():
        logger.error(f"{CHUNKS_PATH} not found — run `python ingest.py` first.")
        sys.exit(1)

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks")

    client = get_client()
    db = chromadb.PersistentClient(path=str(DB_DIR))
    collection = db.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # 餘弦相似度：語意檢索的標準選擇
    )

    # 斷點續跑：跳過已入庫的 chunk
    existing_ids = set(collection.get(include=[])["ids"])
    todo = [c for c in chunks if c["chunk_id"] not in existing_ids]
    if not todo:
        logger.info("Index already up to date — nothing to embed.")
        return
    logger.info(f"{len(existing_ids)} already indexed, {len(todo)} to embed")

    for i in range(0, len(todo), BATCH_SIZE):
        batch = todo[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        vectors = embed_batch(client, texts, task_type="RETRIEVAL_DOCUMENT")

        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=vectors,
            documents=texts,
            metadatas=[
                {"doc_title": c["doc_title"], "doc_url": c["doc_url"], "section": c["section"]}
                for c in batch
            ],
        )
        done = min(i + BATCH_SIZE, len(todo))
        logger.info(f"Indexed {done}/{len(todo)}")
        time.sleep(1)  # 溫和節流，避免免費額度撞牆

    logger.info(f"Done — collection `{COLLECTION_NAME}` now has {collection.count()} vectors at {DB_DIR}")


def smoke_test() -> None:
    """建完索引後跑一次檢索冒煙測試，確認整條路是通的。"""
    client = get_client()
    db = chromadb.PersistentClient(path=str(DB_DIR))
    collection = db.get_collection(COLLECTION_NAME)

    question = "What is a goroutine and how is it different from a thread?"
    logger.info(f"Smoke test query: {question}")
    q_vec = embed_batch(client, [question], task_type="RETRIEVAL_QUERY")[0]

    results = collection.query(query_embeddings=[q_vec], n_results=3)
    print("\n=== 檢索結果 Top 3 ===")
    for rank, (doc, meta, dist) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0]), 1
    ):
        print(f"\n#{rank}  [{meta['doc_title']} › {meta['section']}]  distance={dist:.4f}")
        print(doc[:200].replace("\n", " ") + " ...")


if __name__ == "__main__":
    build_index()
    smoke_test()
