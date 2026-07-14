"""
Export visualization data: read embeddings from Chroma, reduce to 3D, emit JSON.
--------------------------------------------------------------------------------
把 RAG 系統的向量空間變成可視化資料：
    1. 從 Chroma 讀出全部 chunks 的 768 維向量、原文、metadata
    2. 降維到 3D（優先用 UMAP，沒裝就退回 PCA——純 numpy，零額外依賴）
    3. 投影到球殼上（好看，且語意群聚方向保留）
    4. 輸出 viz/viz_data.json 供 viz.html 使用
       （包含原始 768 維向量，讓瀏覽器端能直接算 cosine 相似度做即時檢索）

Usage:
    python export_viz.py
    # 然後: cd viz && python -m http.server 8000
    # 瀏覽器打開 http://localhost:8000/viz.html
"""

import json
import logging
from pathlib import Path

import chromadb
import numpy as np

BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "data" / "chroma_db"
OUT_DIR = BASE_DIR / "viz"
OUT_PATH = OUT_DIR / "viz_data.json"

COLLECTION_NAME = "go_docs"
TEXT_PREVIEW_CHARS = 320   # 前端 tooltip 顯示的文字長度
EMB_ROUND = 4              # 向量小數位數（縮小 JSON 體積）

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def reduce_to_3d(vectors: np.ndarray) -> np.ndarray:
    """768 維 -> 3 維。優先 UMAP（群聚效果最好），沒裝退回 PCA。"""
    try:
        import umap  # pip install umap-learn（可選）
        logger.info("Using UMAP for dimensionality reduction")
        reducer = umap.UMAP(n_components=3, n_neighbors=12, min_dist=0.15, metric="cosine", random_state=42)
        return reducer.fit_transform(vectors)
    except ImportError:
        logger.info("umap-learn not installed — falling back to PCA (pip install umap-learn for better clusters)")
        centered = vectors - vectors.mean(axis=0)
        # SVD 取前三主成分：純 numpy 的 PCA
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return centered @ vt[:3].T


def project_to_shell(coords: np.ndarray, r_min: float = 78.0, r_max: float = 112.0) -> np.ndarray:
    """把 3D 點投影到球殼上：方向保留語意群聚，半徑映射原本離中心的距離。

    這是純視覺美學的選擇（參考星雲式呈現）——群聚在角度方向上依然分明。
    """
    center = coords.mean(axis=0)
    v = coords - center
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    directions = v / norms
    flat = norms.flatten()
    radius = r_min + (r_max - r_min) * (flat - flat.min()) / (flat.max() - flat.min() + 1e-9)
    return directions * radius[:, None]


def main() -> None:
    if not DB_DIR.exists():
        raise SystemExit(f"找不到 {DB_DIR} — 請先執行 python embed_index.py 建立索引。")

    db = chromadb.PersistentClient(path=str(DB_DIR))
    collection = db.get_collection(COLLECTION_NAME)
    data = collection.get(include=["embeddings", "documents", "metadatas"])

    ids = data["ids"]
    embeddings = np.array(data["embeddings"], dtype=np.float32)
    logger.info(f"Loaded {len(ids)} vectors of dim {embeddings.shape[1]} from Chroma")

    coords = project_to_shell(reduce_to_3d(embeddings))

    nodes = []
    for i, cid in enumerate(ids):
        meta = data["metadatas"][i]
        text = data["documents"][i]
        nodes.append({
            "id": cid,
            "doc": meta["doc_title"],
            "section": meta["section"].split(" {#")[0],  # 去掉 FAQ 標題的 anchor 雜訊
            "url": meta["doc_url"],
            "preview": text[:TEXT_PREVIEW_CHARS],
            "pos": [round(float(x), 2) for x in coords[i]],
            "emb": [round(float(x), EMB_ROUND) for x in data["embeddings"][i]],
        })

    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "embed_dim": embeddings.shape[1]}, f, ensure_ascii=False)

    size_mb = OUT_PATH.stat().st_size / 1e6
    logger.info(f"Wrote {len(nodes)} nodes -> {OUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
