# Go Docs RAG — 中文問答 × Go 官方文件

用繁體中文提問，從 Go 官方英文文件中檢索依據、生成有來源標注的中文答案。一個完整的 RAG（Retrieval-Augmented Generation）管線實作：文件擷取、切塊、向量索引、語意檢索、grounded 生成與雙層拒答機制。
**🌌 Live Demo:** [互動式 3D 向量空間視覺化](https://rocklai1207.github.io/go-docs-rag/viz/viz.html) — 在瀏覽器直接體驗檢索流程
```
$ python ask.py "goroutine 和 channel 怎麼搭配使用?"

Goroutine 和 channel 搭配使用主要有以下幾種方式：
1. 同步與訊號傳遞：...
2. 通訊與資料交換：...
3. 資源管理與流量限制（Semaphore）：...
來源：段落 1, 段落 3

--- 檢索到的段落（依相關度排序）---
  [1] Effective Go › Parallelization  (distance=0.251)
  [2] Effective Go › A leaky buffer  (distance=0.258)
  ...
```

## The Problem

台灣工程師讀英文技術文件有真實的語言摩擦，而直接問通用 LLM 又有幻覺風險——答案聽起來對，但無法驗證出處。這個系統讓你用中文提問，答案完全錨定在 Go 官方文件上，並附上可追溯的來源連結。

## Architecture

```
┌─────────────────────┐
│  Go 官方文件 (5 份)   │  Effective Go / FAQ / How to Write
│  golang/website repo │  Go Code / go.mod Ref / Doc Comments
└─────────┬───────────┘
          │ ① ingest.py — 抓取、清理、切塊
          ▼
┌─────────────────────┐
│  chunks.json         │  247 chunks（標題結構優先切分
│  (text + metadata)   │  + 長度上限 + 200 字元 overlap）
└─────────┬───────────┘
          │ ② embed_index.py — Gemini embedding（批次/斷點續跑/429 退避）
          ▼
┌─────────────────────┐
│  Chroma (cosine)     │  向量 + 原文 + 來源 metadata
└─────────┬───────────┘
          │ ③ ask.py — RETRIEVAL_QUERY embedding → Top-K 檢索
          ▼
┌─────────────────────┐      ┌──────────────────────────┐
│  相關度門檻 (L1 拒答)  │─No──▶│ 「文件中沒有相關內容」      │
└─────────┬───────────┘      └──────────────────────────┘
          │ Yes
          ▼
┌─────────────────────┐
│  Gemini 生成          │  temperature=0.2、只依參考段落回答、
│  (L2 拒答 in prompt)  │  繁體中文輸出、標注引用來源
└─────────────────────┘
```

## Key Technical Decisions

| 決策 | 理由 |
|---|---|
| 切塊：標題結構優先、長度上限為輔 | 文件章節本身是語意完整單位；超長章節在段落邊界切分並保留 200 字元 overlap，避免關鍵句被切線截斷 |
| chunk 上限 2000 字元（約 500 tokens） | 太大則單塊混入多主題、向量語意糊掉；太小則上下文不足。以檢索實測校準 |
| embedding 區分 task_type | `RETRIEVAL_DOCUMENT`（建索引）與 `RETRIEVAL_QUERY`（查詢）分開編碼，是 Gemini embedding 對「問題↔文件」不對稱性的內建優化 |
| 雙層拒答 | L1 距離門檻擋「完全無關」（省生成額度）；L2 prompt 規則擋「語意相近但答不了」——實測 "Python GIL" 一題與 Go 並發章節 distance 僅 0.40，正是 L2 的存在理由 |
| 生成 temperature=0.2 | 問答要收斂貼合資料；對照另一個創意生成專案的 0.9，同一參數在不同場景的取捨 |
| 斷點續跑 + 批次 + 429 退避 | 全程可在 Gemini 免費額度內完成；中斷重跑自動跳過已入庫的 chunk |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # 填入免費申請的 Gemini API key

python ingest.py          # ① 抓文件、切塊（純標準庫，無需 key）
python embed_index.py     # ② 建向量索引 + 檢索冒煙測試
python ask.py             # ③ 互動問答；或 python ask.py "你的問題"
```

API key 免費申請：[Google AI Studio](https://aistudio.google.com/apikey)（無需信用卡）。

## Project Structure

```
├── ingest.py         # ① 文件擷取與切塊
├── embed_index.py    # ② embedding 與向量索引
├── ask.py            # ③ 檢索 + 生成問答
├── requirements.txt
├── .env.example
└── data/             # 生成物（gitignored）
    ├── chunks.json
    └── chroma_db/
```

## Honest Limitations & Roadmap

- 語料庫僅涵蓋 5 份核心文件，不含標準庫 API reference — 範圍外問題會誠實拒答
- [ ] 擴充語料至 Go blog 與標準庫文件
- [ ] 檢索品質評估集（golden questions）與自動化回歸測試
- [ ] Hybrid search（向量 + BM25 關鍵字）改善專有名詞檢索
- [ ] Web UI
