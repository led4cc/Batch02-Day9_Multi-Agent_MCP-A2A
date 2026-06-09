# Bai Tap Nhom - Supervisor Agent RAG Chatbot

## Muc tieu

Xay dung RAG chatbot tra loi cau hoi ve phap luat Viet Nam lien quan den ma tuy va cac bai bao lien quan. Phien ban nay duoc toi uu theo supervisor agent architecture: supervisor dieu phoi router, retrieval, generation va citation verifier.

## San pham da thuc hien

- Chatbot UI bang Streamlit: `lab_assignment/app.py`
- Supervisor agent orchestration: `lab_assignment/supervisor_agents.py`
- Retrieval pipeline: tu dong dung `lab_assignment/src/task9_retrieval_pipeline.py`, fallback sang local lexical retrieval neu pipeline chua san sang
- Generation co citation: dung OpenRouter trong `lab_assignment/src/task10_generation.py`, fallback sang deterministic answer tu source documents neu API chua san sang
- Golden dataset 15 cau hoi: `lab_assignment/evaluation/golden_dataset.json`
- Evaluation script: `lab_assignment/evaluation/eval_pipeline.py`
- Evaluation report: `lab_assignment/evaluation/results.md`

## Kien truc he thong

```text
User
  |
  v
Streamlit Chat UI
  |
  v
Supervisor Agent
  |
  +--> Router Agent: phan loai legal/news/RAG meta question
  +--> Retrieval Agent: dung Task 9 neu co, fallback lexical retrieval neu can
  +--> Generation Agent: dung Task 10/OpenRouter neu co, fallback answer tu context
  +--> Citation Verifier Agent: dam bao answer co source/citation
  |
  v
Answer + citations + source documents
```

## Chatbot

Chatbot dap ung yeu cau:

- Giao dien chat bang Streamlit
- Conversation memory bang `st.session_state`
- Goi SupervisorAgent de dieu phoi cac agent con
- Co supervisor trace de debug route/retrieval/generation/citation
- Hien thi source documents, score, retrieval source va evidence preview
- Co che fallback neu OpenRouter hoac retrieval pipeline chua san sang

Chay chatbot:

```bash
uv run streamlit run lab_assignment/app.py
```

## OpenRouter setup

Task 10 dung OpenRouter qua OpenAI-compatible API. Khong can chay local LLM server.

Trong `.env`:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_MAX_TOKENS=1000
```

## Evaluation Pipeline

Framework su dung: local offline evaluator. Ly do: chay duoc trong moi truong lop hoc khong can API key, van bam 4 metric bat buoc trong README:

- Faithfulness
- Answer Relevance
- Context Recall
- Context Precision

Hai cau hinh A/B:

- Config A: hybrid semantic + lexical + RRF + reranking
- Config B: hybrid semantic + lexical + RRF, khong reranking

Chay evaluation:

```bash
python lab_assignment/evaluation/eval_pipeline.py
```

Ket qua duoc ghi vao:

```text
lab_assignment/evaluation/results.md
```

## Golden Dataset

`lab_assignment/evaluation/golden_dataset.json` co 15 cau hoi, bao phu:

- Luat Phong chong ma tuy 2021
- Nghi dinh 105/2021
- Nghi dinh 28/2026
- Tin bai ve nghe si va ma tuy trong corpus da crawl
- Cau hoi ve pipeline RAG, citation va source documents

## Phan cong cong viec

| Thanh vien | MSSV | Nhiem vu | Trang thai |
|-----------|------|----------|------------|
| Thanh vien 1 | TBD | Thu thap legal/news data | Hoan thanh |
| Thanh vien 2 | TBD | Convert markdown va chunk/index | Hoan thanh |
| Thanh vien 3 | TBD | Semantic, lexical search va reranking | Hoan thanh |
| Thanh vien 4 | TBD | Retrieval pipeline va fallback | Hoan thanh |
| Thanh vien 5 | TBD | Streamlit chatbot UI | Hoan thanh |
| Thanh vien 6 | TBD | Golden dataset, evaluation, report | Hoan thanh |

## Huong dan demo

1. Cai dependencies bang uv:

```bash
uv sync
```

2. Tao du lieu neu can:

```bash
uv run python lab_assignment/src/task2_crawl_news.py
uv run python lab_assignment/src/task3_convert_markdown.py
uv run python lab_assignment/src/task4_chunking_indexing.py
```

3. Chay chatbot:

```bash
uv run streamlit run lab_assignment/app.py
```

4. Hoi thu:

```text
Luat Phong chong ma tuy 2021 quy dinh nhung hinh thuc cai nghien nao?
```

```text
Bai bao ve ca si Long Nhat va Son Ngoc Minh de cap den viec gi?
```

5. Chay evaluation:

```bash
uv run python lab_assignment/evaluation/eval_pipeline.py
```

## Luu y ky thuat

- News text hien co mot so van de encoding tu crawl/convert, nen report co recommendation lam sach du lieu.
- Embedding hien la local hashing embedding de chay offline; khi co dieu kien co the thay bang `BAAI/bge-m3`.
- PageIndex task dang duoc mo phong bang vectorless local BM25 fallback, co gan `source="pageindex"` de pipeline phan biet fallback.
