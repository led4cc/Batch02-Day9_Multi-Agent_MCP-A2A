# Bai Tap Nhom - Supervisor Agent RAG Chatbot

## Muc tieu

Xay dung RAG chatbot tra loi cau hoi ve phap luat Viet Nam lien quan den ma tuy va cac bai bao lien quan. Phien ban nay duoc toi uu theo supervisor agent architecture: supervisor dieu phoi router, retrieval, generation va citation verifier.

## San pham da thuc hien

- Chatbot UI bang Streamlit: `lab_assignment/app.py`
- Supervisor agent orchestration: `lab_assignment/supervisor_agents.py`
- Retrieval pipeline: tu dong dung `src/task9_retrieval_pipeline.py` neu co, fallback sang local lexical retrieval neu chua co `src/`
- Generation co citation: tu dong dung `src/task10_generation.py` neu co, fallback sang deterministic answer tu source documents
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
  +--> Generation Agent: dung Task 10/Ollama neu co, fallback answer tu context
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
- Co che fallback neu Ollama hoac thu muc `src/` chua san sang

Chay chatbot:

```bash
streamlit run app.py
```

Hoac:

```bash
streamlit run lab_assignment/app.py
```

## Ollama setup

Task 10 khong dung OpenAI. Can chay Ollama local:

```bash
ollama pull llama3.1
ollama serve
```

Trong `.env`:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
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

`group_project/evaluation/golden_dataset.json` co 15 cau hoi, bao phu:

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

1. Cai dependencies:

```bash
pip install -r requirements.txt
```

2. Tao du lieu neu can:

```bash
python src/task2_crawl_news.py
python src/task3_convert_markdown.py
python src/task4_chunking_indexing.py
```

3. Chay Ollama:

```bash
ollama serve
```

4. Chay chatbot:

```bash
streamlit run app.py
```

5. Hoi thu:

```text
Luat Phong chong ma tuy 2021 quy dinh nhung hinh thuc cai nghien nao?
```

```text
Bai bao ve ca si Long Nhat va Son Ngoc Minh de cap den viec gi?
```

6. Chay evaluation:

```bash
python group_project/evaluation/eval_pipeline.py
```

## Luu y ky thuat

- News text hien co mot so van de encoding tu crawl/convert, nen report co recommendation lam sach du lieu.
- Embedding hien la local hashing embedding de chay offline; khi co dieu kien co the thay bang `BAAI/bge-m3`.
- PageIndex task dang duoc mo phong bang vectorless local BM25 fallback, co gan `source="pageindex"` de pipeline phan biet fallback.
