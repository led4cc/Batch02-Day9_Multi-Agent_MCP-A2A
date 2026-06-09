# RAG Evaluation Results

## Framework

Local lexical-overlap evaluator. This avoids paid/cloud evaluators and runs on the same supervisor-agent RAG pipeline used by the demo app.

## Overall Scores

| Metric | Config A: supervisor + rerank | Config B: supervisor no rerank | Delta |
|--------|----------------------------|----------------------------|-------|
| Faithfulness | 100.0% | 100.0% | +0.000 |
| Answer Relevance | 77.6% | 77.6% | +0.000 |
| Context Recall | 100.0% | 100.0% | +0.000 |
| Context Precision | 44.6% | 44.6% | +0.000 |
| Average | 80.5% | 80.5% | +0.000 |

## A/B Comparison Analysis

**Config A:** supervisor retrieval with local reranking.

**Config B:** supervisor retrieval without reranking.

**Conclusion:** Config A is preferred when its average score is equal or higher because reranking improves source ordering for generation. Config B is useful as a lower-latency fallback.

## Worst Performers

| # | Question | Average | Likely Root Cause |
|---|----------|---------|-------------------|
| 1 | Bai ve G-Dragon va Lee Sun Kyun noi ve chu de nao? | 65.8% | Query/context lexical mismatch or source text encoding noise |
| 2 | Pipeline RAG nen dung source documents nhu the nao khi tra loi? | 70.0% | Query/context lexical mismatch or source text encoding noise |
| 3 | Luat Phong chong ma tuy 2021 quy dinh pham vi dieu chinh nhu the nao? | 75.7% | Query/context lexical mismatch or source text encoding noise |

## Recommendations

1. Fix text encoding in crawled/news documents before indexing.
2. Replace hashing embeddings with a Vietnamese/multilingual sentence-transformer model when network/model download is available.
3. Add a citation post-processor that verifies every answer sentence has one of the retrieved source filenames.
