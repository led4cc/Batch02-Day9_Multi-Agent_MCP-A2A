# RAG Evaluation Results

## Framework

Local lexical-overlap evaluator. This avoids paid/cloud evaluators and runs on the same supervisor-agent RAG pipeline used by the demo app.

## Overall Scores

| Metric | Config A: supervisor + rerank | Config B: supervisor no rerank | Delta |
|--------|----------------------------|----------------------------|-------|
| Faithfulness | 99.6% | 99.6% | -0.000 |
| Answer Relevance | 44.0% | 42.4% | +0.016 |
| Context Recall | 39.1% | 41.3% | -0.022 |
| Context Precision | 25.9% | 23.2% | +0.027 |
| Average | 52.2% | 51.6% | +0.005 |

## A/B Comparison Analysis

**Config A:** supervisor retrieval with local reranking.

**Config B:** supervisor retrieval without reranking.

**Conclusion:** Config A is preferred when its average score is equal or higher because reranking improves source ordering for generation. Config B is useful as a lower-latency fallback.

## Worst Performers

| # | Question | Average | Likely Root Cause |
|---|----------|---------|-------------------|
| 1 | Pipeline RAG nen dung source documents nhu the nao khi tra loi? | 30.8% | Query/context lexical mismatch or source text encoding noise |
| 2 | Nhung hinh thuc cai nghien ma tuy nao duoc quy dinh trong Luat Phong chong ma tuy? | 41.9% | Query/context lexical mismatch or source text encoding noise |
| 3 | Kinh phi cho hoat dong cai nghien ma tuy duoc de cap nhu the nao? | 43.6% | Query/context lexical mismatch or source text encoding noise |

## Recommendations

1. Fix text encoding in crawled/news documents before indexing.
2. Replace hashing embeddings with a Vietnamese/multilingual sentence-transformer model when network/model download is available.
3. Add a citation post-processor that verifies every answer sentence has one of the retrieved source filenames.
