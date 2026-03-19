lc-vs-rag-mk-verdicts/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── .env.example                          # API keys template (never commit .env)
│
│
├── data/
│   ├── README.md                         # explains folder structure + data sources
│   │
│   ├── raw/                              
│   │   ├── txt/
│   │   │   ├── 1_PUU-XIX_2021.txt
│   │   │   └── ...
│   │   └── json/
│   │       ├── 1_PUU-XIX_2021.json
│   │       └── ...
│   │
│   ├── processed/
│   │   ├── cleaned/                      # after text normalization (hyphen fix, CRLF, page nums)
│   │   │   └── .gitkeep
│   │   ├── sectioned/                    # per-verdict JSON with [X.X] sections extracted
│   │   │   └── .gitkeep
│   │   └── embedded/                     # FAISS index + embedding cache (gitignored)
│   │       └── .gitkeep
│   │
│   ├── metadata/
│   │   ├── corpus_stats.csv              # full 887-verdict audit output
│   │   ├── verdicts_metadata.csv         # enriched metadata (from json + text extraction)
│   │   ├── amar_mismatches.csv           # verdicts where JSON amar ≠ text amar (QC flag)
│   │   └── sample_50.csv                 # final stratified 50-verdict sample with strata labels
│   │
│   └── qa_dataset/
│       ├── schema.md                     # field definitions for all QA JSONL files
│       ├── qa_pairs_full.jsonl           # ~350 pairs, all 50 verdicts
│       ├── qa_pairs_ablation.jsonl       # 100-pair stratified subset for ablation
│       ├── qa_pairs_niah.jsonl           # 30 needle-in-a-haystack questions
│       └── qa_pairs_knowledge_update.jsonl  # 3 new verdicts × ~7 questions
│
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── audit.py                      # corpus_stats.csv generator — runs across all 887
│   │   ├── cleaner.py                    # normalize CRLF, fix PDF hyphen breaks, strip page nums
│   │   ├── section_extractor.py          # [X.X] → named sections dict per verdict
│   │   ├── metadata_enricher.py          # merge JSON metadata + text-derived fields
│   │   └── sampler.py                    # stratified 50-verdict sample from corpus_stats
│   │
│   ├── qa/
│   │   ├── __init__.py
│   │   ├── generator.py                  # LLM-assisted QA draft generation (non-evaluator model)
│   │   ├── reviewer_cli.py               # human review: accept / modify / reject per question
│   │   └── iaa_calculator.py             # inter-annotator agreement (Cohen's kappa)
│   │
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── chunkers/
│   │   │   ├── __init__.py
│   │   │   ├── fixed_size.py             # recursive character splitter (256/512/1024 tokens)
│   │   │   └── section_boundary.py       # chunk aligned to MK [X.X] section boundaries
│   │   ├── embedder.py                   # Google text-embedding-004 wrapper + batch logic
│   │   └── vector_store.py               # FAISS index build / save / load
│   │
│   ├── systems/
│   │   ├── __init__.py
│   │   ├── base.py                       # abstract QASystem: query() → ResultDict
│   │   │
│   │   ├── long_context.py               # full doc → prompt (+ LC-Windowed variant)
│   │   ├── simple_rag.py                 # FAISS top-k retrieval → prompt
│   │   │
│   │   └── advanced_rag/
│   │       ├── __init__.py
│   │       ├── pipeline.py               # orchestrates all components; respects ablation flags
│   │       ├── query_rewriter.py         # LLM-based query decomposition / rewriting
│   │       ├── reranker.py               # ms-marco-MiniLM-L-6-v2 cross-encoder
│   │       ├── hybrid_search.py          # BM25 (rank_bm25) + dense RRF fusion
│   │       └── metadata_filter.py        # pre-filter by case_type / section / date
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── faithfulness.py               # LLM-as-judge: decompose → ground each claim
│   │   ├── hallucination.py              # HR = 1 - faithfulness; binary flag at HR > 0.20
│   │   ├── bertscore_eval.py             # IndoBERT-based BERTScore F1
│   │   ├── retrieval_metrics.py          # context precision + context recall (Jaccard ≥ 0.5)
│   │   ├── legal_accuracy_cli.py         # human spot-check tool: 0/1/2 rating per response
│   │   ├── cost_tracker.py               # log input tokens → cost ($) via model price table
│   │   └── latency_tracker.py            # wall-clock per query; p50/p90/p99 reporting
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                     # all hyperparams: model names, chunk sizes, k, costs
│       ├── logger.py                     # structured JSONL run logger
│       ├── token_counter.py              # tiktoken / Gemini token estimator
│       └── text_utils.py                 # shared helpers: truncation, section joining, etc.
│
│
├── experiments/
│   ├── configs/
│   │   ├── phase1_baseline.yaml          # Gemini 2.5 Flash, all 3 architectures, full 350 QA
│   │   ├── phase2_multimodel.yaml        # Gemini 2.5 Flash vs GPT-4o, 2×3 factorial
│   │   ├── ablation.yaml                 # 6 Advanced RAG ablation conditions, 100 QA subset
│   │   ├── chunking_comparison.yaml      # fixed-size vs section-boundary, 100 QA subset
│   │   └── sensitivity.yaml              # chunk_size × top_k grid search
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── runner.py                     # core: load config → run system → log results
│   │   └── batch_evaluator.py            # evaluates a results JSONL against all metrics
│   │
│   ├── run_phase1.py                     # Phase 1: LC vs Simple RAG vs Advanced RAG
│   ├── run_phase2.py                     # Phase 2: Gemini vs GPT-4o × 3 architectures
│   ├── run_ablation.py                   # Advanced RAG component ablation
│   ├── run_niah.py                       # Needle-in-a-haystack experiment
│   ├── run_knowledge_update.py           # 3 new verdicts; update latency + QA perf
│   └── run_chunking_comparison.py        # fixed-size vs section-boundary chunking
│
│
├── results/
│   ├── README.md                         # how to read result JSONL files
│   │
│   ├── phase1/
│   │   ├── lc/
│   │   │   └── run_YYYYMMDD_HHMMSS.jsonl
│   │   ├── simple_rag/
│   │   │   └── run_YYYYMMDD_HHMMSS.jsonl
│   │   └── advanced_rag/
│   │       └── run_YYYYMMDD_HHMMSS.jsonl
│   │
│   ├── phase2/
│   │   ├── gemini_flash/
│   │   └── gpt4o/
│   │
│   ├── ablation/
│   │   ├── simple_rag_baseline/
│   │   ├── plus_query_rewrite/
│   │   ├── plus_reranking/
│   │   ├── plus_hybrid_search/
│   │   ├── plus_metadata_filter/
│   │   └── full_advanced_rag/
│   │
│   └── additional/
│       ├── niah/
│       ├── knowledge_update/
│       └── chunking_comparison/
│
│
├── notebooks/
│   ├── 00_corpus_audit.ipynb             # 887-verdict stats: length dist, jenis, amar, nulls
│   ├── 01_data_quality.ipynb             # amar mismatches, null fields, cleaning artifacts
│   ├── 02_sample_selection.ipynb         # stratified 50-verdict sample validation
│   ├── 03_qa_dataset_audit.ipynb         # QA pair distribution, question type balance
│   ├── 04_phase1_results.ipynb           # main architecture comparison + significance tests
│   ├── 05_phase2_results.ipynb           # 2×3 factorial: model × architecture
│   ├── 06_ablation_analysis.ipynb        # per-component attribution (waterfall chart)
│   ├── 07_length_sensitivity.ipynb       # metrics stratified by short/medium/long
│   ├── 08_niah_analysis.ipynb            # depth-of-needle vs retrieval/attention performance
│   ├── 09_cost_frontier.ipynb            # Pareto: input token cost vs faithfulness scatter
│   └── 10_statistical_tests.ipynb        # Wilcoxon, Kruskal-Wallis, bootstrap CIs
│
│
├── scripts/
│   ├── build_corpus.sh                   # one-shot: audit → clean → section → metadata
│   ├── build_index.sh                    # embed + build FAISS index for all 50 sample verdicts
│   ├── generate_qa_drafts.sh             # LLM-assisted QA generation for all 50 verdicts
│   └── run_all_experiments.sh            # sequential: phase1 → phase2 → ablation → additional
│
│
└── tests/
    ├── conftest.py                        # shared fixtures (sample verdict text, mock configs)
    ├── data/
    │   ├── test_cleaner.py
    │   ├── test_section_extractor.py
    │   └── test_metadata_enricher.py
    ├── systems/
    │   ├── test_long_context.py
    │   ├── test_simple_rag.py
    │   └── test_advanced_rag_pipeline.py
    └── evaluation/
        ├── test_retrieval_metrics.py
        ├── test_cost_tracker.py
        └── test_faithfulness.py