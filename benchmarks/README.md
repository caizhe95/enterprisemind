# Benchmark Notes

This folder contains a fixed 30-sample benchmark set and simulated result files for interview demos.

Run:

```bash
python benchmarks/calc_benchmark.py --perf benchmarks/simulated_perf_runs.csv --rag benchmarks/simulated_rag_scores.csv
```

Current simulated summary:

- Cache optimization: P95 latency `80.26s -> 50.56s` (`-37.0%`)
- Cache optimization: average total tokens `1016.7 -> 782.4` (`-23.0%`)
- Self-RAG: `Recall@5 0.667 -> 0.767` (`+15.0%`)
- Self-RAG: `Answer F1 0.713 -> 0.833` (`+16.8%`)
- Self-RAG: hallucination `0.100 -> 0.033` (`-66.7%`)
