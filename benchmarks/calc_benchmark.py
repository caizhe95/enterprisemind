import argparse
import csv
import statistics
from collections import defaultdict


def pick(row, *keys, default=""):
    for k in keys:
        if k in row:
            return row[k]
    return default


def to_float(v):
    try:
        return float(v)
    except Exception:
        return 0.0


def percentile(values, p):
    if not values:
        return 0.0
    vals = sorted(values)
    k = (len(vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return vals[f]
    return vals[f] * (c - k) + vals[c] * (k - f)


def pct_improve(old, new):
    if old == 0:
        return 0.0
    return (old - new) / old * 100.0


def pct_increase(old, new):
    if old == 0:
        return 0.0
    return (new - old) / old * 100.0


def load_perf(path):
    bucket = defaultdict(lambda: {"latency": [], "tokens": [], "cache_hit": []})
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            mode = pick(row, "mode", "实验组别").strip()
            bucket[mode]["latency"].append(
                to_float(pick(row, "latency_ms", "延迟毫秒"))
            )
            bucket[mode]["tokens"].append(
                to_float(pick(row, "total_tokens", "总Token数"))
            )
            bucket[mode]["cache_hit"].append(
                to_float(pick(row, "cache_hit", "缓存命中"))
            )
    return bucket


def load_rag(path):
    bucket = defaultdict(
        lambda: {"recall": [], "mrr": [], "f1": [], "faith": [], "hall": []}
    )
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            mode = pick(row, "mode", "实验组别").strip()
            bucket[mode]["recall"].append(
                to_float(pick(row, "recall_at_5", "Recall@5"))
            )
            bucket[mode]["mrr"].append(to_float(pick(row, "mrr_at_5", "MRR@5")))
            bucket[mode]["f1"].append(to_float(pick(row, "answer_f1", "答案F1")))
            bucket[mode]["faith"].append(to_float(pick(row, "faithfulness", "忠实度")))
            bucket[mode]["hall"].append(to_float(pick(row, "hallucination", "幻觉率")))
    return bucket


def summarize_perf(perf):
    out = {}
    for mode, vals in perf.items():
        out[mode] = {
            "p50_latency": percentile(vals["latency"], 0.50),
            "p95_latency": percentile(vals["latency"], 0.95),
            "avg_tokens": statistics.fmean(vals["tokens"]) if vals["tokens"] else 0.0,
            "cache_hit_rate": statistics.fmean(vals["cache_hit"])
            if vals["cache_hit"]
            else 0.0,
            "samples": len(vals["latency"]),
        }
    return out


def summarize_rag(rag):
    out = {}
    for mode, vals in rag.items():
        out[mode] = {
            "recall_at_5": statistics.fmean(vals["recall"]) if vals["recall"] else 0.0,
            "mrr_at_5": statistics.fmean(vals["mrr"]) if vals["mrr"] else 0.0,
            "answer_f1": statistics.fmean(vals["f1"]) if vals["f1"] else 0.0,
            "faithfulness": statistics.fmean(vals["faith"]) if vals["faith"] else 0.0,
            "hallucination": statistics.fmean(vals["hall"]) if vals["hall"] else 0.0,
            "samples": len(vals["recall"]),
        }
    return out


def print_block(title, metrics):
    print(f"\n=== {title} ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute 30-sample benchmark improvements"
    )
    parser.add_argument("--perf", required=True, help="Path to perf CSV")
    parser.add_argument("--rag", required=True, help="Path to rag CSV")
    args = parser.parse_args()

    perf = summarize_perf(load_perf(args.perf))
    rag = summarize_rag(load_rag(args.rag))

    required_perf_modes = {"G0", "G1"}
    if not required_perf_modes.issubset(set(perf.keys())):
        raise ValueError("perf CSV must include G0 and G1")

    required_rag_modes = {"G1", "G2"}
    if not required_rag_modes.issubset(set(rag.keys())):
        raise ValueError("rag CSV must include G1 and G2")

    g0 = perf["G0"]
    g1 = perf["G1"]

    cache_improvement = {
        "P95 latency reduction % (G0->G1)": pct_improve(
            g0["p95_latency"], g1["p95_latency"]
        ),
        "Avg token reduction % (G0->G1)": pct_improve(
            g0["avg_tokens"], g1["avg_tokens"]
        ),
        "Cache hit rate (G1)": g1["cache_hit_rate"],
    }

    g1_r = rag["G1"]
    g2_r = rag["G2"]
    self_rag_improvement = {
        "Recall@5 improvement % (G1->G2)": pct_increase(
            g1_r["recall_at_5"], g2_r["recall_at_5"]
        ),
        "MRR@5 improvement % (G1->G2)": pct_increase(
            g1_r["mrr_at_5"], g2_r["mrr_at_5"]
        ),
        "Answer F1 improvement % (G1->G2)": pct_increase(
            g1_r["answer_f1"], g2_r["answer_f1"]
        ),
        "Faithfulness improvement % (G1->G2)": pct_increase(
            g1_r["faithfulness"], g2_r["faithfulness"]
        ),
        "Hallucination reduction % (G1->G2)": pct_improve(
            g1_r["hallucination"], g2_r["hallucination"]
        ),
    }

    print_block("Raw Perf Summary", perf)
    print_block("Raw RAG Summary", rag)
    print_block("Cache Benchmark Result", cache_improvement)
    print_block("Self-RAG Benchmark Result", self_rag_improvement)

    print("\n=== Resume Lines (Template) ===")
    print(
        "Built a 30-sample benchmark; reduced P95 latency by "
        f"{cache_improvement['P95 latency reduction % (G0->G1)']:.1f}% and token cost by "
        f"{cache_improvement['Avg token reduction % (G0->G1)']:.1f}% via cache optimization."
    )
    print(
        "Improved retrieval quality with Self-RAG: Recall@5 +"
        f"{self_rag_improvement['Recall@5 improvement % (G1->G2)']:.1f}%, "
        f"Answer F1 +{self_rag_improvement['Answer F1 improvement % (G1->G2)']:.1f}%, "
        f"hallucination rate -{self_rag_improvement['Hallucination reduction % (G1->G2)']:.1f}%."
    )


if __name__ == "__main__":
    main()
