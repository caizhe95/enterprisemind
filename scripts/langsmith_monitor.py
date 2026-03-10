# langsmith_monitor.py - LangSmith 轻量监控脚本
from datetime import datetime, timedelta
from collections import Counter
from typing import Dict, List

from langsmith import Client

from config import config


class LangSmithMonitor:
    """LangSmith 轻量监控工具。"""

    def __init__(self):
        self.client = None
        if config.LANGSMITH_API_KEY:
            self.client = Client(
                api_key=config.LANGSMITH_API_KEY,
                api_url=config.LANGSMITH_ENDPOINT,
            )

    def get_recent_traces(self, limit: int = 10) -> List[Dict]:
        """获取最近的根运行追踪。"""
        if not self.client:
            return [{"error": "LangSmith 未配置"}]

        try:
            runs = self.client.list_runs(
                project_name=config.LANGSMITH_PROJECT,
                execution_order=1,
                limit=limit,
            )
            return [
                {
                    "run_id": run.id,
                    "name": run.name,
                    "start_time": run.start_time.isoformat()
                    if run.start_time
                    else None,
                    "latency_ms": (run.end_time - run.start_time).total_seconds()
                    * 1000
                    if run.end_time and run.start_time
                    else None,
                    "status": run.status,
                    "total_tokens": getattr(run, "total_tokens", None),
                }
                for run in runs
            ]
        except Exception as e:
            return [{"error": str(e)}]

    def get_self_rag_stats(self, hours: int = 24) -> Dict:
        """统计近期 Self-RAG 关键指标。"""
        if not self.client:
            return {"error": "LangSmith 未配置"}

        try:
            start_time = datetime.now() - timedelta(hours=hours)
            runs = self.client.list_runs(
                project_name=config.LANGSMITH_PROJECT,
                start_time=start_time,
                execution_order=1,
            )

            stats = {
                "total_queries": 0,
                "reflection_distribution": Counter(),
                "grade_distribution": {
                    "highly_relevant": 0,
                    "partially_relevant": 0,
                    "irrelevant": 0,
                },
                "support_distribution": {
                    "fully_supported": 0,
                    "partially_supported": 0,
                    "no_support": 0,
                },
                "latency_with_reflection": [],
                "latency_without_reflection": [],
            }

            for run in runs:
                if not run.inputs:
                    continue

                stats["total_queries"] += 1
                meta = run.extra or {}
                reflection_count = meta.get("reflection_count", 0)
                stats["reflection_distribution"][reflection_count] += 1

                latency = run.latency_ms or 0
                target = (
                    "latency_with_reflection"
                    if reflection_count > 0
                    else "latency_without_reflection"
                )
                stats[target].append(latency)

                grade = meta.get("retrieval_grade")
                if grade in stats["grade_distribution"]:
                    stats["grade_distribution"][grade] += 1

                support = meta.get("support_grade")
                if support in stats["support_distribution"]:
                    stats["support_distribution"][support] += 1

            stats["avg_latency_with_reflection"] = _avg(
                stats["latency_with_reflection"]
            )
            stats["avg_latency_without_reflection"] = _avg(
                stats["latency_without_reflection"]
            )
            stats["reflection_distribution"] = dict(stats["reflection_distribution"])
            return stats
        except Exception as e:
            return {"error": str(e)}

    def print_report(self, mode: str = "performance", limit: int = 5, hours: int = 24):
        """统一打印监控报告。"""
        if mode == "self_rag":
            self._print_self_rag_report(hours=hours)
            return
        self._print_performance_report(limit=limit)

    def _print_performance_report(self, limit: int = 5):
        print("=" * 60)
        print("🔍 智能导购多 Agent 系统性能监控报告")
        print("=" * 60)

        traces = self.get_recent_traces(limit=limit)
        if not traces or "error" in traces[0]:
            print(f"⚠️ 获取追踪失败: {traces[0].get('error', '未知错误')}")
            return

        total_latency = 0
        total_tokens = 0
        valid_latency_count = 0

        for trace in traces:
            latency = trace.get("latency_ms")
            tokens = trace.get("total_tokens")
            if latency is not None:
                total_latency += latency
                valid_latency_count += 1
            if tokens:
                total_tokens += tokens

            status_icon = "✅" if trace.get("status") == "success" else "❌"
            latency_text = f"{latency:.0f}ms" if latency is not None else "N/A"
            print(
                f"{status_icon} {trace['name'][:30]:30} | "
                f"延迟: {latency_text:>8} | "
                f"Token: {tokens if tokens is not None else 'N/A':>6}"
            )

        if valid_latency_count > 0:
            print(f"\n平均延迟: {total_latency / valid_latency_count:.0f}ms")
            print(f"平均Token: {total_tokens / len(traces):.0f}")

        print("\n📊 追踪面板: https://smith.langchain.com")
        print("=" * 60)

    def _print_self_rag_report(self, hours: int = 24):
        print("\n" + "=" * 60)
        print("🔄 Self-RAG 架构性能报告")
        print("=" * 60)

        stats = self.get_self_rag_stats(hours=hours)
        if "error" in stats:
            print(f"⚠️ 获取失败: {stats['error']}")
            return

        total = stats["total_queries"]
        if total == 0:
            print("暂无数据")
            return

        print(f"\n📊 查询统计（近{hours}小时）:")
        print(f"   总查询数: {total}")

        print("\n🔄 反思迭代分布:")
        for count in sorted(stats["reflection_distribution"].keys()):
            num = stats["reflection_distribution"][count]
            pct = num / total * 100
            print(f"   {count}轮反思: {num}次 ({pct:.1f}%)")

        print("\n📈 检索质量分布:")
        for grade, num in stats["grade_distribution"].items():
            pct = num / total * 100 if total else 0
            print(f"   {grade}: {num} ({pct:.1f}%)")

        print("\n🛡️ 生成支持度:")
        support_total = sum(stats["support_distribution"].values())
        for support, num in stats["support_distribution"].items():
            pct = num / max(1, support_total) * 100
            print(f"   {support}: {num} ({pct:.1f}%)")

        avg_with = stats["avg_latency_with_reflection"]
        avg_without = stats["avg_latency_without_reflection"]
        print("\n⏱️ 延迟分析:")
        print(f"   无反思: {avg_without:.0f}ms")
        print(f"   有反思: {avg_with:.0f}ms")
        if avg_with > 0 and avg_without > 0:
            overhead = avg_with - avg_without
            print(f"   反思开销: +{overhead:.0f}ms")

        print("=" * 60)


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0


def show_monitor(mode: str = "performance"):
    """便捷入口。"""
    monitor = LangSmithMonitor()
    monitor.print_report(mode=mode)


if __name__ == "__main__":
    show_monitor()
