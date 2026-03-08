# langsmith_monitor.py - LangSmith 监控工具（Self-RAG增强版）
from datetime import datetime, timedelta
from typing import Dict, List
from collections import Counter as CounterClass
from langsmith import Client
from config import config


class LangSmithMonitor:
    """LangSmith 监控和统计工具（面试展示用）"""

    def __init__(self):
        self.client = None
        if config.LANGSMITH_API_KEY:
            self.client = Client(
                api_key=config.LANGSMITH_API_KEY, api_url=config.LANGSMITH_ENDPOINT
            )

    def get_recent_traces(self, limit: int = 10) -> List[Dict]:
        """获取最近的追踪记录"""
        if not self.client:
            return [{"error": "LangSmith 未配置"}]

        try:
            runs = self.client.list_runs(
                project_name=config.LANGSMITH_PROJECT,
                execution_order=1,  # 只获取根运行
                limit=limit,
            )
            return [
                {
                    "run_id": run.id,
                    "name": run.name,
                    "start_time": run.start_time.isoformat()
                    if run.start_time
                    else None,
                    "latency_ms": (run.end_time - run.start_time).total_seconds() * 1000
                    if run.end_time and run.start_time
                    else None,
                    "status": run.status,
                    "total_tokens": run.total_tokens
                    if hasattr(run, "total_tokens")
                    else None,
                }
                for run in runs
            ]
        except Exception as e:
            return [{"error": str(e)}]

    def get_node_stats(self, run_id: str) -> Dict:
        """获取特定运行的节点统计（面试时展示延迟分解）"""
        if not self.client:
            return {"error": "LangSmith 未配置"}

        try:
            run = self.client.read_run(run_id)
            child_runs = list(
                self.client.list_runs(
                    project_name=config.LANGSMITH_PROJECT, parent_run_id=run_id
                )
            )

            node_latencies = {}
            for child in child_runs:
                if child.start_time and child.end_time:
                    latency = (child.end_time - child.start_time).total_seconds() * 1000
                    node_latencies[child.name] = {
                        "latency_ms": round(latency, 2),
                        "tokens": child.total_tokens
                        if hasattr(child, "total_tokens")
                        else None,
                    }

            return {
                "total_latency_ms": (run.end_time - run.start_time).total_seconds()
                * 1000
                if run.end_time and run.start_time
                else None,
                "node_breakdown": node_latencies,
                "total_tokens": run.total_tokens
                if hasattr(run, "total_tokens")
                else None,
            }
        except Exception as e:
            return {"error": str(e)}

    def get_self_rag_stats(self, hours: int = 24) -> Dict:
        """获取Self-RAG统计（面试亮点：展示反思频率）"""
        if not self.client:
            return {"error": "LangSmith 未配置"}

        try:
            # 获取近期运行
            start_time = datetime.now() - timedelta(hours=hours)
            runs = self.client.list_runs(
                project_name=config.LANGSMITH_PROJECT,
                start_time=start_time,
                execution_order=1,
            )

            # 统计Self-RAG指标
            stats = {
                "total_queries": 0,
                "reflection_distribution": CounterClass(),  # 反思次数分布
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

                # 从metadata提取Self-RAG信息（需要在run中记录）
                meta = run.extra or {}
                reflection_count = meta.get("reflection_count", 0)
                stats["reflection_distribution"][reflection_count] += 1

                # 记录延迟（区分有无反思）
                latency = run.latency_ms or 0
                if reflection_count > 0:
                    stats["latency_with_reflection"].append(latency)
                else:
                    stats["latency_without_reflection"].append(latency)

                # 检索质量分布
                grade = meta.get("retrieval_grade")
                if grade in stats["grade_distribution"]:
                    stats["grade_distribution"][grade] += 1

                # 生成支持度分布
                support = meta.get("support_grade")
                if support in stats["support_distribution"]:
                    stats["support_distribution"][support] += 1

            # 计算平均值
            if stats["latency_with_reflection"]:
                stats["avg_latency_with_reflection"] = sum(
                    stats["latency_with_reflection"]
                ) / len(stats["latency_with_reflection"])
            else:
                stats["avg_latency_with_reflection"] = 0

            if stats["latency_without_reflection"]:
                stats["avg_latency_without_reflection"] = sum(
                    stats["latency_without_reflection"]
                ) / len(stats["latency_without_reflection"])
            else:
                stats["avg_latency_without_reflection"] = 0

            # 转换为普通dict以便JSON序列化
            stats["reflection_distribution"] = dict(stats["reflection_distribution"])

            return stats

        except Exception as e:
            return {"error": str(e)}

    def print_self_rag_report(self):
        """打印Self-RAG专项报告（面试展示用）"""
        print("\n" + "=" * 60)
        print("🔄 Self-RAG 架构性能报告")
        print("=" * 60)

        stats = self.get_self_rag_stats(hours=24)
        if "error" in stats:
            print(f"⚠️ 获取失败: {stats['error']}")
            return

        total = stats["total_queries"]
        if total == 0:
            print("暂无数据")
            return

        print("\n📊 查询统计（近24小时）:")
        print(f"   总查询数: {total}")

        print("\n🔄 反思迭代分布:")
        ref_dist = stats["reflection_distribution"]
        for count in sorted(ref_dist.keys()):
            num = ref_dist[count]
            bar = "█" * (num // max(1, total // 20))
            pct = num / total * 100
            print(f"   {count}轮反思: {num}次 ({pct:.1f}%) {bar}")

        print("\n📈 检索质量分布:")
        for grade, num in stats["grade_distribution"].items():
            pct = (num / total * 100) if total else 0
            icon = (
                "✅"
                if grade == "highly_relevant"
                else "🔄"
                if grade == "partially_relevant"
                else "⚠️"
            )
            print(f"   {icon} {grade}: {num} ({pct:.1f}%)")

        print("\n🛡️ 生成支持度（幻觉检测）:")
        support_total = sum(stats["support_distribution"].values())
        for support, num in stats["support_distribution"].items():
            pct = num / max(1, support_total) * 100
            icon = (
                "✅"
                if support == "fully_supported"
                else "⚠️"
                if support == "partially_supported"
                else "🚨"
            )
            print(f"   {icon} {support}: {num} ({pct:.1f}%)")

        print("\n⏱️ 延迟分析:")
        avg_with = stats["avg_latency_with_reflection"]
        avg_without = stats["avg_latency_without_reflection"]
        print(f"   无反思: {avg_without:.0f}ms")
        print(f"   有反思: {avg_with:.0f}ms")
        if avg_with > 0 and avg_without > 0:
            overhead = avg_with - avg_without
            print(
                f"   反思开销: +{overhead:.0f}ms ({overhead / avg_without * 100:.1f}%)"
            )

        print("=" * 60)

    def print_performance_report(self):
        """打印性能报告（面试时直接展示）"""
        print("=" * 60)
        print("🔍 EnterpriseMind 性能监控报告")
        print("=" * 60)

        traces = self.get_recent_traces(limit=5)
        if not traces or "error" in traces[0]:
            print(f"⚠️ 获取追踪失败: {traces[0].get('error', '未知错误')}")
            return

        print(f"\n最近 {len(traces)} 次查询统计:\n")

        total_latency = 0
        total_tokens = 0
        valid_count = 0

        for trace in traces:
            if trace.get("latency_ms"):
                total_latency += trace["latency_ms"]
                valid_count += 1
            if trace.get("total_tokens"):
                total_tokens += trace["total_tokens"]

            status_icon = "✅" if trace.get("status") == "success" else "❌"
            print(
                f"{status_icon} {trace['name'][:30]:30} | "
                f"延迟: {trace.get('latency_ms', 'N/A'):>8.0f}ms | "
                f"Token: {trace.get('total_tokens', 'N/A'):>6}"
            )

        if valid_count > 0:
            print(f"\n平均延迟: {total_latency / valid_count:.0f}ms")
            print(f"平均Token: {total_tokens / len(traces):.0f}")

        print("\n📊 追踪面板: https://smith.langchain.com")
        print("=" * 60)


# 便捷函数
def show_monitor():
    """快速展示监控（用于面试演示）"""
    monitor = LangSmithMonitor()
    monitor.print_performance_report()


def show_self_rag_monitor():
    """快速展示Self-RAG监控"""
    monitor = LangSmithMonitor()
    monitor.print_self_rag_report()


if __name__ == "__main__":
    show_monitor()
