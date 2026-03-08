# EnterpriseMind

个人企业级多 Agent RAG 开发项目。  
包含检索问答、SQL 查询、联网搜索、计算工具，并支持缓存与查询优化 benchmark。

## 功能概览

- Supervisor 多 Agent 路由（`retrieval / sql / search / calculation`）
- 企业知识库检索（向量 + BM25 + RRF + 可选重排）
- SQL 安全执行（只读查询、危险语句拦截）
- 可选 Self-RAG 链路（可关闭做对照实验）
- 缓存机制（内存 + 持久化）
- Gradio Web 交互界面 + FastAPI 服务接口

## 项目结构

```text
app.py                # Gradio 入口
server.py             # FastAPI 入口
graph/                # 多 Agent 编排与节点
rag/                  # 检索、查询优化、评估逻辑
tools/                # SQL/搜索/计算工具
data/                 # 示例业务数据
benchmarks/           # benchmark 数据与计算脚本
docker-compose.yml    # 本地一键编排（app + postgres）
Dockerfile            # 容器镜像构建
```

## 环境要求

- Python 3.11+
- PostgreSQL 15（本地或 Docker）
- 可用的 LLM（支持双模式：DeepSeek API / Ollama）

## 运行模式（双模式）

### 1. Local 模式（DeepSeek API）

适用：本地开发、无需自建模型服务。

`.env` 关键配置：

- `RUN_MODE=local`
- `DEEPSEEK_API_KEY=...`
- `DEEPSEEK_MODEL=deepseek-chat`
- `DATABASE_URL=postgresql://postgres:123456@localhost:5432/enterprisemind`
- `ENABLE_SELF_RAG=true`
- `ENABLE_QUERY_OPTIMIZATION=true`

### 2. Cloud 模式（Ollama）

适用：云端部署或私有模型推理。

`.env` 关键配置：

- `RUN_MODE=cloud`
- `OLLAMA_BASE_URL=http://<your-host>:11434`
- `OLLAMA_MODEL=deepseek-r1:14b`
- `OLLAMA_EMBEDDING_MODEL=bge-m3`
- `DATABASE_URL=postgresql://postgres:123456@<db-host>:5432/enterprisemind`
- `ENABLE_SELF_RAG=true`
- `ENABLE_QUERY_OPTIMIZATION=true`

## 启动方式

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 按上面的模式配置 `.env`

3. 初始化数据库

```bash
psql -U postgres -d enterprisemind -f init_db.sql
```

4. 启动界面

```bash
python app.py
```

5. 启动 API（可选）

```bash
uvicorn server:api --host 0.0.0.0 --port 8000
```

## Docker 启动（同样支持双模式）

在 `.env` 设置对应模式后执行：

```bash
docker compose up --build
```

- App: `http://localhost:7860`
- PostgreSQL: `localhost:5432`

## Benchmark

`benchmarks/` 目录已提供模拟测试数据：

- `benchmark_30_samples.csv`：30 条评测问题
- `simulated_perf_runs.csv`：缓存性能模拟数据
- `simulated_rag_scores.csv`：检索效果模拟数据
- `calc_benchmark.py`：自动计算提升率

运行：

```bash
python benchmarks/calc_benchmark.py --perf benchmarks/simulated_perf_runs.csv --rag benchmarks/simulated_rag_scores.csv
```

当前模拟结果：

- 缓存优化：`P95 延迟 -37.0%`，`Token 成本 -23.0%`
- 查询优化：`Recall@5 +15.0%`，`Answer F1 +16.8%`，`幻觉率 -66.7%`
