# EnterpriseMind

EnterpriseMind 是一个面向智能导购场景的多 Agent 大模型应用后端。它围绕检索、抽取、推荐、比较和 SQL 分析构建显式工作流，而不是把所有问题都交给单轮自由聊天。

项目默认以 FastAPI 作为主运行形态，Gradio 仅作为可选演示界面。

## 核心特性

- 多 Agent 工作流编排：`Supervisor -> Planner -> Worker -> Judge / Replanner -> Response`
- 面向不同任务类型的专用链路：单事实问答、多字段抽取、商品比较、推荐、聚合分析
- 支持人工确认中断流，用于策略确认、低置信回答和导购信息补充
- FastAPI 服务化接口，包含统一响应结构、健康检查和请求追踪
- 分层缓存：业务缓存和 LangChain 大模型缓存分离
- API 边界、外部搜索和 SQL 路径支持异步 I/O
- Docker Compose 一键拉起应用、PostgreSQL 和 Redis
- 保留核心回归测试和 benchmark 样例

## 架构概览

```text
[控制层]
supervisor
  -> planner
  -> orchestrator
  -> judge / replanner
  -> response_agent
  -> hitl_strategy_confirm / hitl_worker_confirm / hitl_low_conf_confirm

[执行层]
retrieval_agent
  -> hybrid_search
  -> rerank_tool

extraction_agent
  -> structured_extractor
  -> field_normalizer

recommendation_agent
  -> catalog_filter
  -> candidate_ranker

calculation_agent
  -> calculator

sql_agent
  -> sql_guard
  -> sql_query / sql_explain

search_agent
  -> tavily_search
```

## 项目结构

```text
server.py             FastAPI 入口
app.py                可选的 Gradio 演示界面
graph/                Agent 图、规划器、监督器、响应链路
tools/                搜索、SQL、抽取、排序、计算等工具
rag/                  检索引擎、评估器、查询增强
cache/                业务缓存和 LangChain 缓存初始化
memory/               会话记忆和长期记忆
data/                 商品、政策、销售、导购知识数据
benchmarks/           Benchmark 样例和结果计算脚本
tests/                最小回归测试集
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，至少配置以下内容：

- `RUN_MODE`
- `DATABASE_URL`
- `POSTGRES_PASSWORD`
- 当前运行模式所需的模型参数

可选配置：

- `TAVILY_API_KEY`
- `LANGSMITH_*`
- `REDIS_URL`
- `LLM_CACHE_*`
- `PERSISTENT_CACHE_*`

### 3. 初始化数据库

```bash
psql -U postgres -d enterprisemind -f init_db.sql
```

如果你直接使用示例配置，先把 `.env` 里的 `change_me` 替换成实际口令。

### 4. 启动 API 服务

```bash
uvicorn server:api --host 0.0.0.0 --port 8000
```

可用接口：

- `GET /health`
- `GET /ready`
- `GET /docs`
- `POST /chat`
- `POST /decision`

### 5. 可选：启动演示界面

```bash
python app.py
```

## Docker

```bash
docker compose up --build
```

默认会启动：

- FastAPI 应用：`http://localhost:8000`
- PostgreSQL
- Redis

## 缓存设计

项目使用两层缓存：

- 业务缓存：内存缓存 + 持久化缓存，默认 `sqlite`，可切换到 `redis`
- 大模型缓存：通过 LangChain 1.x 的 `set_llm_cache(...)` 挂载全局缓存

Redis 配置示例：

```env
PERSISTENT_CACHE_BACKEND=redis
PERSISTENT_CACHE_REDIS_PREFIX=persistent_cache:
LLM_CACHE_BACKEND=redis
REDIS_URL=redis://localhost:6379/0
```

## 测试

运行最小回归测试集：

```bash
python -m pytest tests/test_async_io_paths.py tests/test_api_endpoints.py tests/test_planner_workflow.py tests/test_response_fields.py tests/test_cache_manager.py tests/test_memory_manager.py -q
```

## Benchmark

`benchmarks/` 目录中提供了样例数据和结果计算脚本：

- `benchmark_30_samples.csv`
- `simulated_perf_runs.csv`
- `simulated_rag_scores.csv`
- `calc_benchmark.py`

示例命令：

```bash
python benchmarks/calc_benchmark.py --perf benchmarks/simulated_perf_runs.csv --rag benchmarks/simulated_rag_scores.csv
```

## Roadmap

- 增加更多 API 集成测试
- 补充持久化 graph checkpoint
- 增强节点级指标和可观测性
- 完善启动脚本和部署自动化
