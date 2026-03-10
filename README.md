# 智能导购多 Agent 系统

面向智能导购场景的多 Agent 问答项目。

项目目标不是做一个通用聊天机器人，而是把导购问题拆成可执行的步骤，让不同 Worker 协作完成单事实问答、多字段抽取、商品比较、预算推荐和销售分析。

## 项目定位

- 业务域: 智能导购
- 核心能力: 检索、结构化抽取、比较、计算、推荐、聚合分析
- 架构重点: `Planner + Orchestrator + Worker + Judge/Replanner + Response`

## 支持的问题类型

- `single_fact`
  例: `智核笔记本2代属于什么品类？`
- `field_list`
  例: `星澜手机1代的品类和价格分别是什么？`
- `comparison`
  例: `星澜手机1代和智核笔记本2代哪个更贵，差多少元？`
- `recommendation`
  例: `预算4000买哪个笔记本？`
- `aggregation`
  例: `前10条销售记录里销售额最高的是哪条？`

## 系统架构

```text
Supervisor
  -> Planner
  -> Orchestrator
  -> Worker(s)
       - retrieval_agent
       - extraction_agent
       - recommendation_agent
       - calculation_agent
       - sql_agent
       - search_agent
  -> Judge / Replanner
  -> Response
```

### 分层架构图

```text
[Control Layer]
  supervisor
    -> planner
    -> orchestrator
    -> judge / replanner
    -> response_agent
    -> hitl_strategy_confirm / hitl_worker_confirm / hitl_low_conf_confirm

[Worker Layer]
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
    -> generate_sql_with_examples
    -> sql_guard
    -> sql_query / sql_explain

  search_agent
    -> tavily_search

[Typical Paths]
  single_fact:
    planner -> retrieval_agent -> extraction_agent -> response_agent

  comparison:
    planner -> retrieval_agent -> extraction_agent
            -> retrieval_agent -> extraction_agent
            -> calculation_agent -> response_agent

  recommendation:
    planner -> retrieval_agent -> extraction_agent
            -> recommendation_agent -> response_agent

  aggregation:
    planner -> sql_agent -> response_agent
```

各模块职责:

- `Supervisor`
  负责高层策略入口和人工确认分支。
- `Planner`
  识别问题类型并生成执行计划。
- `Orchestrator`
  按计划逐步调度 Worker。
- `Worker`
  分别负责检索、结构化抽取、推荐、计算、SQL 分析、外部搜索。
- `Judge`
  检查当前步骤是否满足 `expects` 条件。
- `Replanner`
  当前步骤失败时重试或切换策略。
- `Response`
  优先基于结构化 artifacts 输出答案，不足时才回退到 LLM 生成。

当前共有 `6` 个业务 Worker Agent:

- `retrieval_agent`
  负责召回候选证据，内部支持 `hybrid_search -> rerank_tool` 的两阶段检索。
- `extraction_agent`
  负责从召回结果中抽取字段、指标和候选商品，内部支持 `structured_extractor -> field_normalizer`。
- `recommendation_agent`
  负责候选过滤与排序，内部支持 `catalog_filter -> candidate_ranker`。
- `calculation_agent`
  负责价差、数值表达式等计算。
- `sql_agent`
  负责 SQL 生成、安全检查、执行或解释。
- `search_agent`
  负责外部搜索兜底补证据。

工具层当前的核心能力为:

- `hybrid_search`
- `rerank_tool`
- `structured_extractor`
- `field_normalizer`
- `catalog_filter`
- `candidate_ranker`
- `calculator`
- `sql_guard`
- `sql_query`
- `sql_explain`
- `tavily_search`

## 典型处理流程

### 1. 单事实问答

问题:
`智核笔记本2代属于什么品类？`

流程:
`planner -> retrieval_agent -> extraction_agent -> response_agent`

### 2. 比较题

问题:
`星澜手机1代和智核笔记本2代哪个更贵，差多少元？`

流程:
`planner -> retrieval_agent -> extraction_agent -> retrieval_agent -> extraction_agent -> calculation_agent -> response_agent`

### 3. 推荐题

问题:
`预算9000买哪个游戏笔记本？`

流程:
`planner -> retrieval_agent -> extraction_agent -> recommendation_agent -> response_agent`

推荐逻辑:

- 推荐题的检索会优先召回导购卡片和候选商品
- 抽取阶段会把候选商品结构化为名称、品类、价格、亮点
- 推荐阶段先过滤再排序，会综合预算、品类和场景权重
- 当前支持的轻量场景包括:
  - 办公
  - 学生
  - 游戏
  - 拍照
  - 轻薄续航

### 4. 聚合分析题

问题:
`前10条销售记录里销售额最高的是哪条？`

流程:
`planner -> sql_agent -> response_agent`

## 主要亮点

- 从单 Agent RAG 升级到多 Agent 协作工作流
- 简单问题走短链路，复杂问题走多步骤规划
- 检索与结构化抽取显式分层，避免 retrieval worker 过厚
- 比较题显式拆成“检索 -> 抽取 -> 计算 -> 表达”
- 推荐题显式拆成“召回 -> 抽取 -> 过滤/排序 -> 回答”
- 核心 Worker 采用多工具可选调用，而不是一个 Agent 固定只绑一个工具
- 推荐不只看预算，还支持场景权重
- SQL Agent 内部具备 `生成 -> guard -> 执行/解释` 的小流水线
- Worker 统一输出标准化 contract:
  `status / summary / artifacts / signals / confidence / errors`
- Response 优先读结构化结果，减少自由生成的不稳定性
- Judge + Replanner 负责步骤质量控制和失败重试

## 典型查询示例

更多示例可见 [demo_questions.md](/D:/Pycharm2025/PyCharm%202025.2.5/save/enterprisemind/demo_questions.md)。

具有代表性的查询包括:

1. `智核笔记本2代属于什么品类？`
2. `星澜手机1代的品类和价格分别是什么？`
3. `星澜手机1代和智核笔记本2代哪个更贵，差多少元？`
4. `预算9000买哪个游戏笔记本？`
5. `前10条销售记录里销售额最高的是哪条？`


## 项目结构

```text
app.py                # Gradio 入口
server.py             # FastAPI 入口
graph/                # 多 Agent 编排、规划与响应逻辑
rag/                  # 检索与查询优化
tools/                # SQL / 搜索 / 计算工具
                     # 也包含抽取 / 过滤 / 排序 / guard 工具
data/                 # 产品、政策、销售、导购卡片数据
benchmarks/           # benchmark 数据
tests/                # 回归测试
demo_questions.md     # 面试演示问题清单
```

## 快速运行

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 配置 `.env`

至少需要:

- `RUN_MODE`
- `DATABASE_URL`
- 对应模式的模型配置

双模式说明:

- `local`
  使用 `DeepSeek API + 本地 HuggingFace Embedding`
  必需配置:
  - `RUN_MODE=local`
  - `DEEPSEEK_API_KEY`
  - `DEEPSEEK_MODEL` 可选

- `cloud`
  使用 `Ollama Chat + Ollama Embedding`
  必需配置:
  - `RUN_MODE=cloud`
  - `OLLAMA_BASE_URL`
  - `OLLAMA_MODEL`
  - `OLLAMA_EMBEDDING_MODEL`

3. 初始化数据库

```bash
psql -U postgres -d enterprisemind -f init_db.sql
```

4. 启动界面

```bash
python app.py
```

5. 启动 API

```bash
uvicorn server:api --host 0.0.0.0 --port 8000
```

## Docker 双模式

`docker-compose.yml` 现在会透传 `RUN_MODE` 和模式相关配置。

- 本地 API 模式:
  - `RUN_MODE=local`
  - 提供 `DEEPSEEK_API_KEY`

- Ollama 模式:
  - `RUN_MODE=cloud`
  - 提供 `OLLAMA_BASE_URL`
  - 如 Ollama 运行在宿主机，可使用默认的 `http://host.docker.internal:11434`

## 测试

当前已覆盖:

- 字段抽取
- 字段归一化
- 计划生成
- 比较题计算准备
- 重规划逻辑
- 检索结果重排
- 推荐候选过滤与排序
- SQL 安全检查
- 推荐排序
- 场景权重
- 结构化 response

运行:

```bash
python -m pytest tests\test_response_fields.py tests\test_planner_workflow.py tests\test_recommendation_agent.py -q
```

## 后续优化

- 进一步增强 recommendation worker 的候选召回和排序策略
- 增强 replanner 的动态重规划能力
- 增加更多导购场景测试集
- 增加更细粒度的可视化执行链路展示
