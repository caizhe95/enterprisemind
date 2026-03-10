# 智能导购多 Agent 系统（EnterpriseMind）3 分钟介绍稿

我做的是一个面向智能导购场景的多 Agent 问答系统，不只是普通的 RAG 问答。

这个项目主要解决 5 类问题:

- 商品单事实问答
- 多字段问答
- 商品比较
- 基于预算和场景的推荐
- 销售数据聚合分析

我的核心设计思路是，先让系统理解问题类型，再决定后续执行计划，而不是把所有问题都交给一个模型直接回答。

在架构上，系统分成几个层次:

- `Planner` 负责识别问题类型并生成步骤
- `Orchestrator` 负责按计划调度 Worker
- `Worker` 负责具体任务，比如检索、结构化抽取、推荐、计算、SQL 查询和外部搜索
- `Judge` 负责检查当前步骤是否满足成功条件
- `Replanner` 在失败时做重试或切换策略
- `Response` 负责输出最终答案

现在业务层一共有 6 个 Worker Agent:

- `retrieval_agent`
- `extraction_agent`
- `recommendation_agent`
- `calculation_agent`
- `sql_agent`
- `search_agent`

举个例子，如果用户问:
`星澜手机1代和智核笔记本2代哪个更贵，差多少元？`

系统不会直接生成答案，而是先拆成:

1. 检索第一个商品的证据
2. 抽取第一个商品的价格
3. 检索第二个商品的证据
4. 抽取第二个商品的价格
5. 调用计算 Worker 算差值
6. 最后由 Response 输出结论

推荐题也是类似的，比如:
`预算9000买哪个游戏笔记本？`

它会先检索候选商品和导购卡片，再由 `extraction_agent` 抽成结构化候选，最后交给 `recommendation_agent` 先过滤再排序。当前我还加了轻量场景权重，比如办公、学生、游戏、拍照、轻薄续航，这样推荐结果不会只是简单看价格，而会更贴近用户意图。

工具层我也做成了“一个 Worker 可挂多个相关工具，但不一定每次全用”的形式。比如:

- `retrieval_agent` 会先做 `hybrid_search`，候选较多时再做 `rerank_tool`
- `extraction_agent` 会先做 `structured_extractor`，必要时再做 `field_normalizer`
- `recommendation_agent` 会先做 `catalog_filter`，然后再做 `candidate_ranker`
- `sql_agent` 会先做 `sql_guard`，再决定执行 SQL 还是只解释 SQL

另外，我把 Worker 的输出统一成结构化 contract，Response 会优先基于这些结构化结果回答，而不是完全依赖自由生成，这样可以提升稳定性和可解释性。

这个项目我认为最有价值的点，不是功能堆得多，而是把智能导购问题做成了一个多 Agent 协作流程，并且对复杂问题有明确的拆解、执行和校验机制。
