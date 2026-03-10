# 面试演示问题清单

以下 7 个问题用于演示系统在智能导购场景下的多 Agent 能力。

## 1. 单事实问答
- 问题: `智核笔记本2代属于什么品类？`
- 预期流程: `planner -> retrieval_agent -> extraction_agent -> response_agent`
- 展示点: 简单问题也先检索再抽取，Response 优先消费结构化结果。

## 2. 多字段问答
- 问题: `星澜手机1代的品类和价格分别是什么？`
- 预期流程: `planner -> retrieval_agent -> extraction_agent -> response_agent`
- 展示点: extraction worker 会显式产出字段值，response 优先消费结构化字段结果。

## 3. 比较题
- 问题: `星澜手机1代和智核笔记本2代哪个更贵，差多少元？`
- 预期流程: `planner -> retrieval_agent -> extraction_agent -> retrieval_agent -> extraction_agent -> calculation_agent -> response_agent`
- 展示点: 比较题被拆成检索、抽取、计算、表达四层，更容易观察每一步产物。

## 4. 推荐题-预算
- 问题: `预算4000买哪个笔记本？`
- 预期流程: `planner -> retrieval_agent -> extraction_agent -> recommendation_agent -> response_agent`
- 展示点: 推荐题走候选检索 + 结构化抽取 + 过滤排序，不再等同于普通问答。

## 5. 推荐题-游戏场景
- 问题: `预算9000买哪个游戏笔记本？`
- 预期流程: `planner -> retrieval_agent -> extraction_agent -> recommendation_agent -> response_agent`
- 展示点: 游戏场景会偏向高性能和散热稳定，且推荐前会先抽取候选结构化信息。

## 6. 推荐题-办公场景
- 问题: `预算5000买哪个办公笔记本？`
- 预期流程: `planner -> retrieval_agent -> extraction_agent -> recommendation_agent -> response_agent`
- 展示点: 办公场景会偏向长续航和轻薄便携，可与游戏场景形成对照。

## 7. 聚合分析题
- 问题: `前10条销售记录里销售额最高的是哪条？`
- 预期流程: `planner -> sql_agent -> response_agent`
- 展示点: 结构化分析题直接走 SQL worker，不走普通检索。

## 面试讲解建议
- 先演示第 3 题，说明多 Agent 串联。
- 再演示第 5 题和第 6 题，说明推荐不是只看预算，还会考虑场景权重和候选排序。
- 最后演示第 7 题，说明系统支持结构化分析。
