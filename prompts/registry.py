# prompts/registry.py - Self-RAG专用Prompt注册表
"""Prompt管理注册表 - 支持Self-RAG"""

from typing import Dict, Optional, Any

# Self-RAG专用Prompts
SELF_RAG_PROMPTS = {
    "self_rag_retrieval_eval": """评估以下文档与用户问题的相关性。

用户问题：{question}
文档内容：{document}

{criteria}

请只返回以下之一：HIGHLY_RELEVANT / PARTIALLY_RELEVANT / IRRELEVANT
并简要说明理由（1句话）。

评估结果：""",
    "self_rag_support_eval": """评估AI回答是否得到证据支持（检测幻觉）。

问题：{question}
AI回答：{answer}
证据材料：
{evidence}

请评估：
1. 回答中的每个事实是否都能在证据中找到？
2. 是否有证据外的推测或编造？

返回以下之一并说明：
- FULLY_SUPPORTED: 完全由证据支撑
- PARTIALLY_SUPPORTED: 部分支撑，有部分推测
- NO_SUPPORT: 主要事实无证据支撑（幻觉风险）

评估：""",
    "self_rag_utility_eval": """评估回答对用户的实用性。

用户问题：{question}
AI回答：{answer}

评估标准：
- HIGHLY_USEFUL: 直接、完整回答原问题
- SOMEWHAT_USEFUL: 部分回答，或回答了相关问题但未完全解决原问题  
- NOT_USEFUL: 答非所问或信息不足

返回评级和理由：

评估：""",
    "self_rag_rewrite": """基于失败反馈改写检索查询。

原查询：{original_query}
检索失败原因：{feedback}
已尝试关键词：{previous_keywords}

请生成新的查询策略：
1. 如果是同义词问题，使用不同表达方式
2. 如果是过于宽泛，添加具体限定词
3. 如果是多步问题，分解为子查询

新查询（只输出查询文本）：""",
    "sql_generator": """你是一个SQL生成专家。根据以下数据库Schema和用户问题生成SQL查询。

数据库Schema：
{schema}

用户问题：{question}

要求：
1. 只生成SELECT语句，禁止生成DELETE/UPDATE/INSERT/DROP等危险操作
2. 如果问题不明确，生成最合理的假设
3. 自动添加LIMIT 50限制返回行数
4. 返回纯SQL代码，不要markdown格式

生成的SQL：""",
}


class PromptRegistry:
    """Prompt注册表 - 单例模式"""

    _instance = None
    _prompts: Dict[str, str] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_all()
        return cls._instance

    def _load_all(self):
        """加载所有Prompt模板"""
        # 加载Self-RAG专用Prompts
        for key, value in SELF_RAG_PROMPTS.items():
            self._prompts[key] = value

    @classmethod
    def get(cls, key: str, variables: Optional[Dict[str, Any]] = None) -> str:
        """
        获取Prompt模板并填充变量

        Args:
            key: Prompt模板名称
            variables: 要填充的变量字典

        Returns:
            填充后的Prompt字符串
        """
        # 确保实例已初始化
        if cls._instance is None:
            cls._instance = cls()

        template = cls._instance._prompts.get(key, "")
        if not template:
            print(f"[PromptRegistry] 警告: 未找到Prompt模板 '{key}'")
            return ""

        # 填充变量
        if variables:
            try:
                return template.format(**variables)
            except KeyError as e:
                print(f"[PromptRegistry] 警告: 模板变量缺失 {e}")
                return template

        return template

    @classmethod
    def register(cls, key: str, template: str):
        """注册新的Prompt模板"""
        if cls._instance is None:
            cls._instance = cls()
        cls._instance._prompts[key] = template

    @classmethod
    def list_keys(cls) -> list:
        """列出所有可用的Prompt键"""
        if cls._instance is None:
            cls._instance = cls()
        return list(cls._instance._prompts.keys())


# 向后兼容的便捷函数
def get_prompt(key: str, variables: Optional[Dict[str, Any]] = None) -> str:
    """便捷函数：获取Prompt"""
    return PromptRegistry.get(key, variables)
