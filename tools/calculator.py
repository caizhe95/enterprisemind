"""计算器"""

from typing import Annotated
from langchain_core.tools import tool
import ast
import operator

SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


@tool
def calculator(expression: Annotated[str, "数学表达式，如'1500 * 1.13'"]) -> str:
    """安全数学计算"""

    try:
        expression = expression.strip().replace("^", "**")
        tree = ast.parse(expression, mode="eval")

        def eval_node(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                return SAFE_OPS[type(node.op)](
                    eval_node(node.left), eval_node(node.right)
                )
            elif isinstance(node, ast.UnaryOp):
                return SAFE_OPS[type(node.op)](eval_node(node.operand))
            else:
                raise ValueError(f"不支持: {type(node)}")

        result = eval_node(tree.body)
        return f"{expression} = {result}"

    except ZeroDivisionError:
        return "错误: 除零"
    except Exception as e:
        return f"错误: {e}"
