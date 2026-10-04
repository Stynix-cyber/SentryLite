import ast


def calculate(expression: str):
    try:
        tree = ast.parse(expression, mode="eval")

        if not isinstance(tree, ast.Expression):
            return None

        return ast.literal_eval(expression)

    except (SyntaxError, ValueError):
        return None