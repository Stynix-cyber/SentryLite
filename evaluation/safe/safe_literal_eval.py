import ast

value = input("Value: ")

try:
    result = ast.literal_eval(value)
    print(result)
except (ValueError, SyntaxError):
    print("Invalid value")
