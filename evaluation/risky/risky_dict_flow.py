data = {}

user_value = input("Expression: ")

data["expression"] = user_value

prepared_value = data["expression"]

eval(prepared_value)