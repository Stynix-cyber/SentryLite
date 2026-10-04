def prepare(value):
    return value


user_value = input("Expression: ")

prepared_value = prepare(value=user_value)

eval(prepared_value)