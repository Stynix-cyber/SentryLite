def read_expression():
    return input("Expression: ")


def prepare(value):
    copied = value
    return copied


user_value = read_expression()
prepared_value = prepare(user_value)

eval(prepared_value)