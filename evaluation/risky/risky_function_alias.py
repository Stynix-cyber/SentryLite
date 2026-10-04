def prepare(value):
    return value


alias = prepare

user_value = input("Expression: ")

prepared_value = alias(user_value)

eval(prepared_value)