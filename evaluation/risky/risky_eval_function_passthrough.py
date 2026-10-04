def passthrough(value):
    return value

raw = input("Expression: ")
prepared = passthrough(raw)

eval(prepared)
