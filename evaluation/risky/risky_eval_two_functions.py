def read_value():
    return input("Expression: ")

def prepare(value):
    copied = value
    return copied

raw = read_value()
prepared = prepare(raw)

eval(prepared)
