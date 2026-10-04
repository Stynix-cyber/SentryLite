def read_value():
    return input("Value: ")

def clean(value):
    return value.strip()

def display(value):
    print(value)

data = read_value()
cleaned = clean(data)
display(cleaned)
