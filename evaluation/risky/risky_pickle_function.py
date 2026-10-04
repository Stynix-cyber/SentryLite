import pickle

def passthrough(value):
    return value

data = input("Serialized data: ")
prepared = passthrough(data)

pickle.loads(prepared)
