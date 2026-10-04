import pickle

data = input("Serialized data: ")
copy = data

pickle.loads(copy)
