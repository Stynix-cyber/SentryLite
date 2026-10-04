from pickle import loads as deserialize


user_value = input("Serialized data: ")

deserialize(user_value)