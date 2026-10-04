class RequestData:
    def __init__(self):
        self.value = None


request = RequestData()

user_value = input("Expression: ")

request.value = user_value

prepared_value = request.value

eval(prepared_value)