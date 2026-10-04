import pickle
import subprocess


def calculate(user_input):
    return eval(user_input)


def execute(command):
    subprocess.run(command, shell=True)


def deserialize(data):
    return pickle.loads(data)