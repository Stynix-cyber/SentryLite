from __future__ import annotations

from pathlib import Path


SAFE_DIR = Path("evaluation/safe")
RISKY_DIR = Path("evaluation/risky")


SAFE_SAMPLES = {
    "safe_math.py": """
def calculate(a, b):
    result = a * b + 10
    return result

print(calculate(4, 8))
""",

    "safe_input_print.py": """
name = input("Name: ")
print("Hello", name)
""",

    "safe_input_length.py": """
value = input("Value: ")
length = len(value)
print(length)
""",

    "safe_input_function.py": """
def read_name():
    return input("Name: ")

name = read_name()
print(name)
""",

    "safe_subprocess_list.py": """
import subprocess

subprocess.run(
    ["python", "--version"],
    check=True
)
""",

    "safe_pathlib.py": """
from pathlib import Path

root = Path("src")

for file in root.rglob("*.py"):
    print(file.name)
""",

    "safe_json.py": """
import json

data = '{"name": "test"}'
result = json.loads(data)

print(result["name"])
""",

    "safe_literal_eval.py": """
import ast

value = input("Value: ")

try:
    result = ast.literal_eval(value)
    print(result)
except (ValueError, SyntaxError):
    print("Invalid value")
""",

    "safe_branching.py": """
value = input("Value: ")

if value.isdigit():
    number = int(value)
    print(number * 2)
else:
    print("Not a number")
""",

    "safe_function_chain.py": """
def read_value():
    return input("Value: ")

def clean(value):
    return value.strip()

def display(value):
    print(value)

data = read_value()
cleaned = clean(data)
display(cleaned)
""",

    "safe_file_read.py": """
from pathlib import Path

path = Path("example.txt")

if path.exists():
    text = path.read_text(encoding="utf-8")
    print(text)
""",

    "safe_loop.py": """
values = ["a", "b", "c"]

for value in values:
    print(value.upper())
""",

    "safe_dictionary.py": """
users = {
    "alice": 10,
    "bob": 20,
}

for name, score in users.items():
    print(name, score)
""",

    "safe_exception.py": """
try:
    number = int(input("Number: "))
    print(number)
except ValueError:
    print("Invalid number")
""",

    "safe_class.py": """
class Calculator:
    def add(self, a, b):
        return a + b

calculator = Calculator()
print(calculator.add(5, 7))
""",
}


RISKY_SAMPLES = {
    "risky_eval_direct.py": """
value = input("Expression: ")
eval(value)
""",

    "risky_eval_copy.py": """
value = input("Expression: ")
copy = value
eval(copy)
""",

    "risky_eval_chain.py": """
value = input("Expression: ")
first = value
second = first
third = second
eval(third)
""",

    "risky_eval_function_source.py": """
def read_value():
    return input("Expression: ")

value = read_value()
eval(value)
""",

    "risky_eval_function_passthrough.py": """
def passthrough(value):
    return value

raw = input("Expression: ")
prepared = passthrough(raw)

eval(prepared)
""",

    "risky_eval_two_functions.py": """
def read_value():
    return input("Expression: ")

def prepare(value):
    copied = value
    return copied

raw = read_value()
prepared = prepare(raw)

eval(prepared)
""",

    "risky_exec_direct.py": """
code = input("Code: ")
exec(code)
""",

    "risky_exec_copy.py": """
code = input("Code: ")
prepared = code
exec(prepared)
""",

    "risky_exec_function.py": """
def read_code():
    return input("Code: ")

code = read_code()
exec(code)
""",

    "risky_subprocess_shell.py": """
import subprocess

command = input("Command: ")

subprocess.run(
    command,
    shell=True
)
""",

    "risky_subprocess_copy.py": """
import subprocess

command = input("Command: ")
prepared = command

subprocess.run(
    prepared,
    shell=True
)
""",

    "risky_subprocess_function.py": """
import subprocess

def read_command():
    return input("Command: ")

command = read_command()

subprocess.run(
    command,
    shell=True
)
""",

    "risky_pickle.py": """
import pickle

data = input("Serialized data: ")
pickle.loads(data)
""",

    "risky_pickle_copy.py": """
import pickle

data = input("Serialized data: ")
copy = data

pickle.loads(copy)
""",

    "risky_pickle_function.py": """
import pickle

def passthrough(value):
    return value

data = input("Serialized data: ")
prepared = passthrough(data)

pickle.loads(prepared)
""",
}


def write_samples(
    directory: Path,
    samples: dict[str, str],
) -> None:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for filename, source in samples.items():

        path = directory / filename

        path.write_text(
            source.strip() + "\n",
            encoding="utf-8",
        )


def main() -> None:

    write_samples(
        SAFE_DIR,
        SAFE_SAMPLES,
    )

    write_samples(
        RISKY_DIR,
        RISKY_SAMPLES,
    )

    print("Evaluation dataset generated.")
    print(
        f"Safe samples:  {len(SAFE_SAMPLES)}"
    )
    print(
        f"Risky samples: {len(RISKY_SAMPLES)}"
    )
    print(
        f"Total:         "
        f"{len(SAFE_SAMPLES) + len(RISKY_SAMPLES)}"
    )


if __name__ == "__main__":
    main()