from __future__ import annotations

import random
from pathlib import Path


SEED = 42

SAFE_DIR = Path("dataset/safe")
RISKY_DIR = Path("dataset/risky")

SAMPLES_PER_CLASS = 500


VARIABLE_NAMES = [
    "data",
    "value",
    "text",
    "payload",
    "content",
    "expression",
    "message",
    "item",
]

FUNCTION_NAMES = [
    "process",
    "handle",
    "calculate",
    "convert",
    "read_value",
    "transform",
    "execute_task",
]


def random_variable() -> str:
    return random.choice(VARIABLE_NAMES)


def random_function() -> str:
    return random.choice(FUNCTION_NAMES)


# ------------------------------------------------------------
# SAFE TEMPLATES
# ------------------------------------------------------------

def safe_math() -> str:
    function = random_function()

    return f"""
def {function}(a, b):
    result = a + b
    return result

print({function}(10, 20))
""".strip()


def safe_input_print() -> str:
    variable = random_variable()

    return f"""
{variable} = input("Value: ")

if {variable}:
    print({variable})
""".strip()


def safe_integer_conversion() -> str:
    variable = random_variable()

    return f"""
{variable} = input("Number: ")

try:
    number = int({variable})
    print(number * 2)
except ValueError:
    print("Invalid number")
""".strip()


def safe_list_processing() -> str:
    function = random_function()

    return f"""
def {function}(items):
    result = []

    for item in items:
        if isinstance(item, int):
            result.append(item * 2)

    return result
""".strip()


def safe_subprocess() -> str:
    return """
import subprocess

subprocess.run(
    ["python", "--version"],
    check=True
)
""".strip()


def safe_literal_eval() -> str:
    variable = random_variable()

    return f"""
import ast

{variable} = input("Value: ")

try:
    result = ast.literal_eval({variable})
    print(result)
except (ValueError, SyntaxError):
    print("Invalid value")
""".strip()


SAFE_GENERATORS = [
    safe_math,
    safe_input_print,
    safe_integer_conversion,
    safe_list_processing,
    safe_subprocess,
    safe_literal_eval,
]


# ------------------------------------------------------------
# RISKY TEMPLATES
# ------------------------------------------------------------

def risky_eval_direct() -> str:
    variable = random_variable()

    return f"""
{variable} = input("Expression: ")
result = eval({variable})
print(result)
""".strip()


def risky_eval_propagated() -> str:
    first = random_variable()
    second = first + "_copy"

    return f"""
{first} = input("Expression: ")
{second} = {first}

result = eval({second})
print(result)
""".strip()


def risky_exec() -> str:
    variable = random_variable()

    return f"""
{variable} = input("Code: ")
exec({variable})
""".strip()


def risky_shell() -> str:
    variable = random_variable()

    return f"""
import subprocess

{variable} = input("Command: ")

subprocess.run(
    {variable},
    shell=True
)
""".strip()


def risky_pickle() -> str:
    variable = random_variable()

    return f"""
import pickle

def load_data({variable}):
    return pickle.loads({variable})
""".strip()


RISKY_GENERATORS = [
    risky_eval_direct,
    risky_eval_propagated,
    risky_exec,
    risky_shell,
    risky_pickle,
]


# ------------------------------------------------------------
# FILE GENERATION
# ------------------------------------------------------------

def prepare_directories() -> None:
    SAFE_DIR.mkdir(parents=True, exist_ok=True)
    RISKY_DIR.mkdir(parents=True, exist_ok=True)

    for directory in (SAFE_DIR, RISKY_DIR):
        for file in directory.glob("*.py"):
            file.unlink()


def generate_samples(
    directory: Path,
    generators: list,
    amount: int,
) -> None:

    for index in range(amount):
        generator = random.choice(generators)

        source = generator()

        path = directory / f"sample_{index:04d}.py"

        path.write_text(
            source + "\n",
            encoding="utf-8",
        )


def main() -> None:
    random.seed(SEED)

    prepare_directories()

    generate_samples(
        SAFE_DIR,
        SAFE_GENERATORS,
        SAMPLES_PER_CLASS,
    )

    generate_samples(
        RISKY_DIR,
        RISKY_GENERATORS,
        SAMPLES_PER_CLASS,
    )

    total = SAMPLES_PER_CLASS * 2

    print("SentryLite Dataset Generator")
    print("=" * 40)
    print(f"Seed:          {SEED}")
    print(f"Safe samples:  {SAMPLES_PER_CLASS}")
    print(f"Risky samples: {SAMPLES_PER_CLASS}")
    print(f"Total:         {total}")


if __name__ == "__main__":
    main()