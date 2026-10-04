from pathlib import Path

root = Path("src")

for file in root.rglob("*.py"):
    print(file.name)
