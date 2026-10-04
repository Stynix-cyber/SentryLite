from pathlib import Path

path = Path("example.txt")

if path.exists():
    text = path.read_text(encoding="utf-8")
    print(text)
