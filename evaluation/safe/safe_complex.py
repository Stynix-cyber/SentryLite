from pathlib import Path


def count_python_files(directory: Path) -> int:
    count = 0

    for path in directory.rglob("*.py"):
        if path.is_file():
            count += 1

    return count


def main():
    root = Path("src")

    total = count_python_files(root)

    if total > 0:
        print(f"Found {total} Python files")
    else:
        print("No Python files found")


if __name__ == "__main__":
    main()