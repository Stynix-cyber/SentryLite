import subprocess


def show_version():
    subprocess.run(
        ["python", "--version"],
        check=True,
    )