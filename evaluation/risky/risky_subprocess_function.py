import subprocess

def read_command():
    return input("Command: ")

command = read_command()

subprocess.run(
    command,
    shell=True
)
