import subprocess

command = input("Command: ")
prepared = command

subprocess.run(
    prepared,
    shell=True
)
