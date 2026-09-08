import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--name")
args = parser.parse_args()


def greet(name):
    return print(f"Hello, {name}")

if __name__ == "__main__":
    greet(args.name)