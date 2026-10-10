import argparse
from pathlib import Path
from riskshield.infrastructure.registry import FileModelRegistry
from riskshield.pipeline import train


def main():
    parser = argparse.ArgumentParser(description="RiskShield model lifecycle")
    parser.add_argument("--registry", default="registry")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train-demo")
    train_parser.add_argument("--task", choices=["fraud", "credit"], required=True)
    train_parser.add_argument("--rows", type=int, default=20000)
    register = commands.add_parser("register")
    register.add_argument("--name", required=True)
    register.add_argument("--version", required=True)
    register.add_argument("--source", required=True)
    promote = commands.add_parser("promote")
    promote.add_argument("--name", required=True)
    promote.add_argument("--version", required=True)
    commands.add_parser("list")
    args = parser.parse_args()
    registry = FileModelRegistry(args.registry)
    if args.command == "train-demo":
        train(str(Path("artifacts") / ("credit" if args.task == "credit" else "")), args.rows, args.task)
    elif args.command == "register":
        print(registry.register(args.name, args.version, args.source))
    elif args.command == "promote":
        registry.promote(args.name, args.version)
        print("Active model updated. Use an older version to roll back.")
    else:
        print(registry.names())


if __name__ == "__main__":
    main()
