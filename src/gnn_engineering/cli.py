from __future__ import annotations
import argparse
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser(description="Graph Neural Networks Engineering CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect-config", help="Validate and print an experiment YAML config")
    inspect.add_argument("--config", required=True)
    args = parser.parse_args()

    if args.command == "inspect-config":
        path = Path(args.config)
        if not path.is_file():
            parser.error(f"Config file not found: {path}")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        required = {"experiment", "dataset", "model", "training"}
        missing = required - set(config or {})
        if missing:
            parser.error(f"Missing top-level config sections: {', '.join(sorted(missing))}")
        print(yaml.safe_dump(config, sort_keys=False))


if __name__ == "__main__":
    main()
