"""Export/check a tiny step-zero prompt for an initialization-swap experiment."""

import argparse
import json

from lerobot.utils.prompt_initialization import export_initial_prompt, read_initial_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Extract prompt_rank/step_000000.npz from a completed run")
    export.add_argument("run")
    export.add_argument("--output", required=True)
    export.add_argument("--expect-seed", type=int, required=True)
    check = commands.add_parser("check", help="Validate a transferred initialization before loading a model")
    check.add_argument("path")
    check.add_argument("--tokens", type=int, required=True)
    args = parser.parse_args()
    if args.command == "export":
        metadata = export_initial_prompt(args.run, args.output, args.expect_seed)
    else:
        _, metadata = read_initial_prompt(args.path, args.tokens)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
