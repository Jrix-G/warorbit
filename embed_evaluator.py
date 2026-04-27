"""Embed evaluator_b64.txt into a standalone bot_v6 copy."""

import argparse
import re


def embed_evaluator(input_path="bot_v6.py", b64_path="evaluator_b64.txt",
                    output_path="bot_v6_trained.py"):
    with open(input_path, "r", encoding="utf-8") as f:
        bot_code = f.read()
    with open(b64_path, "r", encoding="ascii") as f:
        b64 = f.read().strip()

    if not b64:
        raise ValueError(f"{b64_path} is empty")

    updated, n = re.subn(
        r'^EVALUATOR_B64\s*=\s*".*"$',
        f'EVALUATOR_B64 = "{b64}"',
        bot_code,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise RuntimeError("Could not find a unique EVALUATOR_B64 assignment")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(updated)
    print(f"Wrote {output_path} with {len(b64)} base64 chars")


def main():
    parser = argparse.ArgumentParser(description="Embed trained evaluator into bot_v6.")
    parser.add_argument("--input", default="bot_v6.py")
    parser.add_argument("--b64", default="evaluator_b64.txt")
    parser.add_argument("--output", default="bot_v6_trained.py")
    args = parser.parse_args()
    embed_evaluator(args.input, args.b64, args.output)


if __name__ == "__main__":
    main()
