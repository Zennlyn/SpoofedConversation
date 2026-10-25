#!/usr/bin/env python3
"""
Transcript Modifier using llama.cpp

Reads a structured transcript (FILEID SENTENCE per line), applies a randomly
chosen prompt to each line via llama.cpp, and writes two output files:

  <stem>.txt    -> FILEID  MODIFIED_SENTENCE
  <stem>.prompt -> FILEID  PROMPTX  PROMPT_TEXT
"""

import argparse
import json
import random
import sys
import urllib.request
import urllib.error
from pathlib import Path

from helper.manipulation_helper import compare_modifications, find_deletions_only, remove_deletions

# ── Prompt pool ───────────────────────────────────────────────────────────────

PROMPTS = [
    "You will repeat what I say but randomly change one or two words in the sentence",
    "You will repeat what I say but randomly change several words in the sentence",
    "You will repeat what I say but randomly change one or two nouns in the sentence",
    "You will repeat what I say but randomly change one or two pronouns in the sentence",
    "You will repeat what I say but randomly change several nouns or pronouns in the sentence",
    "You will repeat what I say but change the sentence slightly to make it means the opposite",
    "You will repeat what I say but change the sentence slightly to make it means something else",
]


# ── Transcript parsing ────────────────────────────────────────────────────────

def parse_transcript(path: str) -> list[tuple[str, str]]:
    """
    Parse a transcript file into (file_id, sentence) pairs.
    Each non-empty line is expected to be:  FILEID<whitespace>SENTENCE
    Lines that don't match are skipped with a warning.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    entries: list[tuple[str, str]] = []
    for lineno, raw in enumerate(lines, 1):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(None, 1)   # split on first whitespace only
        if len(parts) < 2:
            print(f"[!] Line {lineno} skipped (no sentence): {raw!r}", file=sys.stderr)
            continue
        entries.append((parts[0], parts[1]))
    return entries


# ── LLM calls ─────────────────────────────────────────────────────────────────

def build_prompt(instruction: str, sentence: str) -> str:
    return (
        "Do NOT change the sentence structure. Do a text modification as instructed below.\n"
        f"Instruction: {instruction}.\n\n"
        "Output ONLY the modified sentence. "
        "Do NOT include the original sentence, explanations, notes, change logs, dashes, parentheses, or any extra text.\n\n"
        "WRONG output example: 'She went to the store. - changed X to Y'\n"
        "CORRECT output example: 'She went to the store.'\n\n"
        f"Original sentence: {sentence}\n"
        f"Modified: "
    )


def call_server(prompt: str, host: str, port: int, max_tokens: int, temperature: float, stop: list[str]) -> str:
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = json.dumps({
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a text editor. You output ONLY the modified sentence. "
                    "No explanations, no notes, no dashes, no parentheses, no lists of changes. "
                    "A single sentence. Nothing else."
                )
            },
            {
                "role": "user",
                "content": f"{prompt}"
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
    }).encode()

    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
            print(body["choices"][0]["message"]["content"].strip(), "\n")
        return body["choices"][0]["message"]["content"].strip()
    except urllib.error.URLError as exc:
        sys.exit(
            f"[✗] Could not reach llama.cpp server at {url}\n"
            f"    Start it with:  ./llama-server -m <model.gguf> --port {port}\n"
            f"    Error: {exc}"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Modify a structured transcript line-by-line with random prompts via llama.cpp.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output files
------------
  Given --output test01, the script writes:
    test01.txt    → FILEID  MODIFIED_SENTENCE  (one line per entry)
    test01.prompt → FILEID  PROMPTX  PROMPT_TEXT

  It then compares the result with the input and also writes:
    test01_modification_info.txt — per-line insertion/replacement/deletion counts
    test01_deletions_only.txt    — lines with only deletions
    test01_cleaned.txt           — test01.txt without the deletion-only lines

Examples
--------
  python modify_transcript.py --input transcript.txt --output test01

  # Custom port:
  python modify_transcript.py --input transcript.txt --output test01 --port 8081

  # Fix random seed for reproducibility:
  python modify_transcript.py --input transcript.txt --output test01 --seed 42
""",
    )

    p.add_argument("--input",   required=True, help="Structured transcript file (FILEID SENTENCE per line).")
    p.add_argument("--output",  required=True, help="Output stem (e.g. 'test01' → test01.txt + test01.prompt).")
    p.add_argument("--seed",    default=None, type=int,
                   help="Random seed for prompt selection (default: random).")

    sg = p.add_argument_group("server options")
    sg.add_argument("--host",   default="127.0.0.1")
    sg.add_argument("--port",   default=8080, type=int)

    gg = p.add_argument_group("generation options")
    gg.add_argument("--max-tokens",  default=128,  type=int, dest="max_tokens",
                    help="Max tokens per line (default: 128).")
    gg.add_argument("--temperature", default=0.7,  type=float,
                    help="Sampling temperature (default: 0.7).")
    gg.add_argument("--stop",        nargs="*",
                    default=["</s>", "###", "\n"],
                    help="Stop tokens.")

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Parse input
    entries = parse_transcript(args.input)
    if not entries:
        sys.exit("[✗] No valid transcript lines found.")
    print(f"[✓] Loaded {len(entries)} lines from {args.input}", file=sys.stderr)

    # Prepare output paths
    stem = args.output
    txt_path    = Path(f"{stem}.txt")
    prompt_path = Path(f"{stem}.prompt")

    txt_lines:    list[str] = []
    prompt_lines: list[str] = []

    # Process each line
    for i, (file_id, sentence) in enumerate(entries, 1):
        # Pick a random prompt
        prompt_idx  = random.randrange(len(PROMPTS))
        instruction = PROMPTS[prompt_idx]
        prompt_key  = f"PROMPT{prompt_idx}"

        full_prompt = build_prompt(instruction, sentence)

        print(
            f"[{i}/{len(entries)}] {file_id}  →  {prompt_key}: {instruction[:55]}…",
            file=sys.stderr,
        )

        # Call LLM
        modified = call_server(
            prompt=full_prompt,
            host=args.host,
            port=args.port,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stop=args.stop,
        )

        # Sanitise: strip accidental newlines from the model output
        modified = modified.replace("\n", " ").strip()

        txt_lines.append(f"{file_id} {modified}")
        prompt_lines.append(f"{file_id} {prompt_key} {instruction}")

    # Write output files
    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    prompt_path.write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")

    print(f"[✓] Modified transcript → {txt_path}", file=sys.stderr)
    print(f"[✓] Prompt log          → {prompt_path}", file=sys.stderr)

    # Compare against the input, find deletion-only edits, then remove them
    info_path      = f"{stem}_modification_info.txt"
    deletions_path = f"{stem}_deletions_only.txt"
    cleaned_path   = f"{stem}_cleaned.txt"

    compare_modifications(args.input, str(txt_path), info_path)
    found = find_deletions_only(info_path, deletions_path)
    removed, kept = remove_deletions(deletions_path, str(txt_path), cleaned_path)

    print(f"[✓] Comparison report   → {info_path}", file=sys.stderr)
    print(f"[✓] Deletion-only ({len(found)})  → {deletions_path}", file=sys.stderr)
    print(f"[✓] Cleaned transcript  → {cleaned_path} ({removed} removed, {kept} kept)", file=sys.stderr)


if __name__ == "__main__":
    main()