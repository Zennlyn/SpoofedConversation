#!/usr/bin/env python3
"""
Computer WER between a reference and hypothesis text file, both in the `UttID text` format:
    <UttID> <transcript>
    <UttID> <transcript>
    ...

Usage:
    python compute_wer.py --ref ref_text.txt --hyp hyp_text.txt --normalizer basic

Requirements:
    pip install jiwer
"""

import argparse
from pathlib import Path

import jiwer

def load_text_file(path: Path) -> dict:
    texts = {}
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_no}: expected '<id> <text>', got: {line!r}")
            full_id, text = parts
            if full_id in texts:
                raise ValueError(f"{path}:{line_no}: duplicate id '{full_id}'")
            texts[full_id] = text
    return texts

def build_normalizer(kind: str):
    if kind == "none":
        return lambda x: x
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    if kind == "basic":
        return BasicTextNormalizer()
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    return EnglishTextNormalizer({})

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ref", required=True, type=Path,
                   help="Reference filepath")
    p.add_argument("--hyp", required=True, type=Path,
                   help="Hypothesis filepath")
    p.add_argument("--normalizer", choices=["none", "basic", "english"], default="none",
                   help="Text normalization applied to both ref and hyp before scoring. "
                         "Must match whatever was used to produce --hyp.")
    p.add_argument("--per-utterance", type=Path, default=None,
                   help="Optional path to write per-utterance WER as '<id> <wer>' lines.")
    return p.parse_args()

def main():
    args = parse_args()

    ref = load_text_file(args.ref)
    hyp = load_text_file(args.hyp)

    ref_ids, hyp_ids = set(ref), set(hyp)
    only_ref = sorted(ref_ids - hyp_ids)
    only_hyp = sorted(hyp_ids - ref_ids)

    if only_ref:
        print(f"[warn] {len(only_ref)} id(s) in ref but not hyp, excluded, e.g., {only_ref[0]}")
    if only_hyp:
        print(f"[warn] {len(only_hyp)} id(s) in hyp but not ref, excluded, e.g., {only_hyp[0]}")

    common_ids = sorted(ref_ids & hyp_ids)
    if not common_ids:
        raise SystemExit("No overlapping ids between ref and hyp -- nothing to score.")

    normalize = build_normalizer(args.normalizer)
    ref_texts = [normalize(ref[i]) for i in common_ids]
    hyp_texts = [normalize(hyp[i]) for i in common_ids]

    out = jiwer.process_words(ref_texts, hyp_texts) # corpus-level: pooled across all utterances
    print(f"Utterances scored: {len(common_ids)}")
    print(f"WER: {out.wer*100:.2f}%")
    print(f"Substitutions: {out.substitutions}")
    print(f"Deletions: {out.deletions}")
    print(f"Insertions: {out.insertions}")
    print(f"Hits: {out.hits}")

    if args.per_utterance:
        with open(args.per_utterance, "w", encoding="utf-8") as f:
            for full_id, r, h, in zip(common_ids, ref_texts, hyp_texts):
                u = jiwer.process_words([r], [h]) # single-utterance WER
                f.write(f"{full_id} {u.wer*100:.2f}%\n")
        print(f"Per-utterance WER -> {args.per_utterance}")

if __name__ == "__main__":
    main()