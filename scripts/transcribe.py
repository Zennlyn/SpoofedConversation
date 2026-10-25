#!/usr/bin/env python3
"""
Batch-transcribe the generated audio utterances using Whisper large-v3. The output format is as below:
    <UttID> <transcript>
    <UttID> <transcript>
    ...

Requirements
------------
    pip install "transformers>=4.39" torch torchaudio soundfile tqdm

Example Usage:
    python tts_evaluation.py \
        --audio-dir /workspace/lps/wav_audio/train/f5tts \
        --output /workspace/lps/wav_audio/train/f5tts/transcripts/f5tts.txt \
        --model-id openai/whisper-large-v3 \
        --language en \
        --batch-size 8 \
        --normalizer basic
"""

import argparse
import logging
import sys
import os
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import pipeline

# Loads HF_TOKEN from the nearest .env (searched upward from this file)
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tts_evaluation")

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--audio-dir", required=True, type=Path,
                   help="Root directory of generated audio.")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--model-id", default="openai/whisper-large-v3",
                   help="HF model id or local path.")
    p.add_argument("--device", default=None, help="cuda / cuda:0 / cpu.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--language", default=None,
                   help="Force decoding language (e.g. 'en). Omit for auto-detect")
    p.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    p.add_argument("--chunk-length-s", type=float, default=30.0,
                   help="Chunk length for long-form audio.")
    p.add_argument("--normalizer", choices=["none", "basic", "english"], default="basic")
    p.add_argument("--resume", action="store_true",
                   help="Skip full_ids already present in an existing --output file instead of overwriting.")
    return p.parse_args()

# ---------- Helper Functions ----------
def build_normalizer(kind: str):
    if kind == "none":
        return lambda x: x
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    if kind == "basic":
        return BasicTextNormalizer()
    # "english": adds English-specific handling (contractions, number/ordinal
    # spelling, abbreviation expansion) on top of BasicTextNormalizer, matching
    # the convention used in whisper's own English WER evaluations. The optional
    # spelling-variant mapping (english.json) is a nice-to-have, not required.
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    from transformers.utils import cached_file
    import json
    english_spelling_mapping = {}
    try:
        mapping_path = cached_file("openai/whisper-large-v3", "english.json")
        with open(mapping_path, encoding="utf-8") as f:
            english_spelling_mapping = json.load(f)
    except Exception:
        log.warning("Could not fetch english.json spelling-variant mapping; "
                    "proceeding with EnglishTextNormalizer's default rules only.")
    return EnglishTextNormalizer(english_spelling_mapping)

def validate_readable(ids: list, paths: list):
    """Drop files soundfile can't open so one bad file doesn't kill the whole run."""
    good_ids, good_paths, skipped = [], [], []
    for full_id, path in tqdm(zip(ids, paths), total=len(ids), desc="Validating audio files"):
        try:
            sf.info(str(path))  # cheap header read, no full decode
            good_ids.append(full_id)
            good_paths.append(path)
        except Exception as e:
            skipped.append((full_id, str(path), str(e)))
    return good_ids, good_paths, skipped

def load_existing_ids(output_path: Path):
    if not output_path.exists():
        return set()
    ids = set()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            ids.add(line.split(" ", 1)[0])
    return ids

def main():
    args = parse_args()

    if not args.audio_dir.is_dir():
        log.error(f"audio_dir does not exist of is not a directory: {args.audio_dir}")
        sys.exit(1)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if "cuda" in device else torch.foat32
    log.info(f"Loading {args.model_id} on {device} ({dtype}) ...")

    asr = pipeline(
        "automatic-speech-recognition",
        model=args.model_id,
        torch_dtype=dtype,
        device=device,
        chunk_length_s=args.chunk_length_s
    )

    generate_kwargs = {"task": args.task}
    if args.language:
        generate_kwargs["language"] = args.language

    normalize = build_normalizer(args.normalizer)

    paths = sorted(args.audio_dir.glob("*.wav"))
    ids_sorted = [p.stem for p in paths]
    log.info(f"Found {len(ids_sorted)} .wav files under {args.audio_dir}")

    ids_sorted, paths, skipped = validate_readable(ids_sorted, paths)
    if skipped:
        skip_path = args.output.with_suffix(args.output.suffix+".skipped.txt")
        with open(skip_path, "w", encoding="utf-8") as f:
            for full_id, path, err in skipped:
                f.write(f"{full_id}\t{path}\t{err}\n")
        log.warning(f"Skipped {len(skipped)} unreadable files -> {skip_path}")

    if args.resume:
        done_ids = load_existing_ids(args.output)
        pairs = [(i, p) for i, p in zip(ids_sorted, paths) if i not in done_ids]
        log.info(f"Resume: {len(ids_sorted) - len(pairs)} already done, {len(pairs)} remaining")
        ids_sorted = [i for i, _ in pairs]
        paths = [p for _, p in pairs]
        out_mode = "a"
    else:
        out_mode = "a"

    if not ids_sorted:
        log.info("Nothing to transcribe")
        return

    paths_sorted = [str(p) for p in paths]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = asr((p for p in paths_sorted), batch_size=args.batch_size,
                  generate_kwargs=generate_kwargs)
    
    n_written = 0
    with open(args.output, out_mode, encoding="utf-8") as f:
        for full_id, res in tqdm(zip(ids_sorted, results), total=len(ids_sorted),
                                 desc="Transcribing"):
            text = " ".join(res["text"].split()) # Collapse whitespace/newlines
            text = normalize(text)
            f.write(f"{full_id.split("_", 1)[1]} {text}\n")
            n_written += 1

    log.info(f"Wrote {n_written} transcripts -> {args.output}")
    if skipped:
        log.info(f"({len(skipped)} files skipped, see {args.output.with_suffix(args.output.suffix + '.skipped.txt')})")

if __name__ == "__main__":
    main()