#!/usr/bin/env python3
"""
Trim leading and trailing non-speech using Silero VAD.
Internal silence is left untouched.

Usage:
    python trim_silence.py <input_dir> <output_dir>

Notes:
    - A 250 ms padding is added before the [start] of the first speech region and
      after the [end] of the last speech region.
    - The trimming information is recorded in the generated CSV file.
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import csv
from silero_vad import load_silero_vad, get_speech_timestamps

PAD_MS = 250.0
VAD_SR = 16000

THRESHOLD=0.5

MIN_SPEECH_DURATION_MS = 100
MIN_SILENCE_DURATION_MS = 100

def load_audio(path):
    """
    Returns:
        x: mono float32 numpy array
        sr: original sample rate
    """
    x, sr = sf.read(str(path), dtype="float32")

    # convert stereo/multichannel to mono.
    if x.ndim > 1:
        x = x.mean(axis=1)

    if sr != VAD_SR:
        raise ValueError(
            f"Expected {VAD_SR} Hz audio, but got {sr} Hz"
        )
    
    return x, sr

def speech_bounds(x, sr, model):
    """
    Return first/last speech boundary in original sample coordinates.
    Returns:
        (start_sample, end_sample)
        or None if no speech is detected.
    """
    if x.size == 0:
        return None

    speech = get_speech_timestamps(
        x,
        model,
        sampling_rate=sr,
        threshold=THRESHOLD,
        min_speech_duration_ms=MIN_SPEECH_DURATION_MS,
        min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
        return_seconds=True,
    )

    if not speech:
        return None

    speech_start_sec = speech[0]["start"]
    speech_end_sec = speech[-1]["end"]

    start = int(round(speech_start_sec*sr))
    end = int(round(speech_end_sec*sr))

    start = max(0, min(start, x.size))
    end = max(start, min(end, x.size))

    # Apply 250 ms padding
    pad = int(round(sr*PAD_MS/1000.0))

    start = max(0, start-pad)
    end = min(x.size, end+pad)

    return start, end

def main(in_dir, out_dir):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)

    files = sorted(in_dir.rglob("*.wav"))

    if not files:
        print(f"No WAV files found in {in_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(files)} files -> {out_dir}") 
    print(f"VAD sample rate: {VAD_SR} Hz") 
    print(f"Padding: {PAD_MS:.0f} ms") 
    print(f"VAD threshold: {THRESHOLD}") 

    # Load Silero once. 
    print("Loading Silero VAD model...") 
    model = load_silero_vad() 
    print("Model loaded.")

    rows = []

    for i, f in enumerate(files, 1):
        try:
            x, sr = load_audio(f)
            b = speech_bounds(x, sr, model)

            if b is None:
                # Do not destroy files where VAD found nothing
                start = 0
                end = x.size
                note = "no speech found, kept whole file"
            else:
                start, end = b
                note = ""

            y = x[start:end]

            out = out_dir / f.relative_to(in_dir)
            out.parent.mkdir(parents=True, exist_ok=True)

            # Write as PCM16
            sf.write(str(out), y, sr, subtype="PCM_16")

            rows.append({
                "file": str(f.relative_to(in_dir)),
                "orig_sec": round(x.size/sr, 3),
                "head_sec": round(start/sr, 3),
                "tail_sec": round((x.size-end)/sr, 3),
                "out_sec": round(y.size/sr, 3),
                "note": note,
            })
        except Exception as e:
            print(f"\nERROR: {f}") 
            print(f" {type(e).__name__}: {e}") 
            
            rows.append({ 
                "file": str(f.relative_to(in_dir)), 
                "orig_sec": "", 
                "head_sec": "", 
                "tail_sec": "", 
                "out_sec": "", 
                "note": f"ERROR: {type(e).__name__}: {e}", 
            }) 
            
        if i % 200 == 0 or i == len(files): 
            print(f" {i}/{len(files)}")

    csv_path = out_dir / "trim_offsets.csv" 
    with open(csv_path, "w", newline="") as fh: 
        fieldnames = [ "file", "orig_sec", "head_sec", "tail_sec", "out_sec", "note"] 
        writer = csv.DictWriter(fh, fieldnames=fieldnames) 
        writer.writeheader() 
        writer.writerows(rows)

    # Summary
    valid = [r for r in rows if isinstance(r["head_sec"], (int, float))] 
        
    if valid: 
        head = np.array([r["head_sec"] for r in valid]) 
        tail = np.array([r["tail_sec"] for r in valid]) 
        print("-------") 
        print("Summary") 
        print("-------") 
        print(f"Files processed: {len(rows)}") 
        print(f"Successful: {len(valid)}") 
        print(f"Head removed: median {np.median(head):.3f}s, max {head.max():.3f}s") 
        print(f"Tail removed: median {np.median(tail):.3f}s, max {tail.max():.3f}s") 
        print(f"CSV: {csv_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(
            "usage: python trim_silence.py <input_dir> <output_dir>"
        )
    main(sys.argv[1], sys.argv[2])