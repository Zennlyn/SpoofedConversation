#!/usr/bin/env python3
"""
Resample every wav in a directory to 16 kHz mono.

Usage:
    python downsample.py <input_dir> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

TARGET_SR = 16000

def main(in_dir, out_dir):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    files = sorted(in_dir.rglob("*.wav"))
    print(f"{len(files)} files -> {out_dir}")

    for i, f in enumerate(files, 1):
        x, sr = sf.read(str(f), dtype="float64", always_2d=True) # eliminating the need for branching x.ndim
        x = x.mean(axis=1) # mono

        if sr != TARGET_SR:
            x = soxr.resample(x, sr, TARGET_SR, quality="VHQ")

        peak = np.max(np.abs(x)) if x.size else 0.0
        if peak > 1.0:
            x = x / peak * (1-1e-6)

        out = out_dir/f.relative_to(in_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), x, TARGET_SR, subtype="PCM_16")

        if i % 200 == 0 or i == len(files):
            print(f"{i}/{len(files)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python downsample.py <input_dir> <outdir>")
    main(sys.argv[1], sys.argv[2])