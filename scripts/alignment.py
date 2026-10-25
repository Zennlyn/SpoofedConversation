import argparse
import numpy as np
from collections import defaultdict
from pyannote.database.util import load_rttm
from output_alignment.output_alignment import output_alignment

to_not_include = ["test_conv_0000370", "test_conv_0000413", "test_conv_0000440", "test_conv_0000459"] # 5 spks

parser = argparse.ArgumentParser(add_help=False)

parser.add_argument('--detection_output', type=str, required=True)
parser.add_argument('--eer_threshold', type=str, required=True)
parser.add_argument('--diarization_output', type=str, required=True)
parser.add_argument('--save_path', type=str, default='./aligned.npy')

args = parser.parse_args()

detout = defaultdict(list)
diarout = defaultdict(list)

threshold = np.load(args.eer_threshold, allow_pickle=True).get("threshold_scales")[0][1]
spoof = np.load(args.detection_output, allow_pickle=True)
diar = load_rttm(args.diarization_output)

for key in to_not_include:
    diar.pop(key, None)

for out in spoof:
    uttid = out[0][0]
    detout[uttid] = (out[:,2].astype(float) >= threshold).astype(int)

for uttid in list(diar.keys()):
    for seg in diar[uttid]:
        diarout[uttid].append({
            "start": round(seg[0].start, 3),
            "end": round(seg[0].end, 3),
            "speaker_id": seg[1]
        })

aligned = output_alignment(detout, diarout)
np.save(args.save_path, aligned, allow_pickle=True)