import argparse
import numpy as np
from copy import deepcopy

from pyannote.database.util import load_rttm
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.identification import IdentificationErrorRate

parser = argparse.ArgumentParser(add_help=False)

parser.add_argument('--ref_rttm', type=str, default="./examples/diarization_label.rttm",
                    help="Filepath to the ground-truth RTTM file")
parser.add_argument('--bona_ref_rttm', type=str, default="./examples/bonafide_only.rttm",
                    help="Filepath to the ground-truth bonafide-only RTTM file")
parser.add_argument('--spoof_ref_rttm', type=str, default="./examples/spoof_only.rttm",
                    help="Filepath to the ground-truth spoof-only RTTM file")
parser.add_argument('--aligned_output', type=str, default='./examples/aligned_output.npy',
                    help="Filepath to the aligned output (framework) file")

args = parser.parse_args()

ref_rttm = load_rttm(args.ref_rttm)
bona_ref_rttm = load_rttm(args.bona_ref_rttm)
spoof_ref_rttm = load_rttm(args.spoof_ref_rttm)

output = np.load(args.aligned_output, allow_pickle=True).item()
print(f"LEN OUTPUT= {len(output)}")

def pop_label(output):
    for out in output.values():
        for seg in out:
            seg.pop('label', None)

def separate_bonaspoof(output):
    bonafide = {key: [deepcopy(out) for out in output[key] if out["label"]==1] for key in output.keys()}
    spoof = {key: [deepcopy(out) for out in output[key] if out["label"]==0] for key in output.keys()}
    pop_label(bonafide)
    pop_label(spoof)
    return bonafide, spoof

def get_annotations(output):
    output_annotations = {}
    for key in output.keys():
        out_annot = Annotation(uri=key)
        for seg in copied_output[key]:
            segment = Segment(seg["start"], seg["end"])
            out_annot[segment] = seg["speaker_id"]
        output_annotations[key] = out_annot
    return output_annotations

def get_annotation(output):
    output_annotation = Annotation()
    for seg in output:
        segment = Segment(seg["start"], seg["end"])
        output_annotation[segment] = seg["speaker_id"]
    return output_annotation

def map_speaker_id(key, output):
    try:
        spk_map = mapping(reference=ref_rttm.get(key), hypothesis=output_annotation.get(key))
    except:
        print(key)
        print(ref_rttm.get(key))
        print(ref_rttm)
        print(output_annotation.get(key))
        raise

    for out in output:
        if out["speaker_id"] in spk_map.keys():
            out["speaker_id"] = spk_map[out["speaker_id"]]
        else:
            out["speaker_id"] = '0000'

    return output
    
bona_out, spoof_out = separate_bonaspoof(output)

copied_output = {key: deepcopy(output[key]) for key in output.keys()}
pop_label(copied_output)

output_annotation = get_annotations(copied_output)

# separated because it saves the performance result in metric.results_
metric_bona = IdentificationErrorRate()
metric_spoof = IdentificationErrorRate()
metric_all = DiarizationErrorRate()

mapping = DiarizationErrorRate().optimal_mapping

for key in copied_output.keys():
    bona_hyp = map_speaker_id(key, bona_out[key])
    spoof_hyp = map_speaker_id(key, spoof_out[key])
    hyp = bona_hyp + spoof_hyp

    bona_annotation = get_annotation(bona_hyp)
    spoof_annotation = get_annotation(spoof_hyp)
    hypo_annotation = get_annotation(hyp)

    bona_ref = bona_ref_rttm.get(key) if bona_ref_rttm.get(key) else Annotation()
    spoof_ref = spoof_ref_rttm.get(key) if spoof_ref_rttm.get(key) else Annotation()
    hypo_ref = ref_rttm.get(key) if ref_rttm.get(key) else Annotation()
    
    bona_der = metric_bona(bona_ref, bona_annotation)
    spoof_der = metric_spoof(spoof_ref, spoof_annotation)
    der = metric_all(hypo_ref, hypo_annotation)

print(f"DER: {round(abs(metric_all)*100, 2)}")
print(f"FA: {metric_all.accumulated_['false alarm']/metric_all.accumulated_['total']*100:.2f}%")
print(f"MD: {metric_all.accumulated_['missed detection']/metric_all.accumulated_['total']*100:.2f}%")
print(f"Conf: {metric_all.accumulated_['confusion']/metric_all.accumulated_['total']*100:.2f}%")

print("=============================================")
print(f"ADER_bona: {round(abs(metric_bona)*100, 2)}")
print(f"FA: {metric_bona.accumulated_['false alarm']/metric_bona.accumulated_['total']*100:.2f}%")
print(f"MD: {metric_bona.accumulated_['missed detection']/metric_bona.accumulated_['total']*100:.2f}%")
print(f"Conf: {metric_bona.accumulated_['confusion']/metric_bona.accumulated_['total']*100:.2f}%")

print("=============================================")
print(f"ADER_spoof: {round(abs(metric_spoof)*100, 2)}")
print(f"FA: {metric_spoof.accumulated_['false alarm']/metric_spoof.accumulated_['total']*100:.2f}%")
print(f"MD: {metric_spoof.accumulated_['missed detection']/metric_spoof.accumulated_['total']*100:.2f}%")
print(f"Conf: {metric_spoof.accumulated_['confusion']/metric_spoof.accumulated_['total']*100:.2f}%")

np.savez("/workspace/archive/output_tobe_evaluated/ADER_seed5_tdlnemo_clean.npz",
         metric_bona=metric_bona.results_,
         metric_spoof=metric_spoof.results_,
         metric_all=metric_all.results_)