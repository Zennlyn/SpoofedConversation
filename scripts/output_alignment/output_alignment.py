from collections import defaultdict
import math

def output_alignment(detout, diarout):
    aligned_outputs = defaultdict(list)
    # detout contains some utterances that are not in diarout
    for uttid in list(diarout.keys()): 
        aligned = []
        seg_label = detout[uttid]
        diar = diarout[uttid]

        for seg in diar:        
            st, et = seg["start"], seg["end"]
            start_idx = math.floor(st/0.02)
            end_idx = min(math.floor(et/0.02), len(seg_label)-1) # i -> Index+1
            label = seg_label[start_idx]
            seg_start = st
            for i in range(start_idx, end_idx+1):
                if seg_label[i] != label:
                    seg_end = min(st + (i - start_idx + 1)*0.02, et)
                    aligned_outputs[uttid].append({
                        "start": round(seg_start, 3),
                        "end": round(seg_end, 3),
                        "label": label,
                        "speaker_id": seg["speaker_id"]
                    })
                    label = seg_label[i]
                    seg_start = seg_end
            aligned_outputs[uttid].append({
                "start": round(seg_start, 3),
                "end": round(et, 3),
                "label": label,
                "speaker_id": seg["speaker_id"]
            })
    return aligned_outputs