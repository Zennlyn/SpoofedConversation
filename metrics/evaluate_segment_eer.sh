#!/bin/bash

python segment_eer.py \
    --output_name trained_singlereso2_proper_100epochs_lps \
    --save_dir /workspace/outputs/ \
    --ref_rttm /workspace/llama_dataset/rttm/eval.rttm \
    --reco2dur_file /workspace/llama_dataset/reco2dur/reco2dur_eval.txt \
    --keyword _2_ \
    --save_name EER_trained_singlereso_lps

python segment_eer.py \
    --output_name trained_singlereso2_proper_100epochs_sc \
    --save_dir /workspace/outputs/ \
    --ref_rttm /workspace/spoofedconversation/rttm/eval.rttm \
    --reco2dur_file /workspace/spoofedconversation/reco2dur/reco2dur_eval.txt \
    --keyword _2_ \
    --speech_only /workspace/spoofedconversation/speech_regions.npy\
    --save_name EER_trained_singlereso_speech_only

# SC
echo "Computing segmentEER on SC";
python segment_eer.py \
    --output_name trained_singlereso2_proper_100epochs_sc \
    --save_dir /workspace/outputs/ \
    --ref_rttm /workspace/spoofedconversation/rttm/eval.rttm \
    --reco2dur_file /workspace/spoofedconversation/reco2dur/reco2dur_eval.txt \
    --keyword _2_ \
    --save_name EER_trained_singlereso_sc