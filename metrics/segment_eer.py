#!/usr/bin/env python

# Copyright 2023 National Institute of Informatics (author: Lin Zhang, zhanglin@nii.ac.jp)
# Licensed under the BSD 3-Clause License.
#
# Modified by Kelvin, 2026, from
# https://github.com/nii-yamagishilab/PartialSpoof/blob/main/metric/SegmentEER.py

"""
Calculating point-based segment-level EER for spoof localization. 
mean for round with different random seeds
(use spicial to limit the pkl.)

As mentioned in Sec. 4.3 of [Range-Based Equal Error Rate for Spoof Localization]
(https://arxiv.org/pdf/2305.17739.pdf), we can up-sample or down-sample the predicted scores to other reso.
This function is used to downsample/upsample the predicted score as shown in Fig.2 

Upsample: extend this to get finer prediction from coaser one. like upsample 20ms to 40ms
20ms: |--0.3--|--0.8--|
40ms: |0.3|0.3|0.8|0.8|

Downsample: extend this to get finer prediction from coaser one. like upsample 20ms to 40ms
40ms: |0.3|0.6|0.8|0.9|
20ms: |--0.3--|--0.8--|

Requires model_dir and args.save_dir

usage: $: python segment_eer.py --output_name <output_name> --save_dir <save_dir> --ref_rttm <ref_rttm.rttm> --reco2dur_file <reco2dur_file.txt>
            -- keyword <_key_> --speech_only <speech-only_output.npy> --save_name <save_name>
"""

import os
import sys
import pickle
import librosa
import argparse
import math
import numpy as np
import pandas as pd
import scipy.io.wavfile as sciwav

from numpy import nan
from tqdm import tqdm
from eer import compute_eer
from collections import defaultdict
from rttm_tool import get_rttm, rttm2vadvec
np.set_printoptions(linewidth=100_000)

def str2bool(value):
    """
    A function to convert string to bool value.
    """
    if value.lower() in {'True', 'yes', 'true', 't', 'y', '1'}:
        return True
    if value.lower in {'False', 'no', 'false', 'f', 'n', '0'}:
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(add_help=False)
# Necessary Inputs
parser.add_argument('--output_name', type=str, default='output',
                    help="The name of the output")
parser.add_argument('--save_name', type=str, default='segmentEER_result')
parser.add_argument('--save_dir', type=str, default="/workspace/outputs")
parser.add_argument('--ref_rttm', type=str, default="/workspace/spoofedconversation/rttm/eval.rttm",
                    help="Path to the ground-truth timestamp annotations file in the rttm format.")
parser.add_argument('--reco2dur_file', type=str, default="/workspace/spoofedconversation/reco2dur/reco2dur_eval.txt",
                    help="Path to the file consisting duration of each utterance")
parser.add_argument('--speech_only', type=str, default='',
                    help="To evaluate only speech regions in the conversation.")
parser.add_argument('--label2num_file', type=str, default="/workspace/spoofedconversation/label2num_file")
parser.add_argument('--keyword', type=str, default='',
                    help="Keyword for filtering score_ali pkl files, e.g., for specific dates (2021) or scoring branches (_2_)")

# Load Existing Result Options
parser.add_argument('--use_ext_flag', type=str2bool, default=False,
                    help="Load existing results directly instead of recalculating.")

# Evaluation Options
parser.add_argument('--DIGEER', type=str2bool, default=True,
                    help="Calculate the diagonal of the EER matrix. (Measuring performance in the same resolution used during training.)")
parser.add_argument('--UpEER', type=str2bool, default=True,
                    help="Calculate the upper triangle of the EER matrix. (Downsample predicted scores to coarse-grained resolution to measure.)")
parser.add_argument('--LowEER', type=str2bool, default=True,
                    help="Calculate the lower triangle of the EER matrix. (Upsample predicted scores to fine-grained resolution to measure.)")
parser.add_argument('--UTTEER', type=str2bool, default=True,
                    help="Calculate EER for utterance-level spoof detection using segment-level scores (displayed in the last column).")
args = parser.parse_args()

scale_num=7
SSL_shift=1 # since SSL use 20ms as frame shift
Base_step=0.01 # base in sec
Pred_Resolutions = np.array([pow(2, i) for i in np.arange(scale_num)])[SSL_shift:]
Measure_Resolutions = np.array([pow(2, i) for i in np.arange(scale_num)])[:] # default: 10 20 40 80 160 320 640

rttm = get_rttm(args.ref_rttm)
reco2dur = dict([line.split() for line in open(args.reco2dur_file)])
LAB2NUM = dict([[line.split()[0], float(line.split()[1])] for line in open(args.label2num_file)])

assert(args.DIGEER + args.UpEER + args.LowEER > 0)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

def load_scolab(fpath):
    """
    Load sco_ali pkl, and save it to pandas.DataFrame, which is for further upsample/downsample.
    """
    print("loading "+fpath)
    sco_lab = np.load(fpath, allow_pickle=True)
    try:
        sco_lab = np.concatenate(sco_lab)
    except:
        print(sco_lab)
        raise

    sco_lab_pd = pd.DataFrame(sco_lab,
                           columns=('Wavid', 'Predict_Score_0', 'Predict_Score_1', 'True_Class'))
    sco_lab_pd = sco_lab_pd.astype({"Wavid": str, "True_Class": float,
                                    "Predict_Score_0": float, "Predict_Score_1": float})
    return sco_lab_pd

def cal_eer_by_scolab(sco_lab_pd, SpoofRatio=False, utteer_flag=False):
    """
    Normal form is for eer calculation on all data.
    utteer_flag = True for only one utterance's score.
    """
    if 'True_Class'in sco_lab_pd.columns:
        gen=sco_lab_pd[sco_lab_pd.True_Class==LAB2NUM['bonafide']].Predict_Score_1
        spf=sco_lab_pd[sco_lab_pd.True_Class==LAB2NUM['spoof']].Predict_Score_1
    elif 'True_SpoofRatio' in sco_lab_pd.columns:
        sco_lab_pd['True_SpoofRatio']=sco_lab_pd['True_SpoofRatio'].astype(float)
        gen=sco_lab_pd[sco_lab_pd.True_SpoofRatio==0.0].Predict_Score_1
        spf=sco_lab_pd[sco_lab_pd.True_SpoofRatio>0.0].Predict_Score_1

    eer, threshold = compute_eer(gen, spf)
    # if np.isnan(eer):
    #     print(gen)
    #     print("===")
    #     print(spf)
    if utteer_flag:
        return eer
        
    return eer, threshold

def get_pd_in_scoretype(sco_lab_pd, fs, SCORE_TYPE='min', return_eer=False):
    """
    This function is used to downsample the predicted score,
    using SCORE_TYPE to choose score, and with the sliding window (frame_shift) as fs
    """
    fs = min(len(sco_lab_pd), fs) # For the case when fs > len(pd).
    if SCORE_TYPE == 'mean':
        lab = sco_lab_pd['True_Class'].rolling(window=fs).min()[fs-1::fs]
        score = sco_lab_pd['Predicted_Score_1'].rolling(window=fs).mean()[fs-1::fs]
        new_pd = pd.concat([lab, score], axis=1)
    elif SCORE_TYPE == 'max':
        lab = sco_lab_pd['True_Class'].rolling(window=fs).min()[fs-1::fs]
        score = sco_lab_pd['Predicted_Score_1'].rolling(window=fs).max()[fs-1::fs]
        new_pd = pd.concat([lab, score], axis=1)
    else: # default using min
        new_pd = sco_lab_pd[['Predict_Score_1', 'True_Class']].rolling(window=fs).min()[fs-1::fs]

    if len(new_pd) < (len(sco_lab_pd)/fs):
        # For the rest part, longer than shift windows.
        t = sco_lab_pd[['Predict_Score_1', 'True_Class']][int(fs*(len(sco_lab_pd)//fs)):].min()
        new_pd = pd.concat([new_pd, t.set_axis(new_pd.columns).to_frame().T], ignore_index=True)

    return new_pd

def get_dupary_by_scale(sco_lab_pd, scale, to_scale, scale_change_time=0):
    """
    To upsample the predicted score to a finer-grained reso, we need to duplicate them
    """
    uttid = np.unique(sco_lab_pd['Wavid'])[0]
    # Duplicate for score
    scale_change_time = scale_change_time or (scale/to_scale)
    new_pd = pd.DataFrame(np.repeat(sco_lab_pd.to_numpy(), scale_change_time, axis=0),
                          columns=sco_lab_pd.columns)

    # Get new ground-truth based on the new scale
    gt_lab = rttm2vadvec(rttm[uttid], to_scale*Base_step, reco2dur[uttid], skip_last=False,
                         label2num_file=args.label2num_file)
    lab_num = min(len(new_pd), len(gt_lab))
    del_num = max(len(new_pd), len(gt_lab))-lab_num
    pad = [nan for i in np.arange(del_num)] if (len(new_pd)>len(gt_lab)) else []

    new_pd['True_Class'] = np.hstack((gt_lab[:lab_num], pad))
    new_pd=new_pd.dropna().reset_index(drop=True)

    return new_pd['Predict_Score_1'], new_pd['True_Class']

def get_duppd_by_scale(sco_lab_pd, scale, to_scale, scale_change_time=0):
    uttid = np.unique(sco_lab_pd['Wavid'])[0]
    # Duplicate for score
    scale_change_time = scale_change_time or (scale/to_scale)
    new_pd = pd.DataFrame(np.repeat(sco_lab_pd.to_numpy(), scale_change_time, axis=0),
                          columns=sco_lab_pd.columns)

    # Get new ground-truth based on the new scale
    gt_lab = rttm2vadvec(rttm[uttid], to_scale*Base_step, reco2dur[uttid], skip_last=False,
                         label2num_file=args.label2num_file)
    lab_num = min(len(new_pd), len(gt_lab))
    del_num = max(len(new_pd), len(gt_lab))-lab_num
    pad = [nan for i in np.arange(del_num)] if (len(new_pd)>len(gt_lab)) else []

    new_pd['True_Class'] = np.hstack((gt_lab[:lab_num], pad))
    new_pd=new_pd.dropna().reset_index(drop=True)

    return new_pd

def get_speech_only(new_pd, reso):
    vad_out = np.load(args.speech_only, allow_pickle=True).item()
    vad_out_idx = {}
    for fname in vad_out.keys():
        vad_out_idx[fname] = [(math.floor(t[0]/reso), math.ceil(t[1]/reso)) for t in vad_out[fname]]
    if 'Wavid' in new_pd.columns:
        new_pd = new_pd.set_index(["Wavid", new_pd.groupby('Wavid').cumcount()])
    speech_pd = pd.DataFrame(columns=["Predict_Score_1", "True_Class"])
    speech_pd.index = pd.MultiIndex(levels=[[], []],
                                    codes=[[], []])
    for fname in vad_out.keys():
        temp = pd.concat([new_pd.loc[fname].iloc[sid:eid+1] for sid, eid in vad_out_idx[fname]]).reset_index(drop=True)
        temp.index = pd.MultiIndex.from_arrays([[fname]*len(temp), range(len(temp))])
        speech_pd = pd.concat([speech_pd, temp])

    return speech_pd
    
def get_eermatrix_scale(sco_lab_pd, scale):
    """
    This function is used to derive the EER matrix,
    EERs in the diagonal represent the EER for the corresponding resolution.
    EERs in the upper triangle
    And note that upper triangle may be an "underestimation", while lower triangle is more correct.
    """
    eers=np.zeros(len(Measure_Resolutions))
    thresholds=np.zeros(len(Measure_Resolutions))

    index = Measure_Resolutions.tolist().index(scale)
    measure_scale = np.arange(index, len(Measure_Resolutions))
    if args.LowEER:
        measure_scale = np.arange(scale_num)
    new_pd = sco_lab_pd
    for idx in measure_scale:
        if (idx==index) and args.DIGEER:
            print("The same resolution")
            new_pd = sco_lab_pd.copy()
            if args.UTTEER:
                utteer_pd = sco_lab_pd.groupby('Wavid').apply(cal_eer_by_scolab, utteer_flag=True)
                utteer_pd = pd.DataFrame(utteer_pd).reset_index()
                utteer_pd.rename(columns={0: 'EER'}, inplace=True)
                print(utteer_pd)
        elif (idx > index) and args.UpEER:
            # If we want to calculate eer in the upper diagonal
            fs = int(Measure_Resolutions[idx]/scale)
            # we need to divide scale, since the further eer calculation is based on frame shift on basic scale.
            # any function passed to apply is applied along each column/series independent

            # we only use Predict_Score_1, so we only need to consider one column
            # and since here we use Cosine similarity, so higher means more similar, Score_1 for similar to bonafide, smaller Score_1 means spoof.
            new_pd = sco_lab_pd.groupby('Wavid').apply(get_pd_in_scoretype, fs=fs)
        elif (idx < index) and args.LowEER:
            new_pd = sco_lab_pd.groupby('Wavid').apply(get_duppd_by_scale, scale, to_scale=Measure_Resolutions[idx])
        else:
            continue
        # Evalute speech regions only, remove non-speech's scores
        if args.speech_only:
            if (Measure_Resolutions[idx] != scale) and ('Wavid' in new_pd.columns):
                new_pd = new_pd.drop(columns='Wavid')
            speech_pd = get_speech_only(new_pd, Measure_Resolutions[idx]*0.01)
            eers[idx], thresholds[idx] = cal_eer_by_scolab(speech_pd)
        else:
            eers[idx], thresholds[idx] = cal_eer_by_scolab(new_pd)

    if args.UTTEER:
        return eers, thresholds, utteer_pd

    return eers, thresholds, ''

def get_utteer_by_seg(sco_lab_pd):
    """
    Measure Utterance EER based on segment-level score vectors.
    For each utterance, using min() to choose the minimum segment score as the utterance score.
    """
    eer_utt, threshold_utt = cal_eer_by_scolab(sco_lab_pd.groupby(['Wavid']).min())

    return eer_utt, threshold_utt

def get_mean_multi_seed(eers_scales_seed, threshold_scales_seed, return_var=False):
    """
    Calculate average of EERs with different initial random seeds.
    """
    if return_var:
        return np.mean(eers_scales_seed, axis=0), np.mean(threshold_scales_seed, axis=0), \
                np.var(eers_scales_seed, axis=0), np.var(threshold_scales_seed, axis=0)

    return np.mean(eers_scales_seed, axis=0), np.mean(threshold_scales_seed, axis=0), None, None
def print_res(res_data):
    """
    Print results
    """
    eers_scales_seed_mean, threshold_scales_seed_mean, eers_scales_seed_var = get_mean_multi_seed(res_data['eers_scales_seed'], res['threshold_scales_seed'], return_var=args.print_var)
    if args.print_each:
        print('-'*20+'eer for each res')
        print(res_data['eers_scales_seed']*100)
        print('-'*20+'threshold for each res')
        print(res_data['threshold_scales_seed'])

    print('-'*20+'mean')
    print(eers_scales_seed_mean*100)
    if args.print_var:
        print('-'*20+'var')
        print(eers_scales_seed_var*100)

def main():
    np.set_printoptions(linewidth=100_000)

    eer_file=f'{os.path.join(args.save_dir, args.save_name)}.npz'

    if os.path.exists(eer_file) and args.use_ext_flag:
        res_data = np.load(eer_file, allow_pickle=True)
        print("="*10, "{} seed".format(len(res_data['eers_scales_seed'])), '='*10)
        print_res(res_data)
    else:
        model_out_dir = args.save_dir
        eer_scales, threshold_scales, utteer_scales = [], [], {}
        if os.path.exists(model_out_dir):
            # to check whether the aligned score .pkl files exist.
            find_it=False
            while True:
                # to find the time_tag based on the ali.npz
                for score_name in os.listdir(model_out_dir):
                    name, ext = os.path.splitext(score_name)
                    if name.find(args.keyword)>=0 and name.find(f"{args.output_name}_score_ali")>=0 and ext=='.pkl':
                        find_it=True
                        break
                if find_it:
                    break
                print(f"Cannot find score_ali file, please check {model_out_dir}")
                sys.exit()
            time_tag = name.split('_')[-1]
            
            for scale in np.flip(Pred_Resolutions): # [64, 32, 16, 8, 4, 2]
                # Iterate all score branches listed in Pred_Resolutions
                if args.keyword:
                    sco_ali_file = os.path.join(model_out_dir, '{}_score_ali_{}_{}.pkl'.format(args.output_name, args.keyword.split("_")[1], time_tag))
                else:
                    sco_ali_file = os.path.join(model_out_dir, '{}_score_ali_{}_{}.pkl'.format(args.output_name, scale, time_tag))
                sco_lab_pd = load_scolab(sco_ali_file)
                # Segment eer vector. Evaluate this score branch in all possible reso.
                # eers' dim: [1, len(Measure_Resolutions)]
                eers, thresholds, utteer_pd = get_eermatrix_scale(sco_lab_pd, scale)
                # utterance eer from segments score
                eer_utt, threshold_utt = get_utteer_by_seg(sco_lab_pd)
                # append eer (segment eer in x scale_num; utterance eer)
                eer_scales.append(np.concatenate((eers, [eer_utt])))
                threshold_scales.append(np.concatenate((thresholds, [threshold_utt])))
                utteer_scales[f'{scale}'] = utteer_pd # utter_pd's eers == eers[idx] for idx==index?
                if args.keyword:
                    break

        np.savez(eer_file,
                 eers_scales=eer_scales,
                 threshold_scales=threshold_scales,
                 utteer_scales=utteer_scales)

        print('='*10, args.output_name, args.save_name, "="*10)
        print("eer:\n")
        for x in eer_scales:
            print(x*100, "\n")

if __name__ == "__main__":
    main()