import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from IPython.display import Audio
from text_helper import normalize_text

def get_and_save_duration(wav_list, tag, utt_class):
    """
    Records and saves the duration of each utterance into a .txt file.
    """
    total_dur = np.array([])
    with open(f"/workspace/lps_dataset/durations/{tag}/{utt_class}.txt", "a") as f:
        for wav in wav_list:
            uttid = wav.split("/")[-1].split(".")[0]
            dur = round(sf.info(wav).duration, 2)
            total_dur = np.append(total_dur, dur)
            f.write(f"{uttid} {dur:.2f}\n")
        f.close()
    return total_dur

def get_durations(tag, utt_class):
    """
    Return the durations of each utterance in utt_class.
    """
    duration = {}
    with open(f"/workspace/lps_dataset/durations/{tag}/{utt_class}.txt", "r") as f:
        for line in f.readlines():
            uttid, dur = line.strip().split(" ")
            duration[uttid] = float(dur)
    return duration

def get_statistics(durations):
    """
    Returns the statistics (in seconds): min, max, mean, std, total duration, and length.
    """
    return [np.min(durations), np.max(durations), np.mean(durations), np.std(durations), np.sum(durations), len(durations)]

def plot_statistics(durations, bins=30):
    """
    Plot the bar graph distribution of durations.
    """
    mini, maxi, mean, std, total_dur, length = get_statistics(durations)

    stats_text = (
        f"Min: {mini:.2f}s\n"
        f"Max: {maxi:.2f}s\n"
        f"Mean: {mean:.2f}s\n"
        f"Std: {std:.2f}s\n"
        f"Total: {total_dur/3600:.2f}h"
    )

    plt.title(f"length: {length}")

    plt.hist(durations, bins=bins)
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             ha="right", va="top", bbox=dict(facecolor="white"))
    plt.show()

def play_audio(path):
    wav, sr = sf.read(path)
    return Audio(wav, rate=sr)

def get_transcripts(_type, _set) -> dict:
    transcripts = {}
    with open(f"/workspace/lps/transcripts/{_type}/{_set}.txt", "r") as f:
        for line in f.readlines():
            uttid, text = line.strip().split(" ", 1)
            transcripts[uttid] = normalize_text(text)
    return transcripts