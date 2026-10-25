#!/usr/bin/env python3
"""
Generate TTS audio files from a filename list.

Usage:
    python tts_generate.py --model <model_name> --input <input_text.txt> --outdir <output_dir>

Text input format (whitespace-separated, first token = ID, rest = text):
    1272_128104_000001_000000 A "PERPLEXING" ART CRITIC
    1272_128104_000003_000000 There's a lively and rejuvenating bank holiday vibe...

Supported models:
    f5tts           - F5-TTS
    maskgct         - MaskGCT         
    sparktts        - Spark-TTS-0.5B  
    ctboxturbo      - Chatterbox-Turbo
    higgsv3         - Higgs Audio v3 TTS
    fishspeechs2pro - Fish Audio S2 Pro (4B-param Dual-AR TTS)
"""

import argparse
import re
import json
import time
import logging
import os
import sys
import unicodedata
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
from dotenv import load_dotenv

# Loads HF_TOKEN and the API URLs from the nearest .env (searched upward from this file)
load_dotenv()

TRAIN_TAG = "train01"
DEV_TAG = "dev01"
EVAL_TAG = "eval01"

TRAIN_REF = "/workspace/lps_dataset/wav_audio/train/train_bona"
DEV_REF = "/workspace/lps_dataset/wav_audio/dev/dev_bona"
EVAL_REF = "/workspace/lps_dataset/wav_audio/eval/eval_bona"

transcripts = {}

# -------------- Text Normalizer --------------
def normalize_text(text: str) -> str:
    """
    Normalize text for TTS input:
      - Unicode -> ASCII-compatible (NFKC)
      - Curly/smart quotes -> straight quotes, then removed
      - Hyphens and dashes -> space
      - All remaining symbols removed (keeps apostrophes, commas, periods,
        exclamation/question marks for natural prosody)
      - Collapse multiple spaces
      - Lowercase
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace('"', "")
    text = re.sub(r"[-\u2010-\u2015\u2212]", " ", text)
    text = re.sub(r"[^a-zA-Z0-9 \',\.!?]", "", text)
    text = re.sub(r" +", " ", text).strip()
    return text.lower()

# -------------- Helper Functions --------------
def save_wav(audio, sr: int, out_file: Path) -> float:
    """
    Save audio as WAV at its native sample rate.
    Returns the duration in seconds.
    """
    sf.write(str(out_file), audio, sr, format="WAV")
    return len(audio) / sr

def find_longest_ref_audio(tag: str, speaker_id: str, audio_dir: Path) -> [Path,str]:
    """
    Return the WAV file in audio_dir that belongs to speaker_id and has
    the longest duration. Results are cached on the function itself so the
    directory scan and sf.info calls only happen once per speaker per process.

    Files are matched by the naming convention used throughout this script:
        {tag}_{speaker_id}_{...}.wav

    Raises FileNotFoundError if not matching file is found.
    """
    cache: dict[str, tuple[Path, str]] = find_longest_ref_audio.__dict__.setdefault("_cache", {})
    if speaker_id in cache:
        return cache[speaker_id]

    candidates = sorted(audio_dir.glob(f"{tag}_{speaker_id}_*.wav"))
    if not candidates:
        raise FileNotFoundError(
            f"No reference audio found for speaker '{speaker_id}' in {audio_dir}"
        )

    best_path: Path | None = None
    longest_dur: float = 0.0

    for wav_path in candidates:
        try:
            info = sf.info(str(wav_path))
            if info.duration > longest_dur and info.duration <= 12:
                longest_dur = info.duration
                best_path = wav_path
        except Exception:
            # Corrupted or unreadable file: skip silently
            continue

    if best_path is None:
        raise FileNotFoundError(
            f"Could not read any reference audio for speaker '{speaker_id}' in {audio_dir}"
        )
    utt_id = best_path.stem.removeprefix(f"{tag}_")
    duration_s = sf.info(str(best_path)).duration
    print(
        f"[TTS] speaker '{speaker_id}' -> longest ref: "
        f"{best_path.name} ({duration_s:.2f}s, {len(candidates)} candidate(s))"
    )
    result = (best_path, utt_id)
    cache[speaker_id] = result
    return result

def load_transcript(transcript_file: Path) -> dict[str, str]:
    """Load id->text mapping from a transcript filelist."""
    transcripts = {}
    with transcript_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                transcripts[parts[0]] = normalize_text(parts[1])
    return transcripts

def get_ref_audio(audio_dir: Path, speaker_id: str) -> list[Path, str, str]:
    set_ = audio_dir.name
    tag = "train-clean" if set_ == "train_bona" else "dev-clean" if set_ == "dev_bona" else "eval-clean"
    ref_path, ref_id = find_longest_ref_audio(tag, speaker_id, audio_dir)
    ref_text = transcripts.get(ref_id)

    return ref_path, ref_id, ref_text

# -------------- Logging Helper --------------
def make_sample_logger(full_id: str, log_dir: Path) -> tuple[logging.Logger, logging.FileHandler]:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"tts.{full_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers on retry
    handler = logging.FileHandler(log_dir / f"{full_id}.log", mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(handler)
    return logger, handler

def write_manifest(manifest_path: Path, record: dict) -> None:
    with open(manifest_path, "a") as mf:
        mf.write(json.dumps(record) + "\n")

# -------------- Model Loaders --------------
def load_f5tts() -> dict:
    from f5_tts.infer.utils_infer import load_model, load_vocoder
    from hydra.utils import get_class
    from omegaconf import OmegaConf

    f5tts_sr = 24000

    model_cfg = OmegaConf.load(
        "/usr/local/lib/python3.12/dist-packages/f5_tts/configs/F5TTS_v1_Base.yaml"
    )
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch

    vocoder = load_vocoder()
    model = load_model(
        model_cls,
        model_arc,
        "/workspace/tts_models/f5tts/F5TTS_v1_Base/model_1250000.safetensors",
    )
    return {"model": model, "vocoder": vocoder, "rate": f5tts_sr}

def load_maskgct() -> dict:
    VENDOR = "/workspace/tts_models/vendor"
    AMPHION_ROOT = "/workspace/tts_models/Amphion"

    for p in (VENDOR, AMPHION_ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)
    from models.tts.maskgct.maskgct_utils import (
        load_config,
        build_semantic_model,
        build_semantic_codec,
        build_acoustic_codec,
        build_t2s_model,
        build_s2a_model,
        MaskGCT_Inference_Pipeline
    )
    import safetensors.torch as st
    from huggingface_hub import hf_hub_download
    
    maskgct_sr = 24000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_path = f"{AMPHION_ROOT}/models/tts/maskgct/config/maskgct.json"
    cfg = load_config(cfg_path)

    semantic_model, semantic_mean, semantic_std = build_semantic_model(device)
    semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)
    codec_encoder, codec_decoder = build_acoustic_codec(cfg.model.acoustic_codec, device)
    t2s_model = build_t2s_model(cfg.model.t2s_model, device)
    s2a_model_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, device)
    s2a_model_full = build_s2a_model(cfg.model.s2a_model.s2a_full, device)

    # Download and load all six checkpoints
    st.load_model(semantic_codec, hf_hub_download("amphion/MaskGCT", "semantic_codec/model.safetensors"))
    st.load_model(codec_encoder, hf_hub_download("amphion/MaskGCT", "acoustic_codec/model.safetensors"))
    st.load_model(codec_decoder, hf_hub_download("amphion/MaskGCT", "acoustic_codec/model_1.safetensors"))
    st.load_model(t2s_model, hf_hub_download("amphion/MaskGCT", "t2s_model/model.safetensors"))
    st.load_model(s2a_model_1layer, hf_hub_download("amphion/MaskGCT", "s2a_model/s2a_model_1layer/model.safetensors"))
    st.load_model(s2a_model_full, hf_hub_download("amphion/MaskGCT", "s2a_model/s2a_model_full/model.safetensors"))

    pipeline = MaskGCT_Inference_Pipeline(
        semantic_model, semantic_codec,
        codec_encoder, codec_decoder,
        t2s_model, s2a_model_1layer, s2a_model_full,
        semantic_mean, semantic_std, device
    )

    return {"pipeline": pipeline, "rate": maskgct_sr}

def load_sparktts() -> dict:
    SPARK_DIR = "/workspace/tts_models/Spark-TTS"
    MODEL_DIR = "/workspace/tts_models/Spark-TTS-0.5B"

    if not os.path.exists(MODEL_DIR):
        from huggingface_hub import snapshot_download
        snapshot_download("SparkAudio/Spark-TTS-0.5B",
                          local_dir=MODEL_DIR)
    if SPARK_DIR not in sys.path:
        sys.path.insert(0, SPARK_DIR)

    from cli.SparkTTS import SparkTTS
    sparktts_sr = 16000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparkTTS(MODEL_DIR, device)
    return {"model": model, "rate": sparktts_sr}

def load_ctboxturbo() -> dict:
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxTurboTTS.from_pretrained(device=device)
    return {"model": model, "rate": model.sr}

def load_higgsv3() -> dict:
    higgsv3_rate = 24000
    return {"api": os.environ["HIGGSV3_URL"], "rate": higgsv3_rate}

def load_fishspeechs2pro() -> dict:
    fishspeechs2pro_rate = 44100
    return {"api": os.environ["FISHAUDIO_S2PRO_URL"], "rate": fishspeechs2pro_rate}

MODEL_LOADERS: dict[str, callable] = {
    "f5tts": load_f5tts,
    "maskgct": load_maskgct,
    "sparktts": load_sparktts,
    "ctboxturbo": load_ctboxturbo,
    "higgsv3": load_higgsv3,
    "fishspeechs2pro": load_fishspeechs2pro,
}

# -------------- Model Handlers --------------
def run_f5tts(full_id: str, text: str, out_dir: Path, resources: dict, 
              log_dir: Path, manifest_path: Path) -> None:
    """
    Audio generation using F5-TTS.
    Output sampling rate is 24000 Hz.

    Requires: pip install f5-tts, python>=3.11, torch 2.11, torchaudio 2.11, torchcodec 0.11, ffmpeg
    Docs:     https://github.com/SWivid/F5-TTS
    """
    from f5_tts.infer.utils_infer import infer_process

    model = resources["model"]
    vocoder = resources["vocoder"]
    
    out_filename = f"{TRAIN_TAG}-f5tts-full_{full_id}.wav"
    out_file = out_dir/out_filename
    text = normalize_text(text)
    
    audio_dir = Path(TRAIN_REF)
    ref_path, ref_id, ref_text = get_ref_audio(audio_dir, full_id.split("_")[0])

    # Logging
    logger, handler = make_sample_logger(full_id, log_dir)
    logger.info(f"full_id:  {full_id}")
    logger.info(f"out_file: {out_file}")
    logger.info(f"ref_path: {ref_path}")
    logger.info(f"ref_id:   {ref_id}")
    logger.info(f"ref_text: {ref_text}")
    logger.info(f"gen_text: {text}")

    t0 = time.time()
    status, error_msg, duration_s = "ok", None, None

    try:
        audio, sr, _ = infer_process(
            ref_path, ref_text, text, model, vocoder, speed=0.8
        )
        duration_s = save_wav(audio, sr, out_file)
        logger.info(f"saved: {out_file} ({duration_s:.2f}s @ {sr}Hz)")
    except Exception as e:
        status = "error"
        error_msg = str(e)
        logger.error(f"FAILED: {error_msg}")
        raise
    finally:
        elapsed = time.time() - t0
        logger.info(f"elapsed: {elapsed:.2f}s")
        logger.removeHandler(handler)
        handler.close()

        write_manifest(manifest_path, {
            "full_id": full_id,
            "ref_id": ref_id,
            "status": status,
            "error": error_msg,
            "duration_s": duration_s,
            "elapsed_s": round(elapsed, 2),
            "out_file": str(out_file)
        })

def run_maskgct(full_id: str, text: str, out_dir: Path, resources: dict,
                log_dir: Path, manifest_path: Path) -> None:
    """
    Audio generation using MaskGCT.
    Output sampling rate is 24000 Hz.

    Requires:
        git clone https://github.com/open-mmlab/Amphion
        cd /workspace/Amphion (to Amphion directory)
        pip install -r models/tts/maskgct/requirements.txt

    Model checkpoints are downloaded automatically from HuggingFace
    (amphion/MaskGCT) on first run and cached via huggingface_hub.

    Run inside the Amphion directory if facing any issue on importing modules.
    """
    pipeline = resources["pipeline"]
    sr = resources["rate"]

    out_filename = f"{DEV_TAG}-maskgct-full_{full_id}.wav"
    out_file = out_dir/out_filename
    text = normalize_text(text)

    audio_dir = Path(DEV_REF)
    ref_path, ref_id, ref_text = get_ref_audio(audio_dir, full_id.split("_")[0])

    # Logging
    logger, handler = make_sample_logger(full_id, log_dir)
    logger.info(f"full_id:  {full_id}")
    logger.info(f"out_file: {out_file}")
    logger.info(f"ref_path: {ref_path}")
    logger.info(f"ref_id:   {ref_id}")
    logger.info(f"ref_text: {ref_text}")
    logger.info(f"gen_text: {text}")

    t0 = time.time()
    status, error_msg, duration_s = "ok", None, None

    try:
        audio = pipeline.maskgct_inference(
            ref_path, ref_text, text, language="en", target_language="en"
        )
        duration_s = save_wav(audio, sr, out_file)
        logger.info(f"saved: {out_file} ({duration_s:.2f} s @ {sr} Hz)")
    except Exception as e:
        status = "error"
        error_msg = str(e)
        logger.error(f"FAILED: {error_msg}")
        raise
    finally:
        elapsed = time.time() - t0
        logger.info(f"elapsed: {elapsed:.2f}s")
        logger.removeHandler(handler)
        handler.close()

        write_manifest(manifest_path, {
            "full_id": full_id,
            "ref_id": ref_id,
            "status": status,
            "error": error_msg,
            "duration_s": duration_s,
            "elapsed_s": round(elapsed, 2),
            "out_file": str(out_file)
        })

def run_sparktts(full_id:str, text: str, out_dir: Path, resources: dict,
                 log_dir: Path, manifest_path: Path) -> None:
    """
    Audio generation using Spark-TTS-0.5B.
    Output sampling rate is 16000 Hz.

    Requires:
        git clone https://github.com/SparkAudio/Spark-TTS /workspace/Spark-TTS
        cd /workspace/Spark-TTS && pip install -r requirements.txt

    Model weights must be downloaded separately.
    
    Run inside the Spark-TTS directory if facing any issue on importing modules.
    """
    model = resources["model"]
    sr = resources["rate"]

    out_filename = f"{DEV_TAG}-sparktts-full_{full_id}.wav"
    out_file = out_dir/out_filename
    text = normalize_text(text)

    audio_dir = Path(DEV_REF)
    ref_path, ref_id, ref_text = get_ref_audio(audio_dir, full_id.split("_")[0])
    
    # Logging
    logger, handler = make_sample_logger(full_id, log_dir)
    logger.info(f"full_id:  {full_id}")
    logger.info(f"out_file: {out_file}")
    logger.info(f"ref_path: {ref_path}")
    logger.info(f"ref_id:   {ref_id}")
    logger.info(f"ref_text: {ref_text}")
    logger.info(f"gen_text: {text}")

    t0 = time.time()
    status, error_msg, duration_s = "ok", None, None

    try:
        with torch.no_grad():
            audio = model.inference(
                text=text,
                prompt_speech_path=str(ref_path),
                prompt_text = None # Avoiding error inconsistency due to LLM's randomness       
            )
        duration_s = save_wav(audio, sr, out_file)
        logger.info(f"saved: {out_file} ({duration_s:.2f} s @ {sr} Hz)")
    except Exception as e:
        status = "error"
        error_msg = str(e)
        logger.error(f"FAILED: {error_msg}")
        raise
    finally:
        elapsed = time.time() - t0
        logger.info(f"elapsed: {elapsed:.2f}s")
        logger.removeHandler(handler)
        handler.close()

        write_manifest(manifest_path, {
            "full_id": full_id,
            "ref_id": ref_id,
            "status": status,
            "error": error_msg,
            "duration_s": duration_s,
            "elapsed_s": round(elapsed, 2),
            "out_file": str(out_file)
        })

def run_ctboxturbo(full_id: str, text:str, out_dir: Path, resources:dict,
                   log_dir: Path, manifest_path: Path) -> None:
    """
    Audio generation using Chatterbox-Turbo.
    Output sampling rate is 24000 Hz.

    Requires:
        pip install chatterbox-tts
    Docs: https://github.com/resemble-ai/chatterbox

    It also supports paralinguistic tags such as [laugh], [chuckle],
    [cough], [sigh] directly in the text.
    """
    model = resources["model"]
    sr = resources["rate"]

    out_filename = f"{EVAL_TAG}-ctboxturbo-full_{full_id}.wav"
    out_file = out_dir/out_filename
    text = normalize_text(text)

    audio_dir = Path(EVAL_REF)
    ref_path, ref_id, ref_text = get_ref_audio(audio_dir, full_id.split("_")[0])
    
    # Logging
    logger, handler = make_sample_logger(full_id, log_dir)
    logger.info(f"full_id:  {full_id}")
    logger.info(f"out_file: {out_file}")
    logger.info(f"ref_path: {ref_path}")
    logger.info(f"ref_id:   {ref_id}")
    logger.info(f"ref_text: {ref_text}")
    logger.info(f"gen_text: {text}")

    t0 = time.time()
    status, error_msg, duration_s = "ok", None, None

    try:
        with torch.no_grad():
            audio = model.generate(
                text,
                audio_prompt_path=str(ref_path)
            )
            audio = audio.detach().cpu().numpy().squeeze(0)
        duration_s = save_wav(audio, sr, out_file)
        logger.info(f"saved: {out_file} ({duration_s:.2f} s @ {sr} Hz)")
    except Exception as e:
        status = "error"
        error_msg = str(e)
        logger.error(f"FAILED: {error_msg}")
        raise
    finally:
        elapsed = time.time() - t0
        logger.info(f"elapsed: {elapsed:.2f}s")
        logger.removeHandler(handler)
        handler.close()

        write_manifest(manifest_path, {
            "full_id": full_id,
            "ref_id": ref_id,
            "status": status,
            "error": error_msg,
            "duration_s": duration_s,
            "elapsed_s": round(elapsed, 2),
            "out_file": str(out_file)
        })

def run_higgsv3(full_id: str, text: str, out_dir: Path, resources: dict,
                log_dir: Path, manifest_path: Path) -> None:
    """
    Audio generation using Higgs Audio v3 TTS.

    Requires:
        pip install transformers accelerate soundfile torch
    Docs:
        https://github.com/boson-ai/higgs-audio
        https://huggingface.co/bosonai/higgs-tts-3-4b
    Model: bosonai/higgs-audio-v3-tts-4b
    """
    import requests
    api = resources["api"]
    sr = resources["rate"]

    out_filename = f"{EVAL_TAG}-higgsv3-full_{full_id}.wav"
    out_file = out_dir/out_filename
    text = normalize_text(text)

    audio_dir = Path(EVAL_REF)
    ref_path, ref_id, ref_text = get_ref_audio(audio_dir, full_id.split("_")[0])

    # Logging
    logger, handler = make_sample_logger(full_id, log_dir)
    logger.info(f"full_id:  {full_id}")
    logger.info(f"out_file: {out_file}")
    logger.info(f"ref_path: {ref_path}")
    logger.info(f"ref_id:   {ref_id}")
    logger.info(f"ref_text: {ref_text}")
    logger.info(f"gen_text: {text}")

    t0 = time.time()
    status, error_msg, duration_s = "ok", None, None

    try:
        resp = requests.post(
            api,
            json={
                "input": text,
                "references": [{
                    "audio_path": str(ref_path),
                    "text": ref_text
                }],
                "temperature": 0.8,
            },
        )
        if resp.status_code != 200 or not resp.headers.get("content-type", "").startswith("audio/"):
            raise RuntimeError(
                f"HTTP {resp.status_code} ct={resp.headers.get('content-type')} "
                f"body={resp.content[:500]!r}"
            )
        with open(out_file, "wb") as f:
            f.write(resp.content)
        wav, samplerate = sf.read(out_file)
        duration_s = len(wav)/samplerate
        logger.info(f"saved: {out_file} ({duration_s:.2f} s @ {samplerate} Hz)")
    except Exception as e:
        status = "error"
        error_msg = str(e)
        logger.error(f"FAILED: {error_msg}")
        raise
    finally:
        elapsed = time.time() - t0
        logger.info(f"elapsed: {elapsed:.2f}s")
        logger.removeHandler(handler)
        handler.close()

        write_manifest(manifest_path, {
            "full_id": full_id,
            "ref_id": ref_id,
            "status": status,
            "error": error_msg,
            "duration_s": duration_s,
            "elapsed_s": round(elapsed, 2),
            "out_file": str(out_file)
        })

def run_fishaudios2pro(full_id: str, text: str, out_dir: Path, resources: dict,
                        log_dir: Path, manifest_path: Path) -> None:
    """
    Audio generation using Fish Audio S2 Pro (4B-param Dual-AR TTS).
 
    This handler talks to a *self-hosted* Fish Speech API server
    (tools/api_server.py from https://github.com/fishaudio/fish-speech),
    normally started via the accompanying docker-compose.yml:
 
        docker compose --profile server up -d
 
    The server is started with the S2-Pro checkpoint loaded, so every
    request to its `/v1/tts` endpoint is served by S2-Pro — there's no
    separate per-request "model" field. Docs: https://speech.fish.audio/server/
 
    The server URL is read from the FISHAUDIO_S2PRO_URL env var so the
    same script works whether it's reached over the docker-compose network
    (e.g. http://fish-speech-server:8080/v1/tts, the default in
    docker-compose.yml) or on localhost when run outside docker.
    """
    import base64
    import requests
 
    api_url = resources["api"]
    sr = resources["rate"]
 
    out_filename = f"{DEV_TAG}-maskgct-full_{full_id}.wav"
    out_file = out_dir/out_filename
    text = normalize_text(text)

    audio_dir = Path(DEV_REF)
    ref_path, ref_id, ref_text = get_ref_audio(audio_dir, full_id.split("_")[0])
 
    # Logging
    logger, handler = make_sample_logger(full_id, log_dir)
    logger.info(f"full_id:  {full_id}")
    logger.info(f"out_file: {out_file}")
    logger.info(f"ref_path: {ref_path}")
    logger.info(f"ref_id:   {ref_id}")
    logger.info(f"ref_text: {ref_text}")
    logger.info(f"gen_text: {text}")

    t0 = time.time()
    status, error_msg, duration_s = "ok", None, None
 
    ref_b64 = base64.b64encode(ref_path.read_bytes()).decode("ascii")
 
    payload = {
        "text": text,
        "reference_audio": ref_b64,
        "reference_text": normalize_text(ref_text) if ref_text else "",
        "format": "wav",
    }
 
    try:
        resp = requests.post(api_url, json=payload, timeout=300)
        resp.raise_for_status()
        with open(out_file, "wb") as f:
            f.write(resp.content)
        wav, samplerate = sf.read(out_file)
        duration_s = len(wav)/samplerate
        logger.info(f"saved: {out_file} ({duration_s:.2f} s @ {samplerate} Hz)")
    except Exception as e:
        status = "error"
        error_msg = str(e)
        logger.error(f"FAILED: {error_msg}")
        raise
    finally:
        elapsed = time.time() - t0
        logger.info(f"elapsed: {elapsed:.2f}s")
        logger.removeHandler(handler)
        handler.close()

        write_manifest(manifest_path, {
            "full_id": full_id,
            "ref_id": ref_id,
            "status": status,
            "error": error_msg,
            "duration_s": duration_s,
            "elapsed_s": round(elapsed, 2),
            "out_file": str(out_file)
        })

# -------------- Model Registry --------------
# Map CLI model name(s) to handler function.
MODEL_REGISTRY: dict[str, callable] = {
    "f5tts": run_f5tts,
    "maskgct": run_maskgct,
    "sparktts": run_sparktts,
    "ctboxturbo": run_ctboxturbo,
    "higgsv3": run_higgsv3,
    "fishspeechs2pro": run_fishaudios2pro,
}

# -------------- Dispatcher --------------
def dispatch(model: str, full_id: str, text: str, out_dir: Path,
             resources: dict, log_dir: Path, manifest_path: Path) -> None:
    handler = MODEL_REGISTRY.get(model.lower())
    if handler is None:
        supported = ", ".join(sorted(set(MODEL_REGISTRY.keys())))
        raise ValueError(f"Unknown model `{model}`. Supported: {supported}")
    handler(full_id, text, out_dir, resources, log_dir, manifest_path)

# -------------- File Parser --------------
def parse_filelist(input_file: Path) -> list[tuple[str, str]]:
    """
    Parse lines of the form:
        <id> <text ...>
    Returns a list of (id, text) tuples, skipping blank lines and comments.
    """
    entries = []
    with input_file.open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1) # split on first whitespace run
            if len(parts) != 2:
                print(f"WARNING: line {lineno} malformed, skipping: {raw.rstrip()}")
                continue
            entries.append((parts[0], parts[1]))
    return entries

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True,
                    help="TTS model to use", default="f5tts")
    p.add_argument("--input", required=True, type=Path,
                   help="Path to the filelist text file")
    p.add_argument("--outdir", default="./tts_outputs", type=Path,
                   help="Directory for output WAV files (default: ./tts_outputs)")
    p.add_argument("--transcripts", type=Path, default=None,
                   help="Optimal transcript file for reference audio (id -> text)")
    return p

# -------------- Main --------------
def main() -> None:
    args = build_parser().parse_args()

    if not args.input.is_file():
        sys.exit(f"ERROR: Input file not found: {args.input}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    log_dir = args.outdir / "logs"
    manifest_path = args.outdir/"generation_manifest.jsonl"
    
    entries = parse_filelist(args.input)

    print("="*55)
    print(f"TTS Generator")
    print(f"Model: {args.model}")
    print(f"Input: {args.input} ({len(entries)} entries)")
    print("="*55)

    global transcripts
    transcripts = load_transcript(args.transcripts) if args.transcripts else {}

    loader = MODEL_LOADERS.get(args.model.lower())
    if loader is None:
        sys.exit(f"ERROR: No resource loader for model '{args.model}.")
    print(f"[TTS] Loading model resources for '{args.model}'...")
    resources = loader()

    success = failed = 0
    for full_id, text in entries:
        try:
            dispatch(args.model, full_id, text, args.outdir, resources, log_dir, manifest_path)
            success += 1
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            failed += 1
            break
        except Exception as exc:
            print(f"ERROR [{full_id}]: {exc}")
            failed += 1

    print("="*55)
    print(f"Done. Total={len(entries)}, Success={success}, Failed={failed}")
    print("="*55)

    if failed:
        sys.exit(1)

if __name__ == "__main__":
    main()