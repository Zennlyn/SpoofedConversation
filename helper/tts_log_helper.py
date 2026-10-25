import json

def get_records(tag: str, model: str):
    """
    Returns the generation records.
    """
    records = []
    with open(f"/workspace/lps_dataset/wav_audio/{tag}/{model}/generation_manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def find_record_byid(records: list, target_id: str):
    return next((r for r in records if r["full_id"] == target_id), None)

def get_error_records(records: list):
    return [rec for rec in records if rec["status"] == "error"]

def get_logs(tag: str, model: str, target_id: str):
    print(Path(f"/workspace/lps_dataset/wav_audio/{tag}/{model}/logs/{target_id}.log").read_text(encoding="utf-8"))