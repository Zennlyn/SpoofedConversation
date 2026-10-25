import re
from difflib import SequenceMatcher

SEP = "=" * 72


def parse_file(filepath: str) -> dict[str, list[str]]:
    """Return {id: words} for each non-empty line; first duplicate ID wins."""
    entries = {}
    with open(filepath, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            parts = raw.split()
            if not parts:
                continue
            uid, words = parts[0], parts[1:]
            if uid in entries:
                print(f"Warning: duplicate ID '{uid}' on line {lineno} in {filepath}; keeping first.")
            else:
                entries[uid] = words
    return entries


def count_edits(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Return word-level (insertions, replacements, deletions) from ref to hyp."""
    ins = rep = dele = 0
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes():
        ref_span, hyp_span = i2 - i1, j2 - j1
        if tag == "replace":
            rep += min(ref_span, hyp_span)
            dele += max(ref_span - hyp_span, 0)
            ins += max(hyp_span - ref_span, 0)
        elif tag == "insert":
            ins += hyp_span
        elif tag == "delete":
            dele += ref_span
    return ins, rep, dele


def compare_modifications(clean_file: str, mod_file: str, output_file: str = "modification_info.txt") -> dict:
    """Write a per-ID edit report and return the totals."""
    clean = parse_file(clean_file)
    mod = parse_file(mod_file)
    totals = {"insertions": 0, "replacements": 0, "deletions": 0, "lines": 0}

    with open(output_file, "w", encoding="utf-8") as out:
        out.write("Modification report\n")
        out.write(f"  Clean file   : {clean_file}\n")
        out.write(f"  Modified file: {mod_file}\n")
        out.write(f"{SEP}\n\n")

        for uid in sorted(set(clean) | set(mod)):
            ref, hyp = clean.get(uid), mod.get(uid)
            out.write(f"[{uid}]\n")
            if ref is None:
                out.write("  !! ID only in modified file (entire line is an insertion)\n\n")
                continue
            if hyp is None:
                out.write("  !! ID only in clean file (entire line is a deletion)\n\n")
                continue

            ins, rep, dele = count_edits(ref, hyp)
            totals["insertions"] += ins
            totals["replacements"] += rep
            totals["deletions"] += dele
            totals["lines"] += 1

            out.write(f"  Clean   : {' '.join(ref)}\n")
            out.write(f"  Modified: {' '.join(hyp)}\n")
            out.write(f"  Insertions : {ins}\n")
            out.write(f"  Replacements: {rep}\n")
            out.write(f"  Deletions  : {dele}\n\n")

        out.write(f"{SEP}\n")
        out.write(f"TOTALS  (over {totals['lines']} matched lines)\n")
        out.write(f"  Total Insertions : {totals['insertions']}\n")
        out.write(f"  Total Replacements: {totals['replacements']}\n")
        out.write(f"  Total Deletions  : {totals['deletions']}\n")

    return totals


def parse_modification_info(filepath: str) -> list[dict]:
    """Parse a modification report into a list of entry dicts."""
    fields = {
        "Clean": ("clean", str),
        "Modified": ("modified", str),
        "Insertions": ("insertions", int),
        "Replacements": ("replacements", int),
        "Deletions": ("deletions", int),
    }
    entries, current = [], None

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            m = re.match(r"^\[(.+)\]$", line)
            if m:
                current = {"id": m.group(1)}
                entries.append(current)
                continue
            m = re.match(r"^\s+(Clean|Modified|Insertions|Replacements|Deletions)\s*:\s*(.*)$", line)
            if m and current is not None:
                key, cast = fields[m.group(1)]
                value = m.group(2)
                if cast is int:
                    if not value.strip().isdigit():
                        continue
                    value = int(value)
                current[key] = value
    return entries


def find_deletions_only(info_file: str = "modification_info.txt", output_file: str = "deletions_only.txt") -> list[dict]:
    """Write entries with only deletions (no insertions/replacements) and return them."""
    entries = parse_modification_info(info_file)
    found = [
        e for e in entries
        if e.get("insertions", -1) == 0
        and e.get("replacements", -1) == 0
        and e.get("deletions", 0) > 0
    ]

    with open(output_file, "w", encoding="utf-8") as out:
        out.write("Deletion-only entries (Insertions=0, Replacements=0, Deletions>0)\n")
        out.write(f"Source: {info_file}\n")
        out.write(f"{SEP}\n\n")

        if not found:
            out.write("No entries found with only deletions.\n")
        for e in found:
            out.write(f"[{e['id']}]\n")
            out.write(f"  Clean   : {e.get('clean', 'N/A')}\n")
            out.write(f"  Modified: {e.get('modified', 'N/A')}\n")
            out.write(f"  Insertions : {e.get('insertions', 0)}\n")
            out.write(f"  Replacements: {e.get('replacements', 0)}\n")
            out.write(f"  Deletions  : {e.get('deletions', 0)}\n\n")

        out.write(f"{SEP}\n")
        out.write(f"Total deletion-only entries: {len(found)}\n")

    return found


def parse_deletion_ids(filepath: str) -> set[str]:
    """Return all [ID] headers found in a deletions file."""
    with open(filepath, encoding="utf-8") as f:
        return {m.group(1) for line in f if (m := re.match(r"^\[(.+)\]$", line.rstrip()))}


def remove_deletions(
    deletions_file: str = "deletions_only.txt",
    input_file: str = "test01.txt",
    output_file: str = "test01_cleaned.txt",
) -> tuple[int, int]:
    """Drop lines whose ID is in deletions_file; return (removed, kept)."""
    ids = parse_deletion_ids(deletions_file)
    removed = kept = 0

    with open(input_file, encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            parts = line.split()
            if not parts:
                f_out.write(line)
            elif parts[0] in ids:
                removed += 1
            else:
                f_out.write(line)
                kept += 1

    return removed, kept
