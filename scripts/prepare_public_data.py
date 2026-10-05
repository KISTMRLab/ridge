"""Convert licensed BVH takes and word-aligned transcripts to RIDGE contracts.

Motion is kept as neck-centred 3D (XYZ) joint positions at 15 FPS; no 2D
projection is involved. Each take becomes one transcript record with a unique
``record_id`` and a ``speaker`` field, so rules and models can be built per
speaker. Text embeddings use the Sentence-BERT given by ``--sbert`` (a model
name that sentence-transformers downloads on first use, or a local directory).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


JOINTS = ("Hips", "Neck", "Head", "LeftShoulder", "LeftArm", "LeftForeArm",
          "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand")


def _rotation(axis: str, degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    if axis == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def load_bvh(path: Path, target_fps: int = 15, joints: tuple[str, ...] = JOINTS) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    nodes: list[dict] = []
    stack: list[int | None] = []
    pending: int | None = None
    motion_line = next((i for i, line in enumerate(lines) if line.strip() == "MOTION"), None)
    if motion_line is None:
        raise ValueError("BVH is missing MOTION")
    for line in lines[:motion_line]:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] in ("ROOT", "JOINT"):
            pending = len(nodes)
            nodes.append({"name": parts[1], "parent": stack[-1] if stack else None,
                          "offset": np.zeros(3), "channels": []})
        elif parts[0] == "End":
            pending = None
        elif parts[0] == "{":
            stack.append(pending)
        elif parts[0] == "}":
            stack.pop()
        elif parts[0] == "OFFSET" and stack and stack[-1] is not None:
            nodes[stack[-1]]["offset"] = np.array([float(x) for x in parts[1:4]])
        elif parts[0] == "CHANNELS" and stack and stack[-1] is not None:
            nodes[stack[-1]]["channels"] = parts[2:]
    names = [node["name"] for node in nodes]
    missing = [name for name in joints if name not in names]
    if missing:
        raise ValueError(f"BVH missing required joints: {missing}")
    frame_time = float(lines[motion_line + 2].split(":")[-1])
    if frame_time <= 0:
        raise ValueError("BVH Frame Time must be positive")
    values = np.array([[float(v) for v in line.split()] for line in lines[motion_line + 3:] if line.strip()], dtype=np.float64)
    expected = sum(len(node["channels"]) for node in nodes)
    if values.ndim != 2 or values.shape[1] != expected:
        raise ValueError(f"BVH frames must have {expected} channel values")
    output = np.empty((len(values), len(joints), 3), dtype=np.float32)
    lookup = {name: i for i, name in enumerate(joints)}
    for frame, row in enumerate(values):
        transforms = []
        cursor = 0
        for node in nodes:
            local = np.array(node["offset"], dtype=np.float64)
            rotation = np.eye(3)
            for channel in node["channels"]:
                value = row[cursor]
                cursor += 1
                if channel.endswith("position"):
                    local["XYZ".index(channel[0])] += value
                elif channel.endswith("rotation"):
                    rotation = rotation @ _rotation(channel[0], value)
                else:
                    raise ValueError(f"unsupported BVH channel {channel}")
            parent = node["parent"]
            position = local if parent is None else transforms[parent][0] + transforms[parent][1] @ local
            world_rotation = rotation if parent is None else transforms[parent][1] @ rotation
            transforms.append((position, world_rotation))
            if node["name"] in lookup:
                output[frame, lookup[node["name"]]] = position
    duration = (len(output) - 1) * frame_time
    timestamps = np.arange(0, duration + 1e-8, 1 / target_fps)
    source = np.arange(len(output)) * frame_time
    resampled = np.empty((len(timestamps), len(joints), 3), np.float32)
    for j in range(len(joints)):
        for axis in range(3):
            resampled[:, j, axis] = np.interp(timestamps, source, output[:, j, axis])
    return resampled - resampled[:, 1:2]


def timed_words(path: Path, fps: int, frames: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    words = rows[0]["words"] if len(rows) == 1 and "words" in rows[0] else rows
    result = []
    for item in words:
        start = item.get("start_frame", round(float(item["start_seconds"]) * fps) if "start_seconds" in item else None)
        end = item.get("end_frame", round(float(item["end_seconds"]) * fps) if "end_seconds" in item else None)
        if start is None or end is None or not 0 <= int(start) < int(end) <= frames:
            raise ValueError("word timestamps must be ordered and inside the resampled motion")
        result.append({"word": str(item["word"]), "start_frame": int(start), "end_frame": int(end)})
    if not result or any(a["end_frame"] > b["start_frame"] for a, b in zip(result, result[1:])):
        raise ValueError("word timestamps must be non-overlapping and chronological")
    return result


def take_words(path: Path, fps: int, frames: int) -> list[dict]:
    """Timed words from JSONL (seconds or frames) or a BEAT TextGrid."""
    if path.suffix.lower() == ".textgrid":
        from ridge_gesture.annotate import frame_words, read_textgrid
        return frame_words(read_textgrid(path), fps, frames)
    return timed_words(path, fps, frames)


def take_list(args) -> list[dict]:
    if args.manifest:
        items = json.loads(args.manifest.read_text(encoding="utf-8"))
        base = args.manifest.parent
        return [{"bvh": base / item["bvh"] if not Path(item["bvh"]).is_absolute() else Path(item["bvh"]),
                 "transcript": base / item["transcript"] if not Path(item["transcript"]).is_absolute() else Path(item["transcript"]),
                 "record_id": item.get("record_id"), "speaker": item.get("speaker")} for item in items]
    if not args.bvh or len(args.bvh) != len(args.transcript or []):
        raise ValueError("give matching --bvh and --transcript lists, or --manifest")
    return [{"bvh": b, "transcript": t, "record_id": None, "speaker": args.speaker} for b, t in zip(args.bvh, args.transcript)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bvh", type=Path, nargs="+")
    p.add_argument("--transcript", type=Path, nargs="+", help="JSONL words or BEAT TextGrid, one per --bvh")
    p.add_argument("--manifest", type=Path, help="JSON list of {bvh, transcript, speaker?, record_id?}")
    p.add_argument("--speaker", help="speaker for every --bvh take (default: parsed from BEAT names)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--unit-seconds", type=float, default=3)
    p.add_argument("--sbert", default="all-MiniLM-L6-v2",
                   help="Sentence-BERT name (downloaded by sentence-transformers on first use) or a local directory")
    args = p.parse_args()
    if args.fps <= 0 or args.unit_seconds <= 0:
        raise ValueError("fps and unit-seconds must be positive")
    from ridge_gesture.annotate import beat_speaker
    length = round(args.fps * args.unit_seconds)
    records, clips, texts, ids, speakers, motions, skipped = [], [], [], [], [], {}, 0
    for take in take_list(args):
        record_id = take["record_id"] or take["bvh"].stem
        speaker = take["speaker"] if take["speaker"] is not None else beat_speaker(record_id)
        if record_id in motions:
            raise ValueError(f"duplicate record_id {record_id!r}")
        motion = load_bvh(take["bvh"], args.fps)
        words = take_words(take["transcript"], args.fps, len(motion))
        flat = motion.reshape(len(motion), -1)
        motions[record_id] = flat
        record = {"record_id": record_id, "speaker": speaker, "text": " ".join(w["word"] for w in words),
                  "words": words, "fps": args.fps}
        if take["transcript"].suffix.lower() == ".textgrid":
            record["textgrid"] = str(take["transcript"].resolve())
        records.append(record)
        for start in range(0, len(motion) - length + 1, length):
            text = " ".join(w["word"] for w in words if w["end_frame"] > start and w["start_frame"] < start + length)
            if not text:
                skipped += 1  # silent window: no speech to pair with
                continue
            clips.append(flat[start:start + length]); texts.append(text)
            ids.append(f"{record_id}:{start}-{start + length}"); speakers.append(speaker)
    if len(clips) < 2:
        raise ValueError("RIDGE training needs at least two motion windows with overlapping transcript words")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from sentence_transformers import SentenceTransformer
    embeddings = SentenceTransformer(args.sbert).encode(texts, normalize_embeddings=True)
    np.savez(args.output_dir / "train_pairs.npz", text_embeddings=embeddings.astype(np.float32), motion=np.stack(clips),
             ids=np.asarray(ids), texts=np.asarray(texts), speakers=np.asarray(speakers),
             lengths=np.full(len(clips), length), sbert=np.asarray(args.sbert))
    np.savez(args.output_dir / "motion_records.npz", **motions)
    (args.output_dir / "transcripts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    (args.output_dir / "joint_order.json").write_text(json.dumps({"joints": JOINTS, "fps": args.fps, "coordinates": "XYZ; neck centered", "sbert": args.sbert}, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(records), "speakers": sorted(set(speakers)), "pairs": len(clips),
                      "silent_windows_skipped": skipped, "sbert": args.sbert}))


if __name__ == "__main__":
    main()