"""Prepare the RIDGE paper method on public BEAT for the browser demo.

Launcher hook (see scripts/beat_demo/BEAT_INGEST.md): progress goes to stderr
and the last stdout line is ``{"ready": ..., "server_args": [...]}``.

RIDGE is per speaker. One BEAT speaker is the target; the others pretrain:

* target speaker, training takes: timed transcripts -> gesture phrases
  (``ridge-gesture annotate``: content-word heuristic offline, a reviewed JSON
  file, or the paper's LLM prompt through ``--llm-endpoint``/``--llm-command``)
  -> Sentence-BERT strong rules over the speaker's own motion spans
  (``ridge-gesture build-rules``); the same takes give 3 s text-motion pairs
  for fine-tuning;
* other speakers: 3 s text-motion pairs for contrastive pretraining
  (``ridge-gesture train --preset pretrain``), then the target pairs fine-tune
  it (``--preset finetune --init``); ``--no-pretrain`` trains on the target only;
* target speaker, held-out take: transcripts never used for rules or training
  probe the learned fallback (held-out text -> motion top-1 against chance).

Results are cached under ``outputs/paper-method/<settings hash>/``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_method_common as pm  # noqa: E402

PACKAGE = "ridge_gesture"
DEFAULT_SPEAKERS = "1,2,3,4"
DEFAULT_TAKES = 3


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pm.add_source_args(p, DEFAULT_SPEAKERS, DEFAULT_TAKES, None)
    p.add_argument("--target-speaker", help="speaker whose rules and fallback are built (default: first selected speaker)")
    p.add_argument("--heldout-takes", type=int, default=1, help="target takes kept out of rules and training (default 1)")
    p.add_argument("--no-pretrain", action="store_true", help="train the target speaker only (skip stage-1 pretraining)")
    p.add_argument("--sbert", help=f"Sentence-BERT directory (default models/{pm.SBERT_NAME}; env {pm.ENV_SBERT})")
    p.add_argument("--preset", choices=("demo", "paper"), default="demo",
                   help="demo: at most 150 epochs per stage; paper: the CLI presets (300 epochs, early stopping)")
    p.add_argument("--epochs", type=int, help="epochs for each training stage (overrides the preset)")
    p.add_argument("--batch-size", type=int, help="fine-tuning batch size (default: CLI finetune preset, 64)")
    p.add_argument("--gesture-init", type=Path, help="GestureCLR checkpoint that initialises the motion encoder")
    p.add_argument("--threshold", type=float, default=0.72, help="strong-rule similarity threshold (paper 0.72)")
    p.add_argument("--external-annotations", type=Path, help="reviewed JSON {record_id: [phrases]}")
    p.add_argument("--llm-endpoint", help="OpenAI-compatible base URL for the paper's extraction prompt")
    p.add_argument("--model", help="model name for --llm-endpoint")
    p.add_argument("--llm-command", help="local LLM command (prompt on stdin, reply on stdout)")
    p.add_argument("--phrases-per-record", type=int, default=12, help="heuristic phrases per transcript (default 12)")
    return p.parse_args(argv)


def _settings(args, source, kind, descriptors, sbert):
    external = args.external_annotations
    return {"repo": PACKAGE, "source": str(source), "kind": kind, "selection": pm.selection(args),
            "takes": [d["take"] for d in descriptors], "target": args.target_speaker, "heldout": args.heldout_takes,
            "pretrain": not args.no_pretrain, "seed": args.seed, "max_frames": args.max_frames, "preset": args.preset,
            "epochs": args.epochs, "batch_size": args.batch_size, "sbert": sbert, "threshold": args.threshold,
            "gesture_init": str(args.gesture_init) if args.gesture_init else None,
            "annotations": external.read_text(encoding="utf-8") if external else None,
            "llm": [args.llm_endpoint, args.model, args.llm_command], "phrases": args.phrases_per_record}


def _speaker_key(value):
    return (len(value), value)


def write_pairs(path, windows, encoder, sbert):
    if not windows:
        return 0
    texts = [w["text"] for w in windows]
    np.savez(path, text_embeddings=encoder.encode(texts, normalize_embeddings=True).astype(np.float32),
             motion=np.stack([w["positions"].reshape(len(w["positions"]), -1) for w in windows]).astype(np.float32),
             ids=np.asarray([w["id"] for w in windows]), texts=np.asarray(texts),
             speakers=np.asarray([w["speaker"] for w in windows]),
             lengths=np.asarray([len(w["positions"]) for w in windows]), sbert=np.asarray(sbert))
    return len(windows)


def heldout_top1(checkpoint, pairs):
    """Held-out transcript -> its own motion window among all held-out windows (fallback model only)."""
    import torch
    from ridge_gesture.model import encode_text, load_checkpoint
    data = np.load(pairs)
    if len(data["ids"]) < 2:
        return {}
    model, _ = load_checkpoint(checkpoint)
    with torch.no_grad():
        zt = encode_text(model, torch.from_numpy(data["text_embeddings"].astype("float32"))).numpy()
        zm = model.encode_motion(torch.from_numpy(data["motion"].astype("float32")),
                                 torch.from_numpy(data["lengths"].astype("int64"))).numpy()
    top1 = float(((zt @ zm.T).argmax(1) == np.arange(len(zt))).mean())
    return {"heldout_top1": round(top1, 4), "heldout_chance": round(1 / len(zt), 4), "heldout_pairs": int(len(zt)),
            "heldout": "target speaker's held-out take: transcript window -> its own motion window (never trained on)"}


def prepare(args):
    source, kind = pm.find_source(args.processed, args.beat_root)
    if source is None:
        return pm.not_ready("no local BEAT source found", pm.SOURCE_STEPS)
    sbert, why = pm.find_sbert(args.sbert)
    if sbert is None:
        return pm.not_ready(why, [pm.SBERT_STEP])
    if args.llm_endpoint and not args.model:
        return pm.not_ready("--llm-endpoint needs --model")
    beat = pm.ingest()
    select = pm.selection(args)
    descriptors = beat.list_takes(source, kind=kind, **select)
    speakers = sorted({d["speaker"] for d in descriptors}, key=_speaker_key)
    if not descriptors:
        return pm.not_ready(f"no BEAT takes in {source} match the selection", pm.SOURCE_STEPS)
    target = args.target_speaker or speakers[0]
    target_takes = [d for d in descriptors if d["speaker"] == target]
    if not target_takes:
        return pm.not_ready(f"target speaker {target} has no selected takes")
    others = [d for d in descriptors if d["speaker"] != target]
    if not args.no_pretrain and not others:
        return pm.not_ready("pretraining needs at least one other speaker; add --speakers or use --no-pretrain")
    held = max(0, min(args.heldout_takes, len(target_takes) - 1))
    train_takes, held_takes = target_takes[:len(target_takes) - held], target_takes[len(target_takes) - held:]
    code = [pm.ROOT / "src" / PACKAGE, Path(__file__), Path(pm.__file__), pm.SCRIPTS / "beat_demo" / "beat_ingest.py"]
    folder = Path(args.output_root) / pm.cache_key(_settings(args, source, kind, descriptors, sbert), code)
    if not args.force and pm.cached(folder):
        pm.progress(f"cached result {pm.portable(folder)}")
        return pm.ready(folder, pm.cached(folder), cached_result=True)
    if folder.exists():
        shutil.rmtree(folder)
    data = folder / "data"; data.mkdir(parents=True)
    timings, started = {}, time.perf_counter()
    pm.use_repository_package(PACKAGE)
    from ridge_gesture import cli
    from sentence_transformers import SentenceTransformer

    pm.progress(f"loading {len(descriptors)} takes from {source} ({kind}); target speaker {target}")
    t0 = time.perf_counter()
    train_records = pm.load_takes(train_takes, args.max_frames)
    held_records = pm.load_takes(held_takes, args.max_frames)
    other_records = pm.load_takes(others, args.max_frames) if not args.no_pretrain else []
    train_windows = pm.take_windows(train_records)
    held_windows = pm.take_windows(held_records)
    if not held_records:  # one target take: the last quarter of its windows is held out
        cut = max(1, len(train_windows) // 4)
        train_windows, held_windows = train_windows[:-cut], train_windows[-cut:]
        last = min(int(w["id"].rsplit(":", 1)[1].split("-")[0]) for w in held_windows)
        for record in train_records:
            record["words"] = [w for w in record["words"] if w[2] <= last]
    transcripts = data / "transcripts.jsonl"
    with transcripts.open("w", encoding="utf-8") as stream:
        for record in train_records:
            row = {"record_id": record["take"], "speaker": record["speaker"], "fps": pm.FPS,
                   "text": " ".join(w for w, _, _ in record["words"]),
                   "words": [{"word": w, "start_frame": s, "end_frame": e} for w, s, e in record["words"]]}
            grid = record["source"].get("textgrid")
            if grid and kind == "raw" and held_records and not args.max_frames:
                row["textgrid"] = grid  # the paper prompt reads the original TextGrid
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    np.savez(data / "motion_records.npz", **{r["take"]: r["positions"].reshape(len(r["positions"]), -1) for r in train_records})
    encoder = SentenceTransformer(sbert)
    counts = {"finetune_pairs": write_pairs(data / "train_pairs.npz", train_windows, encoder, sbert),
              "heldout_pairs": write_pairs(data / "heldout_pairs.npz", held_windows, encoder, sbert),
              "pretrain_pairs": write_pairs(data / "pretrain_pairs.npz", pm.take_windows(other_records), encoder, sbert)}
    timings["data_seconds"] = round(time.perf_counter() - t0, 2)

    annotations = folder / "annotations.jsonl"
    annotate = ["annotate", "--transcripts", transcripts, "--output", annotations, "--limit", args.phrases_per_record]
    if args.external_annotations:
        annotate += ["--external-annotations", args.external_annotations]
    if args.llm_endpoint:
        annotate += ["--llm-endpoint", args.llm_endpoint, "--model", args.model]
    elif args.llm_command:
        annotate += ["--llm-command", args.llm_command]
    _, timings["annotate_seconds"] = pm.run_cli(cli.main, annotate, "gesture phrase annotation")
    rows = [json.loads(x) for x in annotations.read_text(encoding="utf-8").splitlines() if x.strip()]
    rules_path = folder / "rules.jsonl"
    _, timings["build_rules_seconds"] = pm.run_cli(cli.main, ["build-rules", "--records", transcripts, "--annotations", annotations,
                                                              "--output", rules_path, "--sbert", sbert], "strong rules")
    rules = [json.loads(x) for x in rules_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rules:
        raise ValueError("annotation produced no gesture phrases that align with the transcript")

    epochs = args.epochs or (150 if args.preset == "demo" else None)
    stage = []
    init = None
    if not args.no_pretrain:
        init = folder / "pretrain.pt"
        cmd = ["train", "--pairs", data / "pretrain_pairs.npz", "--preset", "pretrain", "--output", init,
               "--seed", args.seed, "--history", folder / "pretrain.history.json", "--log-every", 25]
        cmd += ["--epochs", epochs] if epochs else []
        cmd += ["--gesture-init", args.gesture_init] if args.gesture_init else []
        result, seconds = pm.run_cli(cli.main, cmd, "stage 1 pretraining (other speakers)")
        timings["pretrain_seconds"] = seconds; stage.append({"stage": "pretrain", **(result or {})})
    checkpoint = folder / "ridge.pt"
    cmd = ["train", "--pairs", data / "train_pairs.npz", "--preset", "finetune", "--output", checkpoint, "--seed", args.seed,
           "--history", folder / "finetune.history.json", "--log-every", 25]
    cmd += ["--epochs", epochs] if epochs else []
    cmd += ["--batch-size", args.batch_size] if args.batch_size else []
    cmd += ["--init", init] if init else (["--gesture-init", args.gesture_init] if args.gesture_init else [])
    result, timings["finetune_seconds"] = pm.run_cli(cli.main, cmd, "stage 2 fine-tuning (target speaker)" if init
                                                     else "per-speaker training (target speaker)")
    stage.append({"stage": "finetune" if init else "train", **(result or {})})
    heldout = heldout_top1(checkpoint, data / "heldout_pairs.npz") if counts["heldout_pairs"] else {}

    probes = []
    for w in held_windows:
        words = w["text"].split()
        if len(words) >= 6:
            probes.append(" ".join(words[:8]))
        if len(probes) == 2:
            break
    phrases = sorted({r["phrase"] for r in rules}, key=lambda p: (not 4 <= len(p.split()) <= 8, -len(p.split()), p))
    suggested = phrases[:4]
    if suggested and probes:
        suggested.append(f"{suggested[0]}. {probes[0]}")
    timings["total_seconds"] = round(time.perf_counter() - started, 2)
    annotator = sorted({r["annotator"] for r in rows})
    metrics = {"target_speaker": target, "strong_rules": len(rules), "annotator": ", ".join(annotator),
               "rejected_phrases": sum(len(r["provenance"]["rejected"]) for r in rows),
               "rule_threshold": args.threshold, **counts,
               "training": "pretrain on other speakers + fine-tune on target" if init else "target speaker only",
               **heldout}
    roles = {"unit": "speaker+take", "seed": args.seed,
             "roles": {"target": [target], "pretrain": sorted({d["speaker"] for d in others}, key=_speaker_key) if init else [],
                       "heldout": [target]},
             "takes": {"target": [d["take"] for d in train_takes], "pretrain": [d["take"] for d in others] if init else [],
                       "heldout": [d["take"] for d in held_takes] or ["last quarter of the target take"]}}
    manifest = {
        "schema": "paperreach.paper-method.v1", "repo": "ridge", "mode": "ridge",
        "created": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "source": {"path": str(source), "kind": kind, "takes": len(descriptors)},
        "roles": roles, "preset": args.preset, "sbert": sbert, "seed": args.seed, "threshold": args.threshold,
        "fps": pm.FPS, "units": "cm (neck-centred)",
        "files": {"rules": "rules.jsonl", "checkpoint": "ridge.pt", "annotations": "annotations.jsonl",
                  "pairs": pm.portable(data / "train_pairs.npz"), "records": pm.portable(data / "motion_records.npz"),
                  "transcripts": pm.portable(transcripts)},
        "stages": stage, "metrics": metrics, "timings": timings, "suggested_queries": suggested, "heldout_probes": probes,
        "summary": {"strong_rules": len(rules), "finetune_pairs": counts["finetune_pairs"],
                    "heldout_top1": heldout.get("heldout_top1"), "heldout_chance": heldout.get("heldout_chance"),
                    "data": f"BEAT {kind}: target speaker {target}" + (f", pretrain speakers {', '.join(roles['roles']['pretrain'])}" if init else "")
                            + f" ({args.preset} preset)",
                    "seconds": timings["total_seconds"]},
    }
    pm.write_manifest(folder, manifest)
    pm.progress(f"done in {timings['total_seconds']} s: {len(rules)} strong rules for speaker {target}; "
                f"held-out top-1 {heldout.get('heldout_top1')} (chance {heldout.get('heldout_chance')})")
    return pm.ready(folder, manifest)


def main(argv=None):
    args = parse(argv)
    try:
        return prepare(args)
    except (Exception, SystemExit) as error:  # the launcher falls back to the default demo
        if isinstance(error, SystemExit) and error.code in (0, None):
            raise
        pm.progress(f"failed: {error.__class__.__name__}: {error}")
        return pm.emit({"ready": False, "reason": f"{error.__class__.__name__}: {error}"}, 1)


if __name__ == "__main__":
    raise SystemExit(main())
