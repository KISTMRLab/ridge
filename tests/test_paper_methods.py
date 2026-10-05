import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ridge_gesture import cli
from ridge_gesture.annotate import (EXTRACTION_PROMPT, PROMPT_SHA256, build_prompt, parse_phrases, read_textgrid,
                                    record_from_textgrid, render_textgrid)
from ridge_gesture.model import GestureEncoder, RidgeModel, load_gesture_init
from ridge_gesture.pipeline import GCA, align_phrase, hybrid_retrieve, validate_phrases

TRANSCRIPT = "okay so we open both hands now and then we open both hands again to point at the chart"


def timed(text, step=3):
    return [{"word": w, "start_frame": i * step, "end_frame": (i + 1) * step} for i, w in enumerate(text.split())]


def bow(texts, vocab=None):
    vocab = vocab or sorted(set(TRANSCRIPT.split()) | {"hello", "there", "please", "the", "big", "idea", "is",
                                                         "simple", "i", "think", "well"})
    rows = np.zeros((len(texts), len(vocab)), np.float32)
    for r, t in enumerate(texts):
        for w in t.lower().split():
            if w in vocab:
                rows[r, vocab.index(w)] += 1
    return rows / np.linalg.norm(rows, axis=1, keepdims=True).clip(1e-8)


def tiny_sbert(path, texts):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import BoW, Dense, Normalize
    torch.manual_seed(0)
    vocab = sorted({w for t in texts for w in t.lower().split()})
    SentenceTransformer(modules=[BoW(vocab), Dense(len(vocab), 16), Normalize()]).save_pretrained(str(path))
    return str(path)


def test_prompt_is_the_paper_text_and_textgrid_round_trip(tmp_path):
    for sentence in ("Task: Extract Key Gesture-Aligned Phrases.",
                     "Read the given content of the TextGrid file, format it into a clean and coherent paragraph.",
                     "Phrases should have a minimum length of 3 words and a maximum of 10 words.",
                     "avoiding over-extraction."):
        assert sentence in EXTRACTION_PROMPT
    words = timed(TRANSCRIPT)
    grid = tmp_path / "2_scott_0_1_1.TextGrid"; grid.write_text(render_textgrid(words), encoding="utf-8")
    assert [w["word"] for w in read_textgrid(grid)] == TRANSCRIPT.split()
    record = record_from_textgrid(grid, 15)
    assert record["speaker"] == "2_scott" and record["words"][3] == {"word": "open", "start_frame": 9, "end_frame": 12}
    prompt = build_prompt(record)
    assert prompt.startswith(EXTRACTION_PROMPT) and 'text = "chart"' in prompt


def test_validation_binds_repeated_phrases_to_occurrences():
    words = timed(TRANSCRIPT)
    accepted, rejected = validate_phrases(
        ["open both hands", "Open both hands!", "open both hands", "point at", "hands to point", "chart now please"],
        words)
    assert [(a["occurrence"], a["word_start"]) for a in accepted] == [(0, 3), (1, 10)]
    reasons = [r["reason"] for r in rejected]
    assert "proposed more often than it occurs" in reasons[0] and "expected 3-10" in reasons[1]
    assert "not a contiguous span" in reasons[3]
    assert align_phrase("open both hands", words, occurrence=1) == (30, 39)
    with pytest.raises(ValueError):
        align_phrase("open both hands", words, occurrence=2)


def test_parse_phrases_accepts_fenced_json_and_lists():
    assert parse_phrases('```json\n{"paragraph": "x", "phrases": ["a b c"]}\n```') == ["a b c"]
    assert parse_phrases('Here: [{"phrase": "a b c"}, "d e f"]') == ["a b c", "d e f"]
    with pytest.raises(ValueError):
        parse_phrases("no json here")


def test_hybrid_scans_every_start_and_prefers_best_span():
    rules = [{"phrase": "point at the chart", "gesture_id": "point", "embedding": bow(["point at the chart"])[0].tolist()},
             {"phrase": "the big idea is simple", "gesture_id": "idea", "embedding": bow(["the big idea is simple"])[0].tolist()}]
    lat = np.eye(2, dtype=np.float32)
    out = hybrid_retrieve("hello there point at the chart please", rules, bow, .9, lat, ["f0", "f1"], lambda _: lat[0])
    assert [(o["source"], o["text"]) for o in out] == [("fallback", "hello there"), ("rule", "point at the chart"),
                                                       ("fallback", "please")]
    out = hybrid_retrieve("okay well so the big idea is simple i think", rules, bow, .8, lat, ["f0", "f1"], lambda _: lat[0])
    rule = [o for o in out if o["source"] == "rule"][0]
    assert rule["text"] == "the big idea is simple" and rule["similarity"] == pytest.approx(1.0)
    assert [o["text"] for o in out if o["source"] == "fallback"] == ["okay well so", "i think"]
    long = hybrid_retrieve(" ".join(["hello"] * 7), rules, bow, .9, lat, ["f0", "f1"], lambda _: lat[0])
    assert [len(o["text"].split()) for o in long] == [4, 3]


def test_gca_clusters_on_the_unit_sphere():
    rng = np.random.default_rng(0)
    text = rng.normal(size=(20, 4)); motion = text + rng.normal(scale=.1, size=(20, 4))
    a = GCA(3, 2).fit(text, motion).score(text, motion)
    scale = rng.uniform(.1, 10, size=(20, 1))
    b = GCA(3, 2).fit(text * scale, motion * scale[::-1]).score(text * scale, motion * scale[::-1])
    assert a == pytest.approx(b)


class _LLM(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _LLM.requests.append((self.path, body))
        reply = '```json\n{"paragraph": "...", "phrases": ["open both hands", "open both hands", "two words"]}\n```'
        data = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_annotate_cli_llm_endpoint_command_and_rules(tmp_path):
    grid = tmp_path / "3_solomon_0_1_1.TextGrid"; grid.write_text(render_textgrid(timed(TRANSCRIPT)), encoding="utf-8")
    server = HTTPServer(("127.0.0.1", 0), _LLM)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        cli.main(["annotate", "--textgrid", str(grid), "--records-output", str(tmp_path / "records.jsonl"),
                  "--llm-endpoint", f"http://127.0.0.1:{server.server_port}/v1", "--model", "stub",
                  "--api-key-env", "", "--output", str(tmp_path / "llm.jsonl")])
    finally:
        server.shutdown()
    path, body = _LLM.requests[-1]
    assert path == "/v1/chat/completions" and body["model"] == "stub"
    assert body["messages"][0]["content"].startswith(EXTRACTION_PROMPT)
    row = json.loads((tmp_path / "llm.jsonl").read_text(encoding="utf-8"))
    assert row["annotator"] == "llm" and row["speaker"] == "3_solomon"
    assert [p["occurrence"] for p in row["phrases"]] == [0, 1]
    assert row["provenance"]["prompt_sha256"] == PROMPT_SHA256 and len(row["provenance"]["rejected"]) == 1
    stub = tmp_path / "llm_stub.py"
    stub.write_text("import sys,json\nprompt=sys.stdin.read()\nassert 'TextGrid content' in prompt\n"
                    "print(json.dumps({'phrases': ['point at the chart']}))\n", encoding="utf-8")
    cli.main(["annotate", "--transcripts", str(tmp_path / "records.jsonl"), "--llm-command",
              f'"{sys.executable}" "{stub}"', "--output", str(tmp_path / "cmd.jsonl")])
    row = json.loads((tmp_path / "cmd.jsonl").read_text(encoding="utf-8"))
    assert row["phrases"][0]["phrase"] == "point at the chart" and row["provenance"]["command"]
    sbert = tiny_sbert(tmp_path / "sbert", [TRANSCRIPT])
    cli.main(["build-rules", "--records", str(tmp_path / "records.jsonl"), "--annotations", str(tmp_path / "llm.jsonl"),
              "--sbert", sbert, "--per-speaker", "--output", str(tmp_path / "rules")])
    rules = [json.loads(x) for x in (tmp_path / "rules" / "3_solomon.rules.jsonl").read_text().splitlines()]
    assert [r["gesture_id"] for r in rules] == ["3_solomon_0_1_1:9-18", "3_solomon_0_1_1:30-39"]
    assert rules[0]["sbert"] == sbert


def test_gesture_init_loads_gestureclr_motion_branch(tmp_path):
    torch.manual_seed(1)
    donor = GestureEncoder(6)
    torch.save({"state": {f"motion3d.{k}": v for k, v in donor.state_dict().items()}, "d2": 4, "d3": 6},
               tmp_path / "gestureclr.pt")
    model = load_gesture_init(RidgeModel(8, 6), tmp_path / "gestureclr.pt")
    assert torch.equal(model.motion.proj.weight, donor.proj.weight)
    with pytest.raises(ValueError):
        load_gesture_init(RidgeModel(8, 7), tmp_path / "gestureclr.pt")


def _pairs(path, n=16, speakers=("a", "b"), texts=True, seed=0):
    rng = np.random.default_rng(seed)
    words = ["open both hands", "point at the chart", "move on now", "hello there"]
    t = [words[i % 4] for i in range(n)]
    arrays = {"text_embeddings": bow(t), "motion": rng.normal(size=(n, 12, 6)).astype("float32"),
              "ids": np.asarray([f"{path.stem}:{i}" for i in range(n)]),
              "speakers": np.asarray([speakers[i % len(speakers)] for i in range(n)]), "sbert": np.asarray("bow")}
    if texts:
        arrays["texts"] = np.asarray(t)
    np.savez(path, **arrays)
    return t


def test_two_stage_training_early_stopping_and_per_speaker(tmp_path):
    _pairs(tmp_path / "wild.npz", 24); _pairs(tmp_path / "beat.npz", 16, seed=1)
    torch.manual_seed(1)
    donor = GestureEncoder(6)
    torch.save({"state": {f"motion3d.{k}": v for k, v in donor.state_dict().items()}, "d3": 6}, tmp_path / "g.pt")
    cli.main(["train", "--pairs", str(tmp_path / "wild.npz"), "--preset", "pretrain", "--epochs", "3",
              "--gesture-init", str(tmp_path / "g.pt"), "--output", str(tmp_path / "stage1.pt"), "--log-every", "0"])
    stage1 = torch.load(tmp_path / "stage1.pt", weights_only=True)
    assert stage1["config"]["batch_size"] == 1000 and stage1["gesture_init"] and stage1["sbert"] == "bow"
    cli.main(["train", "--pairs", str(tmp_path / "beat.npz"), "--init", str(tmp_path / "stage1.pt"), "--epochs", "20",
              "--min-delta", "100", "--patience", "2", "--val-fraction", ".25", "--history", str(tmp_path / "h.json"),
              "--output", str(tmp_path / "stage2.pt"), "--log-every", "0"])
    stage2 = torch.load(tmp_path / "stage2.pt", weights_only=True)
    history = json.loads((tmp_path / "h.json").read_text())["history"]
    assert stage2["config"]["batch_size"] == 64 and stage2["init"] and stage2["val_pairs"] == 4
    assert len(history) == 3 and stage2["text_encoder"] == "frozen"
    cli.main(["train", "--pairs", str(tmp_path / "beat.npz"), "--per-speaker", "--epochs", "2",
              "--output", str(tmp_path / "speakers"), "--log-every", "0"])
    a = torch.load(tmp_path / "speakers" / "a.pt", weights_only=True)
    assert set(a["speakers"]) == {"a"} and len(a["ids"]) == 8
    assert (tmp_path / "speakers" / "b.pt").exists()


def test_finetune_text_and_retrieve(tmp_path):
    t = _pairs(tmp_path / "pairs.npz", 8, speakers=("a",))
    sbert = tiny_sbert(tmp_path / "sbert", t)
    cli.main(["train", "--pairs", str(tmp_path / "pairs.npz"), "--finetune-text", "--sbert", sbert, "--epochs", "2",
              "--output", str(tmp_path / "ft.pt"), "--log-every", "0"])
    ck = torch.load(tmp_path / "ft.pt", weights_only=True)
    assert ck["text_encoder"] == "finetuned" and Path(ck["text_encoder_path"]).is_dir()
    rules = [{"phrase": "open both hands", "gesture_id": "pairs:0", "embedding": [1.0] + [0.0] * 15, "sbert": sbert}]
    (tmp_path / "rules.jsonl").write_text(json.dumps(rules[0]) + "\n")
    cli.main(["retrieve", "--rules", str(tmp_path / "rules.jsonl"), "--checkpoint", str(tmp_path / "ft.pt"),
              "--threshold", "1.01", "--text", "point at the chart now", "--output", str(tmp_path / "seq.json")])
    seq = json.loads((tmp_path / "seq.json").read_text())
    assert [s["source"] for s in seq] == ["fallback"] and seq[0]["gesture_id"].startswith("pairs:")


def _bvh(path, frames):
    joints = ["Neck", "Head", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "RightShoulder", "RightArm",
              "RightForeArm", "RightHand"]
    lines = ["HIERARCHY", "ROOT Hips", "{", "OFFSET 0 0 0", "CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation"]
    for j in joints:
        lines += [f"JOINT {j}", "{", "OFFSET 0 1 0", "CHANNELS 0"]
    lines += ["}"] * (len(joints) + 1)
    lines += ["MOTION", f"Frames: {frames}", "Frame Time: 0.0666666667"]
    lines += [f"0 0 0 {i % 30} 0 0" for i in range(frames)]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_prepare_public_data_multi_take_with_textgrid_and_local_sbert(tmp_path):
    words = [{"word": w, "start_frame": 5 + 4 * i, "end_frame": 8 + 4 * i} for i, w in enumerate(TRANSCRIPT.split())]
    for take in ("1_wayne_0_1_1", "2_scott_0_1_1"):
        _bvh(tmp_path / f"{take}.bvh", 150)
        (tmp_path / f"{take}.TextGrid").write_text(render_textgrid(words), encoding="utf-8")
    sbert = tiny_sbert(tmp_path / "sbert", [TRANSCRIPT])
    subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_public_data.py"),
                    "--bvh", str(tmp_path / "1_wayne_0_1_1.bvh"), str(tmp_path / "2_scott_0_1_1.bvh"),
                    "--transcript", str(tmp_path / "1_wayne_0_1_1.TextGrid"), str(tmp_path / "2_scott_0_1_1.TextGrid"),
                    "--sbert", sbert, "--output-dir", str(tmp_path / "prep")], check=True)
    pairs = np.load(tmp_path / "prep" / "train_pairs.npz")
    assert set(pairs["speakers"].tolist()) == {"1_wayne", "2_scott"} and str(pairs["sbert"]) == sbert
    assert len(set(pairs["ids"].tolist())) == len(pairs["ids"]) == 4  # the silent last window is skipped
    records = [json.loads(x) for x in (tmp_path / "prep" / "transcripts.jsonl").read_text().splitlines()]
    assert [r["record_id"] for r in records] == ["1_wayne_0_1_1", "2_scott_0_1_1"] and records[0]["textgrid"]
    assert set(np.load(tmp_path / "prep" / "motion_records.npz").files) == {"1_wayne_0_1_1", "2_scott_0_1_1"}
