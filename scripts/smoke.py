"""Train the RIDGE core briefly and exercise rule and fallback retrieval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import BoW, Dense, Normalize

from ridge_gesture.model import TextMotionModel, contrastive
from ridge_gesture.pipeline import GCA, align_phrase, heuristic_phrases, hybrid_retrieve


def make_text_encoder(path: Path, texts: list[str]) -> None:
    torch.manual_seed(19)
    vocab = sorted({token for text in texts for token in text.lower().split()})
    SentenceTransformer(modules=[BoW(vocab), Dense(len(vocab), 384), Normalize()]).save_pretrained(str(path))


def run_cli(*args: object) -> None:
    subprocess.run([sys.executable, "-m", "ridge_gesture.cli", *map(str, args)], check=True)


def embed(texts: list[str], width: int = 384) -> np.ndarray:
    rows = []
    for text in texts:
        row = np.zeros(width, np.float32)
        for token in text.lower().split(): row[sum(token.encode("utf-8")) % width] += 1
        row /= max(np.linalg.norm(row), 1e-8); rows.append(row)
    return np.stack(rows)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=Path("outputs/smoke"))
    args = parser.parse_args(); out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(19); rng = np.random.default_rng(19)
    texts = ["open both hands now", "point to the result", "move to the next topic", "welcome everyone today", "explain this important idea", "finish the demonstration"]
    text_vectors = embed(texts); motion = rng.normal(size=(6, 16, 18)).astype("float32")
    ids = np.asarray(["record_0:0-20", *[f"gesture_{i}" for i in range(1, 6)]])
    np.savez(out / "train_pairs.npz", text_embeddings=text_vectors, motion=motion, ids=ids)
    model = TextMotionModel(384, 18); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train(); zt, zm = model(torch.from_numpy(text_vectors), torch.from_numpy(motion)); loss = contrastive(zt, zm)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad(): zt, zm = model(torch.from_numpy(text_vectors), torch.from_numpy(motion))
    torch.save({"state": model.state_dict(), "text_dim": 384, "motion_dim": 18, "ids": ids.tolist(), "motion_latents": zm}, out / "ridge.pt")
    checkpoint = torch.load(out / "ridge.pt", map_location="cpu", weights_only=True)
    model = TextMotionModel(checkpoint["text_dim"], checkpoint["motion_dim"]); model.load_state_dict(checkpoint["state"]); model.eval()
    motion_latents = checkpoint["motion_latents"].numpy(); ids = np.asarray(checkpoint["ids"])

    words = [{"word": w, "start_frame": i * 5, "end_frame": (i + 1) * 5} for i, w in enumerate(texts[0].split())]
    phrase = heuristic_phrases(texts[0], min_words=3, limit=1)[0]; start, end = align_phrase(phrase, words)
    rules = [{"phrase": phrase, "gesture_id": ids[0], "start_frame": start, "end_frame": end, "embedding": embed([phrase])[0].tolist()}]
    (out / "rules.jsonl").write_text(json.dumps(rules[0]) + "\n", encoding="utf-8")
    def encode_fallback(text: str) -> np.ndarray:
        with torch.no_grad(): return model.text(torch.from_numpy(embed([text]))).numpy()[0]
    sequence = hybrid_retrieve("open both hands now then discuss another topic", rules, embed, 0.99, motion_latents, ids.tolist(), encode_fallback)
    (out / "sequence.json").write_text(json.dumps(sequence, indent=2), encoding="utf-8")
    gca = GCA(text_clusters=2, gesture_clusters=1, seed=19).fit(zt.numpy(), motion_latents).score(zt.numpy(), motion_latents)
    transcript = {"record_id": "record_0", "text": texts[0], "words": words}
    (out / "transcripts.jsonl").write_text(json.dumps(transcript) + "\n", encoding="utf-8")
    (out / "reviewed.json").write_text(json.dumps({"record_0": [texts[0]]}), encoding="utf-8")
    make_text_encoder(out / "tiny-sbert", texts + ["open both hands now then discuss another topic"])
    run_cli("annotate", "--transcripts", out / "transcripts.jsonl", "--external-annotations", out / "reviewed.json", "--output", out / "cli-phrases.jsonl")
    run_cli("build-rules", "--records", out / "transcripts.jsonl", "--annotations", out / "cli-phrases.jsonl", "--sbert", out / "tiny-sbert", "--output", out / "cli-rules.jsonl")
    run_cli("train", "--pairs", out / "train_pairs.npz", "--output", out / "cli-ridge.pt", "--epochs", 1, "--batch-size", 6, "--seed", 19)
    run_cli("retrieve", "--rules", out / "cli-rules.jsonl", "--checkpoint", out / "cli-ridge.pt", "--sbert", out / "tiny-sbert", "--threshold", 0.99, "--text", "open both hands now then discuss another topic", "--output", out / "cli-sequence.json")
    np.savez(out / "gca-reference.npz", text_embeddings=zt.numpy(), motion_embeddings=motion_latents)
    np.savez(out / "gca-candidate.npz", text_embeddings=zt.numpy(), motion_embeddings=motion_latents)
    run_cli("eval-gca", "--reference", out / "gca-reference.npz", "--candidate", out / "gca-candidate.npz", "--text-clusters", 2, "--gesture-clusters", 1)
    cli_sequence = json.loads((out / "cli-sequence.json").read_text(encoding="utf-8"))
    if not cli_sequence: raise RuntimeError("installed CLI produced no gestures")
    print(json.dumps({"loss": float(loss.detach()), "retrieval_paths": sorted({x["source"] for x in sequence}), "gca": gca, "output": str(out)}))


if __name__ == "__main__": main()
