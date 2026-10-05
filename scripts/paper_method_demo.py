"""Prepared demo mode: serve RIDGE strong rules and the fitted fallback from public BEAT.

``scripts/demo_server.py --prepared outputs/paper-method/<key>`` loads the
artifacts written by ``prepare_paper_method.py`` and answers ``/api/beat-library``,
``/api/beat-query`` and ``/api/query`` with the repository's ``hybrid_retrieve``:
3-10-word spans above the rule threshold play the target speaker's annotated
motion; the remaining words go to the contrastive text-motion fallback.
"""
from __future__ import annotations

import json

import numpy as np

import paper_method_common as pm

ALGORITHM = ("RIDGE: per-speaker strong rules (annotated 3-10-word phrases aligned to the speaker's motion, "
             "Sentence-BERT threshold) plus a contrastive text-motion fallback pretrained on other speakers "
             "and fine-tuned on the target speaker")
DATA_LABEL = "Public BEAT, one target speaker; local demo-scale fit"


class PreparedDemo:
    def __init__(self, folder, args=None, encoder=None, fallback=None):
        pm.use_repository_package("ridge_gesture")
        from ridge_gesture.cli import fallback_encoder
        from ridge_gesture.model import load_checkpoint
        self.folder, self.manifest = pm.load_manifest(folder)
        files = self.manifest["files"]
        self.rules = [json.loads(x) for x in (self.folder / files["rules"]).read_text(encoding="utf-8").splitlines() if x.strip()]
        pairs = np.load(pm.resolve(files["pairs"]))
        self.motion = {str(i): m for i, m in zip(pairs["ids"], pairs["motion"])}
        self.texts = {str(i): str(t) for i, t in zip(pairs["ids"], pairs["texts"])}
        records = np.load(pm.resolve(files["records"]))
        self.rule_motion = {}
        for rule in self.rules:
            continuous = records[rule["record_id"]]
            start, end = int(rule["start_frame"]), int(rule["end_frame"])
            if not 0 <= start < end <= len(continuous):
                raise ValueError(f"rule {rule['gesture_id']} is outside the prepared motion")
            self.rule_motion[rule["gesture_id"]] = continuous[start:end]
        self.rest = np.median(np.stack([m.reshape(len(m), -1, 3)[0] for m in self.motion.values()]), axis=0)
        model, ck = load_checkpoint(self.folder / files["checkpoint"])
        self.latent = ck["motion_latents"].cpu().numpy().astype("float32")
        self.ids = [str(x) for x in ck["ids"]]
        sbert = getattr(args, "sbert", None)
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            text_model = SentenceTransformer(str(sbert or self.manifest["sbert"]))
            encoder = lambda texts: text_model.encode(list(texts), normalize_embeddings=True)
        self.encode = encoder
        self.fallback = fallback or fallback_encoder(model, ck, sbert)

    def library(self):
        source = lambda gid: {"speaker": self.manifest["metrics"]["target_speaker"], "take": gid.rsplit(":", 1)[0],
                              "window_id": gid, "kind": self.manifest["source"]["kind"]}
        clips = [{"id": r["gesture_id"], "text": r["phrase"], "duration": len(self.rule_motion[r["gesture_id"]]) / pm.FPS,
                  "source": {**source(r["gesture_id"]), "route": "strong_rule", "annotator": r.get("annotator")}}
                 for r in self.rules]
        clips += [{"id": gid, "text": self.texts[gid], "duration": len(self.motion[gid]) / pm.FPS,
                   "source": {**source(gid), "route": "fallback_library"}} for gid in self.ids]
        suggested = list(self.manifest["suggested_queries"]) + list(self.manifest["heldout_probes"])
        return {"ready": True, "prepared": True, "mode": "ridge", "clips": clips, "suggested_queries": suggested,
                "heldout_probes": self.manifest["heldout_probes"], "metrics": self.manifest["metrics"],
                "roles": self.manifest["roles"]["roles"], "strong_rules": [r["phrase"] for r in self.rules],
                "algorithm": ALGORITHM, "data_label": DATA_LABEL}

    def query(self, text, params):
        from ridge_gesture.pipeline import hybrid_retrieve
        threshold = pm.param(params, "threshold", pm.param(params, "strong_rule_threshold", self.manifest["threshold"], float), float)
        floor = pm.param(params, "min_similarity", None, float)
        sequence = hybrid_retrieve(text, self.rules, self.encode, threshold, self.latent, self.ids, self.fallback)
        if not sequence:
            raise ValueError("Query text has no words")
        slots = []
        for entry in sequence:
            gid = entry["gesture_id"]
            if entry["source"] == "rule":
                slots.append({"gesture_id": gid, "text": entry["text"], "frames": pm.frames_m(self.rule_motion[gid]),
                              "route": "strong_rule", "confidence": round(entry["similarity"], 5),
                              "similarity": round(entry["similarity"], 5),
                              "source": {"speaker": self.manifest["metrics"]["target_speaker"], "window_id": gid},
                              "rule_source": {"rule_phrase": entry.get("rule_phrase"), "threshold": threshold},
                              "blend_frames": 5})
            elif floor is not None and entry["similarity"] < floor:
                slots.append(pm.idle_slot(entry["text"], self.rest, "fallback below similarity floor", floor=floor,
                                          similarity=round(entry["similarity"], 5)))
            else:
                slots.append({"gesture_id": gid, "text": entry["text"], "frames": pm.frames_m(self.motion[gid]),
                              "route": "trained_text_motion_fallback", "confidence": round(entry["similarity"], 5),
                              "similarity": round(entry["similarity"], 5),
                              "source": {"speaker": self.manifest["metrics"]["target_speaker"], "window_id": gid,
                                         "window_text": self.texts.get(gid)},
                              "rule_source": {"kind": "contrastive_fallback"}, "blend_frames": 5})
        stored = self.manifest["metrics"]
        metrics = {k: stored.get(k) for k in ("strong_rules", "heldout_top1", "heldout_chance", "target_speaker")}
        metrics.update(rule_count=len(self.rules), threshold=threshold, min_similarity=floor)
        return pm.query_result(slots, algorithm=ALGORITHM, data_label=DATA_LABEL, metrics=metrics,
                               trace={"input": text, "retrieval_text": text},
                               joints=pm.ingest().UPPER_BODY,
                               extra={"threshold": threshold,
                                      "rule_count": sum(s["route"] == "strong_rule" for s in slots),
                                      "fallback_count": sum(s["route"] == "trained_text_motion_fallback" for s in slots)})
