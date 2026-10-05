# Requirements derived from the paper

## Paper facts

- RIDGE first searches an explicit rule base; only below a configured semantic-similarity threshold does it use learned text-motion retrieval.
- Rule phrases are salient, gesture-aligned spans of 3–10 words, proposed by an LLM (and compared with human annotation), aligned back to word timestamps, and stored with the observed motion.
- Rule and query phrases use `all-MiniLM-L6-v2` Sentence-BERT and cosine similarity.
- The fallback uses a Sentence-BERT text branch plus feed-forward projection and a GestureCLR-derived motion branch, aligned in a joint 10D space through in-batch contrastive learning. Motion is neck-normalized. Failed rule segments are typically rechunked to 5–6 words.
- GCA fits semantic text clusters and gesture subclusters inside them, then scores gesture affinity to the appropriate reference structure. Paper tuning uses speaker-independent partitions.

## Reimplementation decisions

- `ridge-gesture annotate` sends the paper's extraction prompt verbatim over each TextGrid to an OpenAI-compatible endpoint or a local LLM command, followed by a one-line JSON output instruction.
  - Proposals are validated as contiguous 3–10-word spans, and repeated phrases bind to successive occurrences. Provenance and rejections are recorded.
  - Reviewed JSON and a content-word heuristic remain as alternatives. No hosted key is embedded.
  - The small browser demo still resolves three cached, provenance-marked LLM annotations.
- Datasets carry unique `record_id`s and a `speaker` field. Rules and models can be built per speaker.
- Sentence-BERT is frozen by default and feeds a trainable feed-forward projection; `--finetune-text` fine-tunes it end to end. The checkpoint stores the Sentence-BERT id or path and whether it was frozen (`text_encoder`). `retrieve` reuses that encoder.
- The motion encoder has the GestureCLR architecture and can be initialised from a GestureCLR checkpoint (`--gesture-init`). It is an independent implementation, not unpublished institute code.
- Two-stage training:
  - presets: pretrain batch 1000, fine-tune batch 64;
  - a validation split, early stopping and an LR schedule;
  - `--init` to continue from stage 1.
  Epoch caps and learning rates are implementation choices. The paper's 2D-to-3D video mapping corpus is not reproduced.
- Hybrid retrieval scores spans at every start and accepts the best-scoring non-overlapping spans above the threshold. Fallback chunks of at most six words stop at the next rule.
- GCA L2-normalises embeddings before Bisecting K-Means so the clustering matches cosine scoring. It always fits on a reference/training file and scores a separate candidate/held-out file. Evaluation samples never alter centroids.

## Acceptance criteria

Commands cover LLM, reviewed or heuristic annotation; rule construction; two-stage contrastive training; hybrid retrieval; and leakage-resistant GCA. Outputs identify whether each result came from `rule` or `fallback`. No direct latent-to-motion decoder is included because the paper describes retrieval and leaves decoding to future work.

## Interactive data handoff

`scripts/start_demo.py` downloads one official BEAT BVH/TextGrid take, builds a local nine-clip bank and paired windows, resolves cached semantic spans to explicit strong rules, and locally fits a compact text-motion fallback in ignored outputs. The small demo substitutes TF-IDF features for Sentence-BERT before contrastive training; the full CLI above retains its documented encoder contract. The trace distinguishes strong-rule from fitted fallback routes. Cached annotations and a tiny fitted model are not evidence of the paper's rule extraction or retrieval quality. The older `--example` path remains an explicitly authored offline fixture. Speech is optional; recordings and fitted weights are not bundled.

## Bundled fictional avatar substitution

Two newly generated fictional CC0 humanoids replace the original avatar assets in the browser demo. They provide a 53-bone rig and named ARKit/viseme targets. Motion retargeting adapts source joints to their bind pose; mouth shapes follow a rule-based text-to-phoneme-to-viseme track timed to speech playback, an approximation rather than forced phoneme alignment. The optional recorded BEAT companion inspects public motion, face and audio files prepared locally, independently of the paper's learned algorithm. No dataset recordings or trained weights are bundled.
