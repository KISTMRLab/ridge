# Requirements derived from the paper

## Paper facts

- RIDGE first searches an explicit rule base; only below a configured semantic-similarity threshold does it use learned text-motion retrieval.
- Rule phrases are salient, gesture-aligned spans of 3–10 words, proposed by an LLM (and compared with human annotation), aligned back to word timestamps, and stored with the observed motion.
- Rule and query phrases use `all-MiniLM-L6-v2` Sentence-BERT and cosine similarity.
- The fallback uses a Sentence-BERT text branch plus feed-forward projection and a GestureCLR-derived motion branch, aligned in a joint 10D space through in-batch contrastive learning. Motion is neck-normalized. Failed rule segments are typically rechunked to 5–6 words.
- GCA fits semantic text clusters and gesture subclusters inside them, then scores gesture affinity to the appropriate reference structure. Paper tuning uses speaker-independent partitions.

## Reimplementation decisions

- Phrase annotation defaults to a deterministic content-word heuristic and accepts reviewed or external-LLM JSON. No hosted LLM call or key is embedded.
- Sentence-BERT embeddings are frozen during compact training; its 384D outputs feed a trainable MLP. This differs from full end-to-end fine-tuning and is explicit in checkpoint metadata.
- The motion encoder is a compact temporal Transformer with sinusoidal frame positions rather than unpublished institute code.
- GCA always fits on a reference/training file and scores a separate candidate/held-out file. Evaluation samples never alter centroids.

## Acceptance criteria

Commands cover annotation, rule construction, contrastive training, hybrid retrieval, and leakage-resistant GCA. Outputs identify whether each result came from `rule` or `fallback`. No direct latent-to-motion decoder is included because the paper describes retrieval and leaves decoding to future work.

## Interactive data handoff

The browser queries the existing retrieval implementation and renders the selected motion frames with a pinned local Three.js module. Its immediate example mode is author-created motion plus explicitly illustrative, untrained vectors. Production mode accepts the documented public BVH/timed-transcript preparation outputs, real local encoder assets and trained checkpoints as appropriate. The preparation adapter preserves motion/transcript alignment and declares skeleton/FPS assumptions; it does not fabricate annotations or evaluation results. Speech is optional and replaceable (Kokoro-82M English/faster-whisper small CPU INT8, with browser voice/typed-input alternatives). Verification must cover clip serialization and algorithm routing, with model quality evaluation deferred to user-prepared public data.
