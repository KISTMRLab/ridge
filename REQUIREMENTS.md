# Requirements derived from the paper

## Paper facts

- RIDGE first searches an explicit rule base; only below a configured semantic-similarity threshold does it use learned text-motion retrieval.
- Rule phrases are salient, gesture-aligned spans of 3–10 words, proposed by an LLM (and compared with human annotation), aligned back to word timestamps, and stored with the observed motion.
- Rule and query phrases use `all-MiniLM-L6-v2` Sentence-BERT and cosine similarity.
- The fallback uses a Sentence-BERT text branch plus feed-forward projection and a GestureCLR-derived motion branch, aligned in a joint 10D space through in-batch contrastive learning. Motion is neck-normalized. Failed rule segments are typically rechunked to 5–6 words.
- GCA fits semantic text clusters and gesture subclusters inside them, then scores gesture affinity to the appropriate reference structure. Paper tuning uses speaker-independent partitions.

## Reimplementation decisions

- The full CLI can use a deterministic content-word heuristic or supplied annotation JSON. The small browser demo resolves three cached, provenance-marked LLM annotations; a separate explicit endpoint command can regenerate strong rules. No hosted key is embedded.
- Sentence-BERT embeddings are frozen during compact training; its 384D outputs feed a trainable MLP. This differs from full end-to-end fine-tuning and is explicit in checkpoint metadata.
- The motion encoder is a compact temporal Transformer with sinusoidal frame positions rather than unpublished institute code.
- GCA always fits on a reference/training file and scores a separate candidate/held-out file. Evaluation samples never alter centroids.

## Acceptance criteria

Commands cover annotation, rule construction, contrastive training, hybrid retrieval, and leakage-resistant GCA. Outputs identify whether each result came from `rule` or `fallback`. No direct latent-to-motion decoder is included because the paper describes retrieval and leaves decoding to future work.

## Interactive data handoff

`scripts/start_demo.py` downloads one official BEAT BVH/TextGrid take, builds a local nine-clip bank and paired windows, resolves cached semantic spans to explicit strong rules, and locally fits a compact text-motion fallback in ignored outputs. The small demo substitutes TF-IDF features for Sentence-BERT before contrastive training; the full CLI above retains its documented encoder contract. The trace distinguishes strong-rule from fitted fallback routes. Cached annotations and a tiny fitted model are not evidence of the paper's rule extraction or retrieval quality. The older `--example` path remains an explicitly authored offline fixture. Speech is optional; recordings and fitted weights are not bundled.

## Bundled fictional avatar substitution

Two newly generated fictional CC0 humanoids replace the original avatar assets in the browser demo. They provide a 53-bone rig and named ARKit/viseme targets. Motion retargeting adapts source joints to their bind pose; speaking envelopes approximate mouth motion rather than phoneme alignment. The optional recorded BEAT companion inspects public motion, face and audio files prepared locally, independently of the paper's learned algorithm. No dataset recordings or trained weights are bundled.
