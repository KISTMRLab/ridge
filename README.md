# RIDGE: Rule‐Infused Deep Learning for Realistic Co‐Speech Gesture Generation

**Ghazanfar Ali, Hwang Youn Kim, Jae‐In Hwang**

**Computer Animation and Virtual Worlds · 2025** · Published

[Paper / publisher](https://doi.org/10.1002/cav.70034) · [Project page](https://ghazanfarali.com/research/ridge/) · [BibTeX](citation.bib) · [Requirements](REQUIREMENTS.md) · [Code & setup](#implementation-and-usage)

> High-confidence rules and learned similarity retrieve semantically aligned gestures.

![RIDGE system architecture: rule-base construction, contrastive representation learning, and threshold-gated hybrid gesture retrieval](paper-assets/ridge-system.png)

*New scientific system diagram generated with image generation, based on RIDGE Figure 1 and Sections 3–4. Both inference paths retrieve recorded gesture clips. The diagram describes the paper's system; reimplementation differences are documented in REQUIREMENTS.md.*

## Why this research

Gesture rules offer strong semantic matches when the right phrase is known, but leave gaps for unfamiliar language. RIDGE combines explicit rules with learned retrieval to improve coverage while retaining recorded motion.

RIDGE first searches a motion-derived rule base enriched with language-model assistance. When a rule does not meet the confidence threshold, a contrastively trained text–motion embedding retrieves an appropriate recorded gesture. Both paths use existing animation segments; direct decoding into new motion frames is described as future work.

## Method at a glance

**Text query** → **Rule match or learned fallback** → **Recorded gesture clip**

| | Research system |
|---|---|
| Input | Text |
| Method | Rule retrieval with confidence gating; contrastive text–motion retrieval fallback |
| Output | Retrieved recorded gesture clips |

## Evidence and scope

Gesture Cluster Affinity: RIDGE 0.73, rule baseline 0.60, end-to-end baseline 0.52, ground truth 0.90

**Attribution:** These findings describe the paper or manuscript, not results obtained with this repository's code.

**Study context:** BEAT co-speech motion and in-the-wild video data.

**Limitations:** Direct latent-to-motion decoding is outside the study. Retrieved motion inherits source quality, including finger artifacts.

## Explore the implementation

Phrase alignment, confidence-gated rules, trainable text/motion projections, recorded-clip fallback and reference-fitted GCA. The compact baseline freezes Sentence-BERT; this differs from the paper's end-to-end training.

This repository contains independently written research code. The institute's original source, datasets and trained models are not distributed. Public-data preparation, commands, assumptions and checks are documented below and in [REQUIREMENTS.md](REQUIREMENTS.md).

## Resources and citation

Read the paper through its [publisher record](https://doi.org/10.1002/cav.70034). PDFs are hosted by publishers or preprint archives rather than stored in this repository.

Please cite the research paper when using its ideas; [download the BibTeX citation](citation.bib). The implementation has its own documented scope.

## Implementation and usage

<!-- implementation-guide -->

Independent educational reimplementation of *RIDGE: Rule-Infused Deep Learning for Realistic Co-Speech Gesture Generation* (Ali, Kim, and Hwang, Computer Animation and Virtual Worlds 2025, DOI: [10.1002/cav.70034](https://doi.org/10.1002/cav.70034)). RIDGE retrieves recorded clips through a high-confidence phrase rule first and a contrastively learned text-motion space otherwise. It does not decode new animation frames and is not institute source code.

### Setup and public data

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e . pytest
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` instead of the `source` line.

Run a CPU smoke workflow with procedurally generated inputs:

```bash
python scripts/smoke.py
```

It generates transcripts, reviewed annotations, 384-D paired embeddings, motion, and a local SentenceTransformer fixture, then invokes the installed `annotate`, `build-rules`, `train`, `retrieve`, and `eval-gca` CLI paths. Both rule and neural fallback retrieval are exercised, with results under `outputs/smoke/`. The local encoder replaces only downloadable Sentence-BERT weights; real `all-MiniLM-L6-v2` embeddings use the same checkpoint and index path.

Request [BEAT](https://pantomatrix.github.io/BEAT/) from its maintainers and prepare its text, timestamps, and upper-body motion under its license. Public videos may augment pretraining only when you have permission to process them. No BEAT files, wild videos, annotations, weights, proprietary prompts, or reported scores are bundled.

Transcript JSONL rows contain `record_id`, `text`, and `words: [{word,start_frame,end_frame}]`. Motion remains in a user-managed store keyed by the generated `record_id:start-end` gesture ID. Contrastive `pairs.npz` contains normalized `text_embeddings[N,384]`, neck-centered `motion[N,F,D]`, and string `ids[N]`. SBERT embeddings must come from `all-MiniLM-L6-v2` unless you intentionally retrain and rebuild every index.

```bash
ridge-gesture annotate --transcripts data/transcripts.jsonl --output outputs/phrases.jsonl
# Optional reviewed/LLM file: JSON object mapping record_id to 3-10 word phrase lists
ridge-gesture annotate --transcripts data/transcripts.jsonl --external-annotations data/reviewed.json --output outputs/phrases.jsonl
ridge-gesture build-rules --records data/transcripts.jsonl --annotations outputs/phrases.jsonl --output outputs/rules.jsonl
ridge-gesture train --pairs data/train_pairs.npz --output checkpoints/ridge.pt
ridge-gesture retrieve --rules outputs/rules.jsonl --checkpoint checkpoints/ridge.pt --threshold 0.72 --text "Explain the next important action" --output outputs/sequence.json
ridge-gesture eval-gca --reference data/gca_reference.npz --candidate data/gca_heldout_predictions.npz
python -m pytest
```

The annotation command uses a transparent heuristic unless reviewed annotations are supplied. The threshold is a validation parameter, not a paper-fixed universal value. Retrieval output labels every segment `rule` or `fallback`; output gesture IDs are the keys a separate renderer uses to load motion.

GCA reference and candidate files both contain `text_embeddings` and `motion_embeddings`. The reference file must contain training/reference speakers only. The candidate file contains held-out predictions; it never participates in fitting text clusters or gesture subclusters.

### Limits and licenses

This implementation freezes Sentence-BERT and trains its projection, while the paper describes end-to-end training; see `REQUIREMENTS.md`. Heuristic phrases are not equivalent to expert or LLM annotation. GCA measures affinity to a fitted cluster structure and still needs perceptual validation. Source motion quality, including finger artifacts, carries into retrieved clips. Code is MIT licensed; BEAT, pretrained encoders, videos, and annotations keep separate terms.

### Citation

Machine-readable metadata is in [citation.bib](citation.bib).

```bibtex
@article{ali2025ridge, title={RIDGE: Rule-Infused Deep Learning for Realistic Co-Speech Gesture Generation}, author={Ali, Ghazanfar and Kim, HwangYoun and Hwang, Jae-In}, journal={Computer Animation and Virtual Worlds}, volume={36}, number={4}, pages={e70034}, year={2025}, doi={10.1002/cav.70034}}
```
