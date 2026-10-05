# RIDGE: Rule‐Infused Deep Learning for Realistic Co‐Speech Gesture Generation

**Ghazanfar Ali, Hwang Youn Kim, Jae‐In Hwang**

**Computer Animation and Virtual Worlds · 2025** · Published

[Paper / publisher](https://doi.org/10.1002/cav.70034) · [Project page](https://ghazanfarali.com/research/ridge/) · [BibTeX](citation.bib) · [Requirements](REQUIREMENTS.md) · [Code & setup](#implementation-and-usage)

> High-confidence rules and learned similarity retrieve semantically aligned gestures.

![RIDGE system architecture: rule-base construction, contrastive representation learning, and threshold-gated hybrid gesture retrieval](paper-assets/ridge-system.png)

*Graphical abstract diagram. Confidence-gated rules and learned similarity retrieve recorded gesture clips.*

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

<!-- demo-preview:start -->
## Demo preview

![Ridge runnable demo](demo-assets/preview.png)

*The prepared BEAT sequence shows explicit strong-rule and locally fitted fallback routes. This preview is not a paper benchmark.*

From the repository root, using the Python environment described below:

```sh
python -m pip install -e .
python -m pip install -r scripts/requirements-demo.txt
python scripts/start_demo.py
```

Open **http://127.0.0.1:8080/**. First launch downloads one official BEAT BVH and matching TextGrid, prepares nine clips and disjoint paired windows in ignored `outputs/`, resolves three cached semantic annotations to strong rules, and fits a compact text-motion fallback locally. The fallback uses TF-IDF text features in place of Sentence-BERT for this small demo. Choose suggested utterances to compare strong-rule and trained fallback routes, then click **Play speech + gesture**. Stop cancels speech, and scrubbing previews a pose. The first launch also downloads pinned Three.js modules. Public recordings and fitted weights remain local.

The 3D presentation uses shared Three.js avatar components and bundled fictional CC0 characters. The paper-specific algorithms and data adapters live in this repository.


To replace the demo motion with an existing processed BEAT take, run `python scripts/prepare_beat_demo.py --processed /path/to/processed/beat`, then restart the server. Use `--rebuild --epochs 80` to regenerate the public sample and refit the small adapter. For a larger bank, the documented full-data CLI below retains the paper-specific input contracts.

<!-- demo-preview:end -->

## Implementation and usage

<!-- implementation-guide -->

Independent educational reimplementation of *RIDGE: Rule-Infused Deep Learning for Realistic Co-Speech Gesture Generation* (Ali, Kim, and Hwang, Computer Animation and Virtual Worlds 2025, DOI: [10.1002/cav.70034](https://doi.org/10.1002/cav.70034)). RIDGE retrieves recorded clips through a high-confidence phrase rule first and a contrastively learned text-motion space otherwise. It does not decode new animation frames and is not institute source code.

The default browser path is the prepared BEAT demo above. The older `python scripts/demo_server.py --example` path, when the prepared BEAT cache is absent, remains an offline fixture with author-created motion and illustrative vectors. The prepared-data commands below retain the full contrastive training and hybrid retrieval contracts.

The default strong rules come from three cached semantic annotations with explicit provenance. To run your own OpenAI-compatible extractor against the prepared local bank, supply an endpoint and model, then rebuild the local RIDGE index with those reviewed rules:

```sh
python scripts/beat_semantics.py --endpoint http://127.0.0.1:1234/v1 --model MODEL --output outputs/beat-library/strong-rules.json
python scripts/prepare_beat_demo.py --strong-rules outputs/beat-library/strong-rules.json
```

The extractor copies contiguous transcript spans and checks their alignment to bank clips. It does not infer gesture meaning from motion. The demo's cached annotations were generated with an assistant and remain reviewable in `scripts/beat-semantic-annotations.json`. The full paper method follows the rule-map and pose-matching lineage of [Automatic Text-to-Gesture](https://github.com/ghazanPK/automatic-text-to-gesture) and [Wild Pose Matching](https://github.com/ghazanPK/wild-pose-matching), but this repository runs independently.

```bash
python -m pip install -e .
python scripts/prepare_viewer.py --out static/vendor
python scripts/demo_server.py --example
```

### Setup and public data

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e . pytest
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` instead of the `source` line.

Run a CPU verification workflow with procedurally generated inputs:

```bash
python scripts/verify.py
```

It generates transcripts, reviewed annotations, 384-D paired embeddings, motion, and a local SentenceTransformer fixture, then invokes the installed `annotate`, `build-rules`, `train`, `retrieve`, and `eval-gca` CLI paths. Both rule and neural fallback retrieval are exercised, with results under `outputs/verification/`. The local encoder replaces only downloadable Sentence-BERT weights; real `all-MiniLM-L6-v2` embeddings use the same checkpoint and index path.

Request [BEAT](https://pantomatrix.github.io/BEAT/) from its maintainers and prepare its text, timestamps, and upper-body motion under its license. Public videos may augment pretraining only when you have permission to process them. No BEAT recordings, wild videos, original performer annotations, trained weights, proprietary prompts, or reported scores are bundled. The demo includes three cached semantic phrase annotations from the public transcript, with their provenance recorded.

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

### Prepare data and compare retrieval branches in the browser

`scripts/prepare_public_data.py` converts a licensed BVH and timestamped JSONL transcript to 15 FPS neck-centered motion, phrase records and real `all-MiniLM-L6-v2` text embeddings. Use at least six seconds and the script's upper-body joint names. One word per JSONL line or one record containing `words` is accepted; timestamps can use seconds or frame indices. The adapter does not synthesize expert/LLM phrase annotations. Review `annotate` output or supply `--external-annotations` for meaningful rules.

```bash
python scripts/prepare_public_data.py --bvh data/licensed_motion.bvh --transcript data/words.jsonl --output-dir data/prepared
ridge-gesture annotate --transcripts data/prepared/transcripts.jsonl --output outputs/phrases.jsonl
ridge-gesture build-rules --records data/prepared/transcripts.jsonl --annotations outputs/phrases.jsonl --output outputs/rules.jsonl
ridge-gesture train --pairs data/prepared/train_pairs.npz --epochs 20 --output checkpoints/ridge.pt
python scripts/prepare_viewer.py --out static/vendor
python scripts/demo_server.py --data-dir data/prepared --rules outputs/rules.jsonl --checkpoint checkpoints/ridge.pt
```

The threshold slider changes the rule gate live. The trace labels each selected clip as `rule` or `fallback`, shows similarity and plays its actual motion frames. Use speaker-separated reference and candidate files with `ridge-gesture eval-gca` for the GCA diagnostic; the local demo does not assert the paper's reported score. The compact checkpoint freezes Sentence-BERT and trains a text projection and temporal motion encoder. `scripts/verify.py` uses random arrays plus a local illustrative text encoder solely to exercise interfaces.

[Automatic text-to-gesture](https://github.com/ghazanPK/automatic-text-to-gesture) is the rule-mining precursor; [wild pose matching](https://github.com/ghazanPK/wild-pose-matching) and [multilingual gesture synthesis](https://github.com/ghazanPK/multilingual-gesture) develop the GestureCLR lineage. These are research references, not package dependencies.

### Limits and licenses

This implementation freezes Sentence-BERT and trains its projection, while the paper describes end-to-end training; see `REQUIREMENTS.md`. Heuristic phrases are not equivalent to expert or LLM annotation. GCA measures affinity to a fitted cluster structure and still needs perceptual validation. Source motion quality, including finger artifacts, carries into retrieved clips. Code is MIT licensed; BEAT, pretrained encoders, videos, and annotations keep separate terms.

### Citation

Machine-readable metadata is in [citation.bib](citation.bib).

```bibtex
@article{ali2025ridge, title={RIDGE: Rule-Infused Deep Learning for Realistic Co-Speech Gesture Generation}, author={Ali, Ghazanfar and Kim, HwangYoun and Hwang, Jae-In}, journal={Computer Animation and Virtual Worlds}, volume={36}, number={4}, pages={e70034}, year={2025}, doi={10.1002/cav.70034}}
```

### Optional local speech adapters

The viewer can speak its query or transcribe user-selected audio. Browser voice and typed text work without model weights. Install `python -m pip install -e ".[speech]"` for local adapters. Obtain Kokoro files from [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) yourself: `config.json`, `kokoro-v1_0.pth` and `voices/af_heart.pt`. Set `KOKORO_MODEL_DIR` to their parent folder before launching the server. Follow [Kokoro's English phonemizer setup](https://github.com/hexgrad/kokoro), including espeak-ng where required, then choose Local Kokoro. For ASR, set `WHISPER_MODEL_DIR` to a user-downloaded [faster-whisper](https://github.com/SYSTRAN/faster-whisper) small model directory containing `model.bin` and its tokenizer/configuration files. ASR runs on CPU with INT8, requests word timestamps and VAD, and disables implicit model downloads. No speech model files or audio recordings are included in this repo.

<!-- avatar-recorded-motion:start -->
## Bundled characters and recorded public motion

The browser demos include Rowan and Mira, two new fictional GLB characters built with MPFB and MakeHuman community assets under CC0 1.0. See [avatar licensing and provenance](static/avatars/LICENSE.md). Use the character selector in the stage. The shared renderer supports body bones, ARKit facial channels, and approximate speaking motion.

Recorded motion is adapted to the characters' proportions. Palm landmarks set hand orientation; finger curl uses bounded hinge bends and preserves the character's finger spacing. Thumb-base opposition stays in the authored pose, with conservative recorded curl at the remaining joints. Distal bends are estimated from the preceding joint when fingertip landmarks are absent. Use the companion's hand close-up views to inspect the result.

The [avatar motion companion](static/recorded-motion.html) opens at `/recorded-motion.html` while the demo server is running. A small authored motion and face sample loads automatically; click **Play** without uploading files. It also plays locally selected BEAT motion, face, and WAV files on the bundled characters. These are presentation and data-inspection tools, separate from the paper implementation. No BEAT recording, dataset archive, or trained model is bundled. For recorded public motion, install the one preparation dependency and fetch a small official sample into ignored `outputs/beat-demo/`:

```sh
python -m pip install numpy
python scripts/beat_demo/fetch_modalities.py --speaker 1 --sequence 1_wayne_0_1_1 --include-bvh --max-bytes 25000000 --output-dir outputs/beat-demo/source
python scripts/beat_demo/prepare_bvh.py --bvh outputs/beat-demo/source/1_wayne_0_1_1.bvh --output outputs/beat-demo/sample/1_wayne_0_1_1-raw-motion.json --frames 120
python scripts/beat_demo/prepare_modalities.py --sequence 1_wayne_0_1_1 --source outputs/beat-demo/source --output outputs/beat-demo/sample --frames 120
```

Open the companion and select `outputs/beat-demo/sample/1_wayne_0_1_1-raw-motion.json`, `1_wayne_0_1_1-face.json`, and `1_wayne_0_1_1.wav`. The downloader caps each original file at 25 MB; the prepared clip contains up to 120 frames. The viewer uses local files and does not upload them. For other BEAT takes, substitute a matching official speaker and sequence ID.

If you already have OmniMo's processed 52-joint Unity humanoid data, use that normalized motion instead:

```sh
python scripts/beat_demo/prepare.py --dataset /path/to/processed/beat --speaker 1 --take 1_wayne_0_1_1 --output outputs/beat-demo/sample/1_wayne_0_1_1-motion.json --max-frames 120
```

Select the resulting `*-motion.json` in the companion. Its metadata carries the humanoid joint mapping and source-to-avatar coordinate conversion. The viewer fits source FK directions from the avatar's bind pose, following the spine explicitly at branching joints. This avoids applying incompatible source bone twist to the MPFB skin; it does not reproduce exact performer twist. The adapter supports Unity proximal/intermediate/distal finger names. Raw BVH remains a public-data alternative; do not mix the two skeleton conventions.
<!-- avatar-recorded-motion:end -->
