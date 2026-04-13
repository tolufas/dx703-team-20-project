# Milestone 02 OPTIMAL EXPERIMENTS — Notebook Explainer

## Overview

This notebook implements the **Milestone 3 optimal RoBERTa classifier** for the HuffPost News Category dataset. It is the third iteration in a series:

- **Milestone 1:** EDA and problem framing
- **Milestone 2:** Baseline experiments (Embedding, BiLSTM, DistilBERT)
- **This notebook (Milestone 3):** Optimal RoBERTa model with six targeted improvements over the best Milestone 2 result

**Target to beat:** DistilBERT OPTIMAL v1 — test accuracy **0.6457**, macro-F1 **0.5824**

---

## Dataset

**Source:** HuffPost News Category Dataset v2 (HuggingFace `datasets`)

| Property | Value |
|---|---|
| Raw records | 200,853 |
| After filtering | 200,848 |
| Raw categories | 41 |
| Categories after consolidation | **34** |
| Train / Val / Test | 160,678 / 20,085 / 20,085 |
| Split strategy | Stratified by category (80/10/10) |

**Text construction:** `headline.lower() + " [sep] " + short_description.lower()`

**Class imbalance:** 32× ratio — POLITICS (32,739 samples) vs EDUCATION (1,004 samples). Class weights range from 0.18× (POLITICS) to 5.88× (EDUCATION).

### Label Consolidation: 41 → 34 Classes

Seven near-synonym pairs (those with F1 < 0.45 in Milestone 2) were merged:

| Merged | Into |
|---|---|
| ARTS & CULTURE | ARTS |
| CULTURE & ARTS | ARTS |
| STYLE | STYLE & BEAUTY |
| HEALTHY LIVING | WELLNESS |
| THE WORLDPOST | WORLDPOST |
| PARENTS | PARENTING |
| TASTE | FOOD & DRINK |

---

## Six Key Improvements Over Baseline

| # | Improvement | Replaces |
|---|---|---|
| 1 | Label consolidation (41→34) | Raw 41 classes |
| 2 | CLS token pooling | GlobalAveragePooling |
| 3 | Focal loss (γ=2.0) | Cross-entropy + class weights |
| 4 | Val_accuracy early stopping | Val_loss early stopping |
| 5 | Warmup + cosine decay LR schedule | Fixed LR in Phase 2 |
| 6 | Extended Phase 2 (40 epochs, patience=6) | 10 epochs, patience=3 |

**Rationale for improvement #4:** In Milestone 2 DistilBERT training, val_loss bottomed at epoch 2 while val_accuracy continued improving through epoch 6. Monitoring val_loss caused premature stopping, leaving accuracy gains on the table.

**Rationale for improvement #6:** The original 10-epoch / patience=3 limit was a compute constraint, not a model one. With extended training and patience=6 the model can continue improving through slow plateaus, particularly on minority classes under focal loss.

---

## Experiment Design

The notebook runs **three sequential experiments** overnight, each from fresh pretrained weights:

| # | Model | γ | Batch | Est. time |
|---|---|---|---|---|
| 1 | roberta-base | 2.0 | 32 | ~2–3 hrs |
| 2 | roberta-base | 1.5 | 16 | ~2–3 hrs |
| 3 | roberta-large | 2.0 | 16 | ~5–7 hrs |
| | | | **Total** | **~9–13 hrs** |

**Why compare γ=2.0 vs γ=1.5?** Higher γ down-weights easy examples more aggressively. On a 34-class imbalanced dataset it's unclear which wins without empirical comparison — γ=1.5 may generalize better if γ=2.0 over-focuses on noisy hard examples.

**Why add roberta-large?** RoBERTa-large (355M params, 24 layers, hidden=1024) represents a genuine accuracy ceiling increase vs. base (125M, 12 layers, hidden=768), typically +1–3% on classification. It fits in 12GB VRAM with batch=16 and fp16.

Each experiment saves to its own output directory (`results_base_gamma2.0/`, etc.) containing:
- `best_p1.weights.h5` — best Phase 1 checkpoint
- `best_p2.weights.h5` — best Phase 2 checkpoint
- `metrics.txt` — val/test accuracy and macro-F1
- `classification_report.txt` — per-class precision/recall/F1

---

## Model Architecture

The same architecture is used for base and large; hidden dimension scales automatically with the loaded model.

```
Input (input_ids, attention_mask) [batch, 128]
    ↓
TFRobertaModel (frozen in Phase 1, unfrozen in Phase 2)
    → last_hidden_state [batch, 128, 768/1024]
    ↓
CLS token extraction: hidden[:, 0, :]  [batch, 768/1024]
    ↓
Dropout(0.3)
    ↓
Dense(34, softmax, dtype=float32)
```

| Model | Params | Phase 1 Trainable | Phase 2 Trainable |
|---|---|---|---|
| roberta-base | 125M | 26K (head only) | 125M (all) |
| roberta-large | 355M | ~50K (head only) | 355M (all) |

**Notes:**
- `dtype='float32'` on the final Dense is required for numerical stability under mixed precision — keeps the softmax in fp32 regardless of the fp16 policy applied to the rest of the model.
- `training=False` is intentionally omitted from the base model call so Keras propagates the runtime training flag through the graph, enabling RoBERTa's internal dropout during `model.fit()`.

**Tokenizer:** `RobertaTokenizerFast` (BPE), max_length=128. Only ~0.1% of samples exceed 128 tokens. Tokenization is done once before the experiment loop — the resulting numpy arrays are rebatched per-experiment with the appropriate batch size.

---

## Training: Two-Phase Fine-Tuning

### Phase 1 — Head Only (Frozen Base)

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Loss | Focal loss (γ per experiment) |
| Early stopping | val_accuracy, patience=4 |
| Max epochs | 10 |
| Batch size | 32 (base) / 16 (large) |

**Goal:** Warm up the classification head before the base is unfrozen, preventing large random-initialized gradients from destabilizing pretrained weights. Head should converge in 3–4 epochs; 10 epoch cap with patience=4 gives it room.

### Phase 2 — Full Fine-Tuning (Unfrozen Base)

| Setting | Value |
|---|---|
| Optimizer | Adam + WarmupCosineDecay schedule |
| Peak LR | 2e-5 |
| Warmup | Linear 0 → 2e-5 over 2 epochs |
| Decay | Cosine 2e-5 → ~0 over remaining 38 epochs |
| Total schedule steps | 40 × steps_per_epoch |
| Loss | Focal loss (γ per experiment) |
| Early stopping | val_accuracy, patience=6 |
| Max epochs | 40 |
| Batch size | 32 (base) / 16 (large) |

**Why 2-epoch warmup?** With a 40-epoch total budget, a 1-epoch warmup is a smaller fraction of training than with the original 10-epoch plan. Two epochs gives the base more time to activate gradually before full LR hits.

**Why patience=6?** With focal loss on 34 imbalanced classes, val_accuracy can plateau for 2–3 epochs then tick up again as the model refines minority class boundaries. Patience=3 risks cutting off runs that are still improving.

**Goal:** Adapt all base parameters while preventing catastrophic forgetting via the controlled warmup. The extended cosine tail (epochs 10–40) provides strong LR-decay regularization during the final stages of convergence.

### Focal Loss

Replaces cross-entropy + class_weight. Applies a modulating factor `(1-p)^γ` that down-weights easy/confident predictions and focuses training on hard examples and minority classes:

```
loss = mean((1 - p_correct)^γ * cross_entropy)
```

### Mixed Precision

`tf_keras.mixed_precision.set_global_policy("mixed_float16")` is set at the top of the notebook. On the RTX 3060 (Ampere, compute 8.6) this gives ~1.7× faster training with no accuracy impact. The policy must be set via `tf_keras.mixed_precision` — not `tensorflow.keras.mixed_precision` — as these are separate namespaces and only the former applies to `TFRobertaModel`.

---

## Evaluation

Once all experiments complete, the notebook produces:

1. **Results comparison table** — all three experiments vs. DistilBERT baseline, with Δ accuracy and Δ macro-F1
2. **Training curves** — accuracy and loss per epoch for each experiment, with Phase 1/2 boundary marked (one row per experiment)
3. **Per-class F1 comparison** — side-by-side bar chart for all 34 classes across experiments
4. **Sample misclassifications** — from the best-performing experiment

**Baseline categories to watch from Milestone 2:**

| Category | Milestone 2 F1 | Notes |
|---|---|---|
| GOOD NEWS | 0.352 | Hardest — vague/overlapping content |
| IMPACT | 0.371 | Hardest |
| EDUCATION | 0.383 | Hardest — smallest class (1,004 samples) |
| STYLE & BEAUTY | 0.868 | Easiest |
| WEDDINGS | 0.859 | Easiest |
| SPORTS | 0.844 | Easiest |

---

## Milestone 2 Baseline Comparison (Context)

| Model | Test Accuracy | Macro-F1 | Notes |
|---|---|---|---|
| Embedding + GAP baseline | 0.5832 | — | Word-order-insensitive ceiling |
| BiLSTM + GloVe (frozen) | 0.5289 | — | Frozen embeddings underperform |
| BiLSTM + GloVe (trainable) | 0.6433 | — | Best non-transformer result |
| DistilBERT frozen base | 0.6017 | — | Phase 1 only |
| **DistilBERT OPTIMAL v1** | **0.6457** | **0.5824** | **← target to beat** |
| RoBERTa-base γ=2.0 | *pending* | *pending* | |
| RoBERTa-base γ=1.5 | *pending* | *pending* | |
| RoBERTa-large γ=2.0 | *pending* | *pending* | |

---

## Key Findings from Prior Milestones

1. **Word order matters:** BiLSTM (+0.6%) beats bag-of-words; sequence is critical for overlapping categories.
2. **Pretrained embeddings are decisive:** GloVe gives +6.2% over random init.
3. **Trainable embeddings required:** Frozen embeddings significantly underperform.
4. **Full fine-tuning is critical:** Two-phase fine-tuning (+4.4%) over frozen-base approach.
5. **Monitor val_accuracy not val_loss:** On imbalanced multi-class data, these diverge — val_loss may bottom while accuracy is still rising.

---

## Libraries & Environment

| Component | Version |
|---|---|
| TensorFlow | 2.21.0 |
| Keras (legacy) | tf_keras |
| Transformers | 4.47.1 |
| HuggingFace datasets | — |
| scikit-learn | — |
| GPU | NVIDIA RTX 3060 (12,288 MB) |
| Mixed precision | mixed_float16 (tf_keras namespace) |

---

## Results Analysis — Section to Add

> **Once training completes, add results here covering:**
>
> - Phase 1 and Phase 2 training curves for all three experiments
> - Final val accuracy, val macro-F1, test accuracy, test macro-F1 per experiment
> - Full comparison table vs. DistilBERT baseline
> - γ=2.0 vs γ=1.5: which focal loss setting wins and on what class types?
> - roberta-base vs roberta-large: accuracy ceiling gain and whether it justifies the 4× compute cost
> - Per-class F1: which of the 34 consolidated classes improved most vs. Milestone 2?
> - Whether merging the 7 label pairs measurably improved their F1
> - Analysis of remaining hard categories (GOOD NEWS, IMPACT, EDUCATION)
> - Misclassification patterns — are errors still concentrated on semantically similar pairs?
> - Verdict: did RoBERTa + the six improvements beat the 0.6457 / 0.5824 baseline?
