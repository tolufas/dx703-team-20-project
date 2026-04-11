# Milestone 02 — Three-Way Notebook Comparison: TCB vs. RTF vs. LJK

## Completion Status

| Problem | TCB | RTF | LJK |
|---|---|---|---|
| 1 – Data Preparation | ✅ Complete | ✅ Complete | ✅ Complete |
| 2 – Baseline Model | ✅ Complete | ✅ Complete | ✅ Complete |
| 3 – Custom Model | ✅ Complete | ✅ Complete | ❌ Not implemented |
| 4 – Pretrained Model | ✅ Complete | ✅ Complete | ❌ Not implemented |
| 5 – Comparative Evaluation | ✅ Complete | ✅ Complete | ❌ Not implemented |

LJK submitted Problems 1 and 2 only. All written answers for Problems 3–5 are template placeholders with no code or results.

---

## Quick Reference: All Results

| Model | TCB Test Acc | TCB Macro-F1 | RTF Test Acc | RTF Macro-F1 | LJK Test Acc | LJK Macro-F1 |
|---|---|---|---|---|---|---|
| Baseline | 0.5249 | 0.4487 | **0.5832** | — | 0.5176 | — |
| Custom | 0.5206 | **0.4626** | **0.6433** | — | — | — |
| DistilBERT | **0.6457** | **0.5824** | 0.6017 | 0.45 | — | — |

---

## 1. Data Preparation

### Comparison Table

| Aspect | TCB | RTF | LJK |
|---|---|---|---|
| Raw records | 200,853 | 200,853 | 200,853 |
| Empty record removal | 5 near-empty (text ≤ 5 chars) | 6 empty headlines | 6 empty headlines |
| Duplicate removal | None | 491 removed | ~1,398 removed (headline+category) |
| **Final record count** | **200,848** | **200,356** | **199,449** |
| Train / Val / Test | 160,678 / 20,085 / 20,085 | 160,284 / 20,036 / 20,036 | 159,559 / 19,945 / 19,945 |
| Text construction | `headline [sep] short_description` | `headline [SEP] short_description` | `headline [SEP] short_description` |
| Tokenization | Custom `TextVectorization` (20k vocab) | Custom `TextVectorization` (20k vocab) | **DistilBERT `AutoTokenizer`** (30,522 vocab) |
| Max sequence length | 128 | 128 | 128 |
| Label encoding | Integer → `sparse_categorical_crossentropy` | One-hot → `categorical_crossentropy` | Integer → `sparse_categorical_crossentropy` |
| Batch size (non-BERT) | 256 | 128 | 256 |
| Class weighting | `compute_class_weight("balanced")` | `compute_class_weight("balanced")` | `compute_class_weight("balanced")` |

### Key Differences

**LJK tokenized with DistilBERT's tokenizer in Problem 1.** This is the most architecturally significant data prep decision across all three notebooks. Instead of using a Keras `TextVectorization` layer fitted on the training data (TCB, RTF), LJK applied `AutoTokenizer.from_pretrained("distilbert-base-uncased")` to the entire dataset upfront. This produced `input_ids` from DistilBERT's 30,522-token WordPiece vocabulary, which were then passed directly as integer sequences to every model — including the baseline Embedding model.

The consequence is that LJK's "baseline" Embedding layer is embedding DistilBERT subword tokens, not simple whitespace-split words. This is a notable difference: the vocabulary is richer (30,522 subword units rather than 20,000 full words), and the tokenization is linguistically more sophisticated. However, the embedding weights themselves are still randomly initialized — LJK is not using the DistilBERT weights in the baseline, just its vocabulary.

**Duplicate removal:** All three removed empty headlines. LJK removed the most records (deduplication on headline+category), followed by RTF. TCB kept duplicates, arguing they carry legitimate label information.

**Separator token casing:** TCB used lowercase `[sep]`, RTF and LJK used `[SEP]`. For DistilBERT (uncased), this makes no difference. For TCB's custom vocabulary it is purely cosmetic.

---

## 2. Baseline Model

### Architecture Comparison

| Layer | TCB | RTF | LJK |
|---|---|---|---|
| Embedding vocab | 20,000 (custom vocab) | 20,000 (custom vocab) | **30,522 (DistilBERT vocab)** |
| Embedding dim | 64 | 64 | 64 |
| Pooling | GlobalAveragePooling1D | GlobalAveragePooling1D | GlobalAveragePooling1D |
| Hidden Dense | Dense(256, relu) | Dense(64, relu) | Dense(128, relu) |
| Regularization | Dropout(0.3) | None | None |
| Output | Dense(41, softmax) | Dense(41, softmax) | Dense(41, softmax) |
| `mask_zero` | Yes | No | No |
| **Total params** | **1,307,177** | **~1,284,000** | **~1,967,000** |

LJK's baseline has the most parameters despite similar architecture, solely due to the larger embedding matrix (30,522 × 64 = 1,953,408 vs. 20,000 × 64 = 1,280,000).

### Training Configuration

| Setting | TCB | RTF | LJK |
|---|---|---|---|
| Optimizer | Adam(lr=1e-3) | Adam(default) | Adam(default) |
| Loss | sparse_categorical_crossentropy | categorical_crossentropy | sparse_categorical_crossentropy |
| Max epochs | 30 | 20 | 20 |
| Batch size | 256 | 128 | 256 |
| Early stopping patience | 5 | 5 | 5 |

### Results

| Metric | TCB | RTF | LJK |
|---|---|---|---|
| Best epoch | 5 | 14 | 12 |
| Train time | **1.3 min** | 4.75 min | 1.3 min |
| Val accuracy | 0.5230 | **0.5849** | 0.5138 |
| Test accuracy | 0.5249 | **0.5832** | 0.5176 |
| Test macro-F1 | **0.4487** | — | — |

### Analysis

RTF had the strongest baseline by a significant margin (+5.8% over TCB, +6.6% over LJK). As discussed in the TCB/RTF comparison, this is likely driven by RTF's smaller batch size (128 vs 256) and duplicate removal.

LJK's baseline was the weakest despite having the largest embedding matrix. The DistilBERT vocabulary's 30,522 subword tokens are not necessarily more useful than a 20k word-level vocabulary when the embeddings are randomly initialized — subword tokenization only provides an advantage when using the pretrained weights that correspond to those tokens. Without pretrained weights, the larger vocabulary just means more parameters to learn from the same data.

TCB's `mask_zero=True` and Dropout(0.3) were theoretically better design choices for regularization but didn't translate to higher baseline accuracy.

All three baselines show the same pattern: training accuracy significantly outpaces validation accuracy (overfitting), early stopping triggers before max epochs, and test accuracy closely tracks validation accuracy.

---

## 3. Custom Model

*LJK did not implement this problem.*

### TCB vs. RTF

| Aspect | TCB | RTF |
|---|---|---|
| Embedding | Random init, 128-dim, 20k vocab | **GloVe 100D pretrained**, trainable |
| Architecture | 2× stacked Bidirectional LSTM (128→64) | 1× Bidirectional LSTM (64 units) |
| Hidden Dense | Dense(256) + BatchNorm + Dropout(0.4) | Dense(64) + Dropout(0.3/0.2) |
| Total params | 3,032,105 | ~2,100,000 |
| Test accuracy | 0.5206 | **0.6433** |
| Test macro-F1 | **0.4626** | — |
| Train time | 5.3 min | 121.7 min |
| Best epoch | 2 | 4 |

RTF's GloVe pretrained embeddings provided the decisive advantage (+12.3% accuracy). TCB's deeper architecture (stacked BiLSTMs, BatchNorm) didn't compensate for starting with randomly initialized embeddings on a 160k dataset. The key lesson: **pretrained embeddings at the word level gave a larger boost than architectural complexity**.

RTF also explored the problem empirically, testing three variants (frozen GloVe GAP, GloVe+Dropout, GloVe+BiLSTM+Dropout) before selecting the best. TCB went straight to a single architecture.

---

## 4. Pretrained Model (DistilBERT)

*LJK did not implement this problem.*

### TCB vs. RTF

| Aspect | TCB | RTF |
|---|---|---|
| Framework | TensorFlow / tf_keras | **PyTorch** (TF workaround failed) |
| Base model | distilbert-base-uncased | distilbert-base-uncased |
| Fine-tuning strategy | **Two-phase** (frozen head → full fine-tune) | Frozen base only (head trained) |
| Phase 1 LR | 1e-3 | 2e-5 |
| Phase 2 LR | 2e-5 | — |
| Classification head | GAP → Dropout(0.3) → Dense(41) | Dense(64) → Dense(41) |
| Batch size | 32 | 32 |
| Val accuracy | **0.6476** | 0.6056 |
| Test accuracy | **0.6457** | 0.6017 |
| Test macro-F1 | **0.5824** | 0.45 |
| Train time | 135.2 min | 120.9 min |
| Early stopping triggered | Yes | No (still improving at epoch 10) |

TCB's two-phase fine-tuning (+4.4% accuracy, +0.132 macro-F1 over RTF) is the standard recommended approach for transformer fine-tuning and makes the difference here. RTF's frozen base limited adaptation to the HuffPost domain; the model was still improving at epoch 10 and had not converged, meaning it was compute-constrained rather than capacity-constrained.

RTF had to switch to PyTorch due to the Keras 2/3 version conflict with the `transformers` library. TCB resolved this by pinning `transformers==4.47.1` and explicitly importing `tf_keras`.

---

## 5. Strategic Decisions — Three-Way Summary

### Data pipeline design

LJK made the most forward-looking decision in Problem 1: by applying the DistilBERT tokenizer upfront, all downstream models share the same tokenized representation. This eliminated the need to re-tokenize for the pretrained model in Problem 4 (had it been implemented) — the `input_ids` produced in Problem 1 would have been directly usable by `TFDistilBertModel`. The trade-off is that the baseline embedding model uses a much larger vocabulary than necessary.

TCB and RTF both used a custom `TextVectorization` layer, which is efficient for the RNN-based models but required a second tokenization pass for DistilBERT (Problem 4). This caused practical friction: TCB had to manage two tokenization pipelines, and RTF had to handle the framework switch to PyTorch.

### Pretrained knowledge injection

| Notebook | Where pretrained knowledge was used | Effect |
|---|---|---|
| TCB | DistilBERT weights (Problem 4 only) | +27% over baseline |
| RTF | GloVe embeddings (Problem 3) + DistilBERT weights (Problem 4) | +10.3% at P3, +1.9% at P4 |
| LJK | DistilBERT tokenizer vocabulary (Problem 1 setup only) | Minimal effect on baseline |

RTF was the only notebook to use pretrained knowledge at the embedding layer for the custom model, which was the single highest-leverage decision across all three notebooks at the custom model tier. TCB reserved all pretrained knowledge for the transformer model. LJK used DistilBERT's vocabulary but not its weights anywhere.

### Empirical rigor

RTF tested the most variants (3 custom model architectures), demonstrating iterative tuning. TCB implemented one architecture per problem but solved harder engineering problems (Keras conflict, two-phase fine-tuning). LJK completed two problems cleanly.

### Evaluation completeness

Only TCB reported macro-F1 consistently across all three models. RTF reported it only for DistilBERT. LJK reported only accuracy. Macro-F1 is the more informative metric for a 41-class imbalanced dataset — without it, it's impossible to know how well a model handles minority classes.

---

## 6. Overall Rankings

### By best single model achieved

| Rank | Notebook | Best model | Test Acc | Test Macro-F1 |
|---|---|---|---|---|
| 1 | **TCB** | DistilBERT (two-phase fine-tune) | **0.6457** | **0.5824** |
| 2 | **RTF** | GloVe + BiLSTM | **0.6433** | — |
| 3 | **LJK** | Baseline only | 0.5176 | — |

### By strongest baseline

| Rank | Notebook | Baseline Test Acc |
|---|---|---|
| 1 | **RTF** | **0.5832** |
| 2 | TCB | 0.5249 |
| 3 | LJK | 0.5176 |

### By completeness

| Rank | Notebook | Problems completed |
|---|---|---|
| 1 (tie) | **TCB** | 5/5 with full metrics |
| 1 (tie) | **RTF** | 5/5 (F1 missing for P2/P3) |
| 3 | **LJK** | 2/5 |

---

## 7. What Each Notebook Could Learn from the Others

**TCB from RTF:**
- Use GloVe or another pretrained embedding in the custom model — a shallow BiLSTM with GloVe handily beats a deep BiLSTM trained from scratch at this data scale
- Test multiple variants before committing to an architecture for the custom model

**TCB from LJK:**
- Pre-tokenize with the DistilBERT tokenizer in Problem 1 so a single pipeline feeds all three models — eliminates the second tokenization pass in Problem 4

**RTF from TCB:**
- Two-phase fine-tuning (frozen head → full fine-tune) is essential for DistilBERT — frozen-base-only limits the model and leaves performance on the table
- Report macro-F1 for all models, not just DistilBERT — accuracy alone is misleading on a 32× imbalanced dataset
- Pinning `transformers==4.47.1` and using `tf_keras` avoids the PyTorch workaround

**RTF from LJK:**
- Pre-tokenizing with the DistilBERT tokenizer upfront simplifies the pretrained model pipeline

**LJK from TCB and RTF:**
- Complete Problems 3–5
- Report macro-F1 alongside accuracy
