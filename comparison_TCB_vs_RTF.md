# Milestone 02 — Notebook Comparison: TCB vs. RTF

## Quick Reference: Final Results

| Model | TCB Test Acc | TCB Test Macro-F1 | RTF Test Acc | RTF Test Macro-F1 |
|---|---|---|---|---|
| Baseline | 0.5249 | 0.4487 | **0.5832** | — |
| Custom | 0.5206 | **0.4626** | **0.6433** | — |
| DistilBERT | **0.6457** | **0.5824** | 0.6017 | 0.45 |

**Summary:** RTF's baseline and custom model substantially outperformed TCB's. TCB's DistilBERT substantially outperformed RTF's. The best single model overall was TCB's DistilBERT (0.6457 acc / 0.5824 F1). RTF's best model was the custom BiLSTM+GloVe (0.6433 acc). They converge to nearly the same top accuracy via completely different paths.

---

## 1. Data Preparation

| Aspect | TCB | RTF |
|---|---|---|
| Raw records | 200,853 | 200,853 |
| Duplicate removal | None | Removed 491 duplicates |
| Empty record handling | Dropped 5 combined-text ≤ 5 chars | Dropped 6 empty headlines; filled 19,707 empty descriptions |
| **Final record count** | **200,848** | **200,356** |
| Train / Val / Test | 160,678 / 20,085 / 20,085 | 160,284 / 20,036 / 20,036 |
| Text construction | `headline [sep] short_description` | `headline [SEP] short_description` |
| Label encoding | Integer (sparse_categorical_crossentropy) | One-hot (categorical_crossentropy) |
| Vocabulary cap | 20,000 | 20,000 |
| Max sequence length | 128 | 128 |
| Batch size (non-BERT) | 256 | 128 |
| Class weighting | `compute_class_weight("balanced")` | `compute_class_weight("balanced")` |

### Key differences

**Duplicate removal:** RTF removed 491 duplicate text entries; TCB kept them on the grounds that identical headlines across different categories carry genuine label information. This is a defensible disagreement — RTF's view is that near-duplicates inflate training signal artificially; TCB's view is that the same headline legitimately appearing under different categories is informative ambiguity the model should learn.

**Label encoding:** Functionally equivalent. Integer labels + `sparse_categorical_crossentropy` (TCB) and one-hot + `categorical_crossentropy` (RTF) produce identical gradient updates. TCB's approach is slightly more memory-efficient (no one-hot matrix).

**Batch size:** TCB used 256, RTF used 128. Larger batches give noisier gradient estimates per update but run faster on GPU; smaller batches can generalize slightly better. This difference likely contributed to differing baseline results.

---

## 2. Baseline Model

### Architectures

**TCB:**
```
Embedding(20000, 64, mask_zero=True)
→ GlobalAveragePooling1D
→ Dense(256, relu)
→ Dropout(0.3)
→ Dense(41, softmax)
Total: 1,307,177 params
```

**RTF:**
```
Embedding(20000, 64)
→ GlobalAveragePooling1D
→ Dense(64, relu)
→ Dense(41, softmax)
Total: ~1,283,625 params (estimated)
```

### Training config

| Setting | TCB | RTF |
|---|---|---|
| Optimizer | Adam(lr=1e-3) | Adam(default) |
| Max epochs | 30 | 20 |
| Batch size | 256 | 128 |
| Early stopping patience | 5 | 5 |
| `mask_zero` | Yes | No |

### Results

| Metric | TCB | RTF |
|---|---|---|
| Best epoch | 5 | 14 |
| Train time | 1.3 min | 4.75 min |
| Val accuracy | 0.5230 | 0.5849 |
| Test accuracy | 0.5249 | **0.5832** |
| Test macro-F1 | 0.4487 | — (not reported) |

### Analysis

RTF's baseline was **+5.8% more accurate** than TCB's. The architectural differences are small (Dense 256 vs 64, Dropout vs none), so the gap is likely driven more by the **duplicate removal** and **smaller batch size** in RTF. Removing 491 duplicates improves the effective quality of the training set for the baseline because the duplicate entries can create shortcut patterns. RTF's baseline also trained for 14 epochs before peaking (vs. 5 for TCB), which suggests RTF's smaller batches allowed the optimizer to find a flatter, more generalizable minimum.

TCB added `Dropout(0.3)` and `mask_zero=True` which were theoretically sounder choices for regularization and handling padding — but didn't translate to better accuracy at this model tier.

---

## 3. Custom Model

### Approach — Fundamentally Different

This is the most significant divergence between the two notebooks.

**TCB** built a deeper from-scratch BiLSTM with **no pretrained embeddings**, relying entirely on the 160k training samples to learn embeddings:

```
Embedding(20000, 128, mask_zero=True)    ← randomly initialized
→ Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))
→ Bidirectional(LSTM(64, dropout=0.2))   ← stacked
→ Dense(256, relu)
→ BatchNormalization
→ Dropout(0.4)
→ Dense(41, softmax)
Total: 3,032,105 params
```

**RTF** used **GloVe 100-dimensional pretrained embeddings** with a shallower single-layer BiLSTM, iterating through three variants before settling on the best:

| RTF Variant | Architecture | Test Acc |
|---|---|---|
| V1: GloVe frozen | Embedding(GloVe, frozen) → GAP → Dense(64) | 0.5289 |
| V2: GloVe + Dropout | Embedding(GloVe, frozen) → GAP → Dropout(0.4) → Dense(64) → Dropout(0.3) | 0.4550 |
| V3: GloVe + BiLSTM (final) | Embedding(GloVe, **trainable**) → BiLSTM(64) → Dropout(0.3) → Dense(64) → Dropout(0.2) | **0.6433** |

### Training config

| Setting | TCB | RTF (V3) |
|---|---|---|
| Embedding | Random init, 128-dim | GloVe 100-dim, trainable |
| LSTM layers | 2 (stacked BiLSTM) | 1 (single BiLSTM) |
| LSTM units | 128 → 64 | 64 |
| Optimizer | Adam(5e-4) | Adam(default) |
| Max epochs | 30 | 20 |
| Batch size | 256 | 128 |

### Results

| Metric | TCB | RTF (V3) |
|---|---|---|
| Best epoch | 2 | 4 |
| Train time | 5.3 min | 121.7 min |
| Val accuracy | 0.5196 | **0.6512** |
| Test accuracy | 0.5206 | **0.6433** |
| Test macro-F1 | **0.4626** | — (not reported) |

### Analysis

RTF's custom model was **+12.3% more accurate** than TCB's. The decisive factor was **GloVe pretrained embeddings**. By initializing embeddings from a 6-billion-token corpus instead of random initialization, RTF's model started with word-level semantic relationships already encoded. When those embeddings were made trainable, the model fine-tuned them toward the news domain — combining broad linguistic knowledge (from GloVe) with task-specific adaptation.

TCB's approach doubled down on architecture (stacked BiLSTM, BatchNorm, larger embedding dim) but the additional complexity didn't compensate for learning embeddings from scratch on 160k examples. The model also hit its val_loss minimum at epoch 2 — suggesting it peaked extremely early, likely because the randomly-initialized embeddings had limited useful signal to extract via sequential processing.

RTF's V2 result (Dropout 0.4 → test acc 0.4550) is a useful cautionary data point: overly aggressive dropout on a frozen embedding model actively hurt performance by dropping too much of the already-limited signal coming through the non-trainable layer.

**Key takeaway:** For this dataset size and task, **pretrained embeddings mattered more than architectural depth**. A shallower model with GloVe beat a deeper model without it by a wide margin.

---

## 4. Pretrained Model (DistilBERT)

### Approach — Critical Difference in Fine-Tuning Strategy

Both used `distilbert-base-uncased`, but the fine-tuning strategies diverged entirely.

**TCB:** Two-phase fine-tuning, all in TensorFlow/Keras
- Phase 1: Freeze DistilBERT base, train classification head (GAP → Dropout(0.3) → Dense(41)) at lr=1e-3 for up to 5 epochs
- Phase 2: Unfreeze entire model, full fine-tune at lr=2e-5 for up to 5 epochs
- Total: both phases combined

**RTF:** Single phase, frozen base only, in **PyTorch** (switched frameworks due to TF incompatibility)
- Freeze entire DistilBERT base
- Train classification head only (Dense(64) → Dense(41)) at lr=2e-5 for up to 10 epochs
- No full fine-tuning attempted

### Why RTF used PyTorch

RTF's environment could not run `TFDistilBertModel` due to a `transformers` version incompatibility — the same Keras 2/3 conflict that was encountered in TCB's notebook (see Engineering Notes in technical_writeup.md). RTF worked around it by switching to PyTorch. TCB resolved it by pinning `transformers==4.47.1` and explicitly importing `tf_keras`.

### Results

| Metric | TCB | RTF |
|---|---|---|
| Training time | 135.2 min | 120.9 min |
| Val accuracy | **0.6476** | 0.6056 |
| Test accuracy | **0.6457** | 0.6017 |
| Test macro-F1 | **0.5824** | 0.45 |
| Early stopping triggered | Yes | No (still improving at epoch 10) |

### Analysis

TCB's DistilBERT was **+4.4% more accurate** and **+0.133 better macro-F1** than RTF's. The entire gap comes down to **full fine-tuning vs. frozen base**:

- RTF's frozen base kept all 66M DistilBERT parameters fixed. Only the ~2,600-param classification head (Dense(64) + Dense(41)) was learned. The pretrained representations were never adapted to the HuffPost domain. At epoch 10, val_loss was still declining — the model had not converged and was compute-limited, not capacity-limited.

- TCB's Phase 2 allowed all 66M parameters to shift toward the HuffPost domain at a carefully controlled lr=2e-5. This is the standard recommended approach for transformer fine-tuning and is why the TCB model substantially outperformed the frozen-base approach.

RTF acknowledged this explicitly: "Full fine-tuning would require 6–8+ additional hours (not pursued)." TCB's 135.2-minute run on the RTX 3060 made full fine-tuning feasible where it wasn't for RTF.

Interestingly, RTF's frozen DistilBERT (0.6017) was actually **the worst of RTF's three models** — outperformed by both the baseline (0.5832) and the custom BiLSTM (0.6433). The model still had not converged by epoch 10. This illustrates that frozen-base transfer learning for a 41-class classification task with significant domain specificity is insufficient.

---

## 5. Overall Comparison

### Performance by model tier

```
Test Accuracy:
          TCB      RTF
Baseline  0.5249   0.5832   ← RTF +5.8%
Custom    0.5206   0.6433   ← RTF +12.3%
BERT      0.6457   0.6017   ← TCB +4.4%

Test Macro-F1:
          TCB      RTF
Baseline  0.4487   —
Custom    0.4626   —
BERT      0.5824   0.45     ← TCB +13.2%
```

### What each notebook did better

**RTF did better at:**
- Baseline and custom model accuracy — substantially
- Empirical iteration: tested 3 variants of the custom model before picking the best
- Using GloVe pretrained embeddings (the single highest-leverage decision in the custom model tier)
- Providing more detailed confusion matrix and per-class breakdown for DistilBERT

**TCB did better at:**
- DistilBERT — substantially, due to two-phase fine-tuning
- Reporting macro-F1 consistently across all three models (RTF only reported it for DistilBERT)
- GPU-accelerated TF pipeline (avoided the PyTorch workaround)
- Resolving the Keras 2/3 version conflict rather than switching frameworks

### The key strategic trade-off

RTF invested pretrained knowledge at the **embedding layer** (GloVe), which paid off at the custom model tier. TCB invested pretrained knowledge at the **transformer level** (full DistilBERT fine-tuning), which paid off at the pretrained model tier.

Neither approach dominates outright. If compute is limited, GloVe+BiLSTM gives excellent accuracy-per-minute (RTF: 0.6433 in ~2 hours). If a GPU is available and the goal is maximum accuracy, full DistilBERT fine-tuning wins (TCB: 0.6457 acc, 0.5824 F1 in ~2 hours).

### Metric reporting gap

RTF did not report macro-F1 for the baseline or custom model — only accuracy. This makes a direct comparison on the primary metric impossible for those tiers. Given RTF's DistilBERT had test macro-F1 of 0.45 vs. TCB's 0.5824, TCB's full fine-tune approach likely would have produced much better F1 for the minority classes even if accuracies were comparable.

---

## 6. Methodological Differences Summary

| Decision | TCB | RTF | Impact |
|---|---|---|---|
| Duplicate removal | No | Yes (491 removed) | RTF baseline higher |
| Embedding strategy | Random init | GloVe 100D pretrained | RTF custom model higher |
| Custom model depth | 2×BiLSTM stacked | 1×BiLSTM | Less important than embedding choice |
| BERT fine-tuning | Two-phase (frozen → full) | Frozen base only | TCB BERT much higher |
| BERT framework | TF/Keras (tf_keras) | PyTorch (workaround) | Compatibility approach differed |
| Loss function | sparse_categorical_crossentropy | categorical_crossentropy | Functionally equivalent |
| Batch size | 256 | 128 | Smaller batch helped RTF baseline |
| Primary metric | Macro-F1 + accuracy | Accuracy (F1 only for BERT) | TCB more complete evaluation |
| Variant search | Single model per tier | 3 variants for custom | RTF more systematic exploration |
