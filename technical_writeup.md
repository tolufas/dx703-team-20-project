# DX703 Milestone 02 — Technical Writeup

## Overview

This document describes the full ML pipeline built for HuffPost news category classification across five problems: data preparation, a bag-of-embeddings baseline, a custom Bidirectional LSTM, DistilBERT transfer learning, and comparative evaluation.

**Dataset:** HuffPost News Category Dataset — 200,853 records, 41 topical categories (e.g., POLITICS, WELLNESS, TRAVEL).
**Task:** Multi-class text classification (41 classes) from short news headlines and descriptions.
**Primary metric:** Macro-averaged F1 (treats all 41 classes equally regardless of sample count).
**Hardware:** NVIDIA RTX 3060 (12 GB VRAM) via WSL2, TensorFlow 2.20, CUDA 12.5.

---

## Problem 1 — Data Preparation and Splits

### Loading

The dataset is loaded from HuggingFace via `load_dataset("json", ...)` and cached to disk with `save_to_disk` / `load_from_disk` to avoid repeated downloads. Each record contains `headline`, `short_description`, and `category`.

### Text Normalization

Headlines and short descriptions are lowercased and concatenated with a literal `[sep]` separator token:

```python
text = headline.lower() + " [sep] " + short_description.lower()
```

The `[sep]` token mirrors the BERT `[SEP]` convention, which becomes meaningful when the same text field is passed to DistilBERT in Problem 4. 5 near-empty records (combined text ≤ 5 characters) were dropped, leaving **200,848 records**.

### Label Encoding

String category labels are integer-encoded using HuggingFace's `class_encode_column`, which produces a consistent `label_names` list and integer `category` column. This is required for `sparse_categorical_crossentropy` in Keras.

### Stratified 80/10/10 Split

A two-stage stratified split is used to preserve class proportions in all three subsets:

```python
# Stage 1: carve off 10% test set
tmp = huff2.train_test_split(test_size=0.10, seed=42, stratify_by_column="category")

# Stage 2: split remaining 90% into 80% train / 10% val (1/9 of 90% = 10%)
train_val = tmp["train"].train_test_split(test_size=1/9, seed=42, stratify_by_column="category")
```

**Final sizes: 160,678 train / 20,085 val / 20,085 test.**

### TextVectorization

A `tf_keras.layers.TextVectorization` layer is fitted on the training set only (preventing data leakage). Configuration: `max_tokens=20,000`, `output_mode="int"`, `output_sequence_length=128`. The 128-token cap covers >99.9% of samples per analysis done in Milestone 1. Vocabulary size reached the 20,000 cap.

```python
vectorizer = layers.TextVectorization(max_tokens=20_000, output_mode="int", output_sequence_length=128)
vectorizer.adapt(train_texts)  # fit on train only
```

### tf.data Pipelines

All three splits are wrapped in `tf.data.Dataset` pipelines:

```python
def make_dataset(texts, labels, shuffle=False):
    d = tf.data.Dataset.from_tensor_slices((texts, labels))
    if shuffle:
        d = d.shuffle(len(texts), seed=SEED)
    d = d.batch(BATCH_SIZE)
    d = d.map(lambda x, y: (vectorizer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return d.prefetch(tf.data.AUTOTUNE)
```

Vectorization happens inside the pipeline via `.map()` so it runs on GPU when available. `AUTOTUNE` for both parallelism and prefetching eliminates I/O bottlenecks.

### Class Weighting

The dataset has a 32.6× imbalance (POLITICS: ~32,739 samples vs. EDUCATION: ~1,004 samples). Balanced class weights are computed to counteract this:

```python
cw_array = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=train_labels)
class_weight = dict(enumerate(cw_array))
```

**Actual weight range: 0.150× (POLITICS) to 4.874× (EDUCATION).** The `class_weight` dict is passed to every `.fit()` call so the loss function penalizes errors on minority classes proportionally more.

---

## Problem 2 — Baseline Model (Embedding + GAP)

### Architecture

```
Embedding(20000, 64, mask_zero=True)   → 1,280,000 params
→ GlobalAveragePooling1D
→ Dense(256, relu)                     →    16,640 params
→ Dropout(0.3)
→ Dense(41, softmax)                   →    10,537 params
─────────────────────────────────────────────────────────
Total: 1,307,177 params (4.99 MB)
```

This is a **bag-of-embeddings** classifier. Each token is mapped to a 64-dimensional vector; `GlobalAveragePooling1D` averages across all token positions, collapsing the sequence into a single fixed-size vector. This discards word order but is extremely fast.

**`mask_zero=True`** on the Embedding layer tells downstream layers to ignore the zero-padding added by `output_sequence_length=128`. Without this, padding tokens contribute equally to the pooled representation.

### Training

- **Optimizer:** Adam with `lr=1e-3`
- **Loss:** `sparse_categorical_crossentropy` (accepts integer labels directly, no one-hot encoding needed)
- **Epochs:** up to 30 with `EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True)`
- **class_weight** passed to counteract imbalance

### Results

| Metric | Value |
|---|---|
| Best epoch | 5 (of 10 trained) |
| Training time | 1.3 min |
| Val accuracy | 0.5230 |
| Test accuracy | **0.5249** |
| Val macro-F1 | 0.4449 |
| Test macro-F1 | **0.4487** |

Val accuracy climbed from 0.4345 (epoch 1) to 0.5230 (epoch 5) then plateaued. Early stopping triggered at epoch 10 and restored epoch-5 weights.

### Limitations

`GlobalAveragePooling1D` destroys sequential structure. The phrase "not guilty" is treated identically to "guilty not" — both average to the same pooled vector. This is a fundamental ceiling for this architecture on tasks where phrase-level semantics matter.

---

## Problem 3 — Custom Model (Bidirectional LSTM)

### Architecture

```
Embedding(20000, 128, mask_zero=True)  → 2,560,000 params
→ Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))  → 263,168 params
→ Bidirectional(LSTM(64, dropout=0.2))                          → 164,352 params
→ Dense(256, relu)                                              →  33,024 params
→ BatchNormalization                                            →   1,024 params
→ Dropout(0.4)
→ Dense(41, softmax)                                            →  10,537 params
────────────────────────────────────────────────────────────────────────────────
Total: 3,032,105 params (11.57 MB), of which 512 are non-trainable (BN running stats)
```

### Key Design Choices

**Bidirectional LSTM:** Each LSTM reads the sequence in both forward and backward directions. The forward pass captures left-to-right context (e.g., "trump said..."), the backward pass captures right-to-left context (e.g., "...the president trump"). Concatenating both gives the model a wider view of local context around each token.

**Stacked BiLSTMs:** The first BiLSTM outputs a full sequence (`return_sequences=True`), allowing the second BiLSTM to build higher-level abstractions over the first layer's representations. This is analogous to stacking convolutional layers in a CNN.

**Recurrent dropout vs. regular dropout:** `dropout=0.2` inside the LSTM applies dropout to the recurrent connections (not just the input), which is more effective for regularizing recurrent networks.

**BatchNormalization:** Placed after the Dense layer (before Dropout) to normalize the activations, stabilizing training when the embedding dimension is larger (128 vs. 64 in the baseline).

### Training

- **Optimizer:** Adam with `lr=5e-4` (lower than baseline to prevent oscillation with recurrent gradients)
- **Epochs:** up to 30, `EarlyStopping(patience=5, monitor="val_loss")`

### Results

| Metric | Value | vs. Baseline |
|---|---|---|
| Best epoch | 2 (of 7 trained) | — |
| Training time | 5.3 min | +4.0 min |
| Val accuracy | 0.5196 | −0.0034 |
| Test accuracy | **0.5206** | **−0.0043** |
| Val macro-F1 | 0.4647 | +0.0198 |
| Test macro-F1 | **0.4626** | **+0.0139** |

**Notable training dynamics:** val_loss and val_accuracy diverged significantly. Val loss was lowest at epoch 2 (1.7317), but val accuracy continued improving through epoch 6 (0.5593). Because early stopping monitored `val_loss`, epoch-2 weights were restored — yielding better-calibrated probability distributions but lower raw accuracy than epoch 6 would have produced. This split means the BiLSTM **improved macro-F1 over the baseline (+0.0139) but slightly reduced test accuracy (−0.0043)**, reflecting better minority-class coverage at the cost of some majority-class confidence.

---

## Problem 4 — Pretrained Model (DistilBERT Transfer Learning)

### Model Selection

**DistilBERT-base-uncased** (`distilbert-base-uncased`) was chosen over full BERT because it retains ~97% of BERT's NLU performance with 40% fewer parameters and 60% faster inference. This makes it practical to fine-tune on a single consumer GPU.

DistilBERT is a transformer encoder pretrained on English Wikipedia and BookCorpus using masked language modeling (MLM) and knowledge distillation from BERT-base. It has 6 transformer layers, 12 attention heads, and 768-dimensional hidden states.

### Tokenization

DistilBERT requires its own WordPiece tokenizer rather than the simple whitespace tokenizer used in Problems 2–3:

```python
bert_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
enc = bert_tokenizer(texts, max_length=128, truncation=True, padding="max_length", return_tensors="np")
# Returns: enc["input_ids"], enc["attention_mask"]
```

The `attention_mask` tells the model which positions are real tokens (1) vs. padding (0), replacing the `mask_zero` mechanism used in the Keras Embedding layer. Tokenization of all three splits (160,678 + 20,085 + 20,085 sequences) completed in **8.9 seconds**.

### Architecture

```
input_ids  (None, 128) ─┐
                         ├→ TFDistilBertModel → last_hidden_state (None, 128, 768)  → 66,362,880 params
attention_mask (None, 128)┘
→ GlobalAveragePooling1D  → (None, 768)
→ Dropout(0.3)
→ Dense(41, softmax)      → (None, 41)                                              →     31,529 params
────────────────────────────────────────────────────────────────────────────────────────────────────────
Total: 66,394,409 params (253.27 MB)
Phase 1 trainable: 31,529 (head only) | Phase 2 trainable: 66,394,409 (full)
```

### Two-Phase Fine-Tuning

**Phase 1 — Head only (frozen base):**

```python
distilbert_base.trainable = False
# Compile with lr=1e-3, train for up to 5 epochs
```

The DistilBERT base is frozen and only the classification head (`GAP → Dropout → Dense(41)`) is trained. This is done first to initialize the head to reasonable weights before the base is unlocked. Without this warm-up, the randomly-initialized head would produce large gradients that corrupt the pretrained representations.

**Phase 2 — Full fine-tune:**

```python
distilbert_base.trainable = True
# Recompile with lr=2e-5, train for up to 5 epochs
```

The full model is then fine-tuned end-to-end at a very small learning rate (`2e-5`). The small LR is critical — it prevents *catastrophic forgetting*, where gradient updates erase the general language knowledge encoded in the pretrained weights.

`last_hidden_state` gives the per-token contextual embeddings from the final transformer layer. `GlobalAveragePooling1D` reduces them to a single sentence embedding, which the classification head then maps to class probabilities.

### Results

| Metric | Value | vs. Baseline | vs. BiLSTM |
|---|---|---|---|
| Training time | **135.2 min** | +133.9 min | +129.9 min |
| Val accuracy | 0.6476 | +0.1246 | +0.1280 |
| Test accuracy | **0.6457** | **+0.1208** | **+0.1251** |
| Val macro-F1 | 0.5848 | +0.1399 | +0.1201 |
| Test macro-F1 | **0.5824** | **+0.1337** | **+0.1198** |

Batch size was reduced to 32 (vs. 256 for Problems 2–3) to fit the transformer's memory footprint within 12 GB VRAM, resulting in 5,022 steps per epoch vs. 628 for the RNN models.

### Computational Cost

DistilBERT's self-attention mechanism scales as O(n²) in sequence length, making it significantly more expensive per token than the LSTM's O(n). At ~95 ms/step and 5,022 steps per epoch, each epoch takes approximately 8 minutes. Total training time of 135.2 min reflects both phases on RTX 3060. On CPU this run would take 12–24 hours.

---

## Problem 5 — Comparative Evaluation

### Summary Table

| Model | Params | Test Acc | Test Macro-F1 | Train Time |
|---|---|---|---|---|
| Baseline (Embedding+GAP) | 1,307,177 | 0.5249 | 0.4487 | 1.3 min |
| Custom (BiLSTM) | 3,032,105 | 0.5206 | 0.4626 | 5.3 min |
| Pretrained (DistilBERT) | 66,394,409 | **0.6457** | **0.5824** | 135.2 min |

### Ordering Anomaly: Accuracy vs. Macro-F1

The results reveal a split ordering across metrics:
- **Macro-F1:** DistilBERT (0.5824) > BiLSTM (0.4626) > Baseline (0.4487) — consistent with architectural progression
- **Test accuracy:** DistilBERT (0.6457) > **Baseline (0.5249) > BiLSTM (0.5206)** — BiLSTM fell below baseline

The BiLSTM's lower accuracy is explained by early-stopping behavior: the val_loss minimum at epoch 2 produced conservative, better-calibrated predictions that helped minority classes (hence higher F1) but were slightly less decisive on the majority classes that dominate accuracy. This confirms macro-F1 as the more meaningful metric for this imbalanced dataset.

### Macro-F1 vs. Accuracy

Accuracy is dominated by majority classes (POLITICS is 16.3% of the data). Macro-F1 gives equal weight to all 41 classes, so improvements in minority-class recall show up in F1 but not necessarily in accuracy. The narrowing gap between accuracy and macro-F1 across models (0.076 → 0.058 → 0.063) indicates DistilBERT distributes its gains more evenly across all 41 classes.

### Per-Class F1 — DistilBERT

**5 hardest classes:**

| Class | F1 |
|---|---|
| GOOD NEWS | 0.352 |
| IMPACT | 0.371 |
| EDUCATION | 0.383 |
| FIFTY | 0.405 |
| WORLD NEWS | 0.408 |

**5 easiest classes:**

| Class | F1 |
|---|---|
| DIVORCE | 0.795 |
| HOME & LIVING | 0.826 |
| TRAVEL | 0.832 |
| WEDDINGS | 0.859 |
| STYLE & BEAUTY | 0.868 |

The hardest classes share two characteristics: either very broad, topic-agnostic framing (GOOD NEWS covers any positive story across all domains; IMPACT is a meta-category for socially significant articles) or overlap with adjacent classes (WORLD NEWS vs. WORLDPOST). FIFTY is difficult because its human-interest headlines share vocabulary with PARENTING, HOME & LIVING, and WELLNESS. The easiest classes have highly domain-specific vocabulary — wedding, travel, and beauty articles use terminology that rarely appears in other categories.

### Sample Misclassifications (DistilBERT)

| True label | Predicted | Example headline |
|---|---|---|
| ENTERTAINMENT | COMEDY | "watch the funniest scene in 'top five'" |
| MEDIA | CRIME | "david hogg calls on media to stop naming santa fe school shooter" |
| POLITICS | WORLDPOST | "obama and india: love in the time of cholera" |
| TASTE | BUSINESS | "here's what you could buy if you quit your crazy starbucks habit" |
| PARENTING | PARENTS | "6 qualities kids need to succeed -- and one they don't" |

These errors reflect genuine label ambiguity in the dataset rather than model failure. PARENTING vs. PARENTS and ENTERTAINMENT vs. COMEDY are near-synonymous categories that would likely be merged in a cleaner taxonomy.

---

## Engineering Notes

### Keras 2 / Keras 3 Version Conflict

**Root cause:** TensorFlow 2.16+ ships with Keras 3 as its default backend. The `transformers` library (4.x) internally imports `tf_keras` (the legacy Keras 2 package) via `from tf_keras import ...`. This creates a class hierarchy split:

- `keras.Sequential` (imported via `from tensorflow import keras`) → resolves to `tf_keras.Sequential` after transformers import
- `layers.Embedding` (imported via `from tensorflow.keras import layers`) → resolves to `keras.src.layers.Embedding` (Keras 3)

These two hierarchies are incompatible, causing:

```
TypeError: The added layer must be an instance of class Layer.
Received: layer=<Embedding> of type <class 'keras.src.layers.core.embedding.Embedding'>
```

**Fix:** Import everything explicitly from `tf_keras`, bypassing `tensorflow.keras` entirely:

```python
import tf_keras as keras
from tf_keras import layers
```

This ensures `keras.Sequential` and all `layers.*` classes come from the same `tf_keras` package, regardless of what TF or transformers have imported internally.

**Why `TF_USE_LEGACY_KERAS=1` fails in Jupyter:** This environment variable must be set before the TensorFlow C extension loads. In a running Jupyter kernel, TF is often already imported (or cached in `sys.modules`) before the cell runs, so setting the env var in a cell has no effect. The explicit `import tf_keras` approach works because it bypasses the env var mechanism entirely.

### transformers Version Pin

`transformers==4.47.1` is pinned explicitly because:
- `transformers` 5.x dropped all TensorFlow model classes (`TFDistilBertModel`, `TFBertModel`, etc.)
- The last version with full TF support is 4.47.1

Both `tf_keras` and `transformers==4.47.1` are auto-installed at the top of the notebook:

```python
for pkg in ["tf_keras", "transformers==4.47.1"]:
    subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=True)
```

### GPU Setup (WSL2 + RTX 3060)

TensorFlow 2.20 requires CUDA 12.5. The system had CUDA 13.1 installed, so the CUDA runtime libraries needed to come from pip packages instead:

```
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 ...
```

These are added to `LD_LIBRARY_PATH` in `.bashrc` along with the WSL2 stub library path:

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/path/to/nvidia/cuda_runtime/lib:...:$LD_LIBRARY_PATH
```

The `/usr/lib/wsl/lib/` directory contains stub `.so` files provided by the WSL2 NVIDIA kernel driver. If the Windows GPU driver is updated without restarting WSL (`wsl --shutdown` in PowerShell), these stubs become stale and `cuInit()` returns `CUDA_ERROR_NO_DEVICE`. The fix is to restart WSL to refresh the stubs. GPU was confirmed as `/physical_device:GPU:0` (NVIDIA GeForce RTX 3060, 9709 MB allocated).

### pandas `.map()` vs. `.applymap()`

`DataFrame.applymap()` was removed in pandas 3.0. The equivalent for element-wise operations on DataFrames is now `.map()`:

```python
# pandas 3.x:
results_df[float_cols] = results_df[float_cols].map(lambda x: f"{x:.4f}")
```

### classification_report with Missing Classes

When predicting 41 classes on a small test set, some minority classes may have zero predictions. Passing `labels=np.arange(NUM_CLASSES)` forces all 41 classes to appear in the report; `zero_division=0` suppresses the warning for classes with no true or predicted samples:

```python
report_dict = classification_report(
    test_labels, test_preds_p,
    labels=np.arange(NUM_CLASSES),
    target_names=label_names,
    output_dict=True,
    zero_division=0,
)
```
