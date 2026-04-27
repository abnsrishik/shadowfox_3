# 🧠 Task 3: Language Model Exploration and Analysis

> Internship Task 3 — NLP & Machine Learning  
> A from-scratch interpolated trigram language model with full evaluation, visualization, and analysis.

---

## 📌 Overview

This project implements a **word-level interpolated trigram language model** entirely from scratch in Python — no external ML libraries, no API keys, no model downloads. The notebook explores core language modeling concepts including tokenization, smoothing, perplexity evaluation, text generation, and visualization.

The model is intentionally transparent and inspectable: every probability used for prediction can be printed and explained, making it ideal for understanding the mechanics behind language models before scaling to transformers like GPT or BERT.

---

## 📁 Repository Structure

```
├── Task_3_Language_Model_Analysis.ipynb   # Main notebook (self-contained)
└── README.md                              # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

```bash
pip install notebook
```

### Run the Notebook

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
jupyter notebook Task_3_Language_Model_Analysis.ipynb
```

> ✅ No additional packages required. The notebook uses only Python standard library modules (`math`, `random`, `re`, `collections`).

---

## 🔬 What's Inside the Notebook

| Section | Description |
|---|---|
| **1. Problem Statement** | Research questions and project goals |
| **2. LM Selection** | Why an interpolated trigram model was chosen over GPT/BERT |
| **3. Implementation** | From-scratch `NGramLanguageModel` and `InterpolatedTrigramLanguageModel` classes |
| **4. Evaluation** | Perplexity comparison across unigram, bigram, trigram, and interpolated models |
| **5. Exploration & Analysis** | Next-word predictions and multi-prompt generation comparison |
| **6. Visualization** | SVG bar charts and probability heatmap (no Matplotlib needed) |
| **7. Generation Metrics** | Distinct token ratio and repetition rate across models |
| **8. Findings** | Key takeaways from quantitative and qualitative analysis |
| **9. Ethics & Alignment** | Bias, privacy, reliability, and transparency considerations |
| **10. Conclusion** | Insights, limitations, and future improvement directions |

---

## 🧮 Model Architecture

The chosen model combines three n-gram orders using linear interpolation:

```
P(w | context) = 0.15 · P_unigram(w)
               + 0.30 · P_bigram(w | w-1)
               + 0.55 · P_trigram(w | w-2, w-1)
```

**Smoothing:** Add-alpha (Laplace) smoothing with α = 0.25 to handle unseen n-grams.  
**Generation:** Temperature-scaled top-k sampling with a fixed random seed for reproducibility.

### Why Not GPT / BERT?

| Factor | Transformer (GPT/BERT) | Interpolated Trigram *(chosen)* |
|---|---|---|
| Interpretability | Opaque internal weights | Every probability is inspectable |
| Setup | Large downloads / paid API | Zero external dependencies |
| Learning value | Hides core LM mechanics | Exposes tokenization, smoothing, perplexity |
| Compute | GPU recommended | Runs in seconds on any CPU |
| Reproducibility | Varies by model version | Fully deterministic with fixed seed |

---

## 📊 Sample Results

### Perplexity Comparison (lower = better)

| Model | Validation Perplexity |
|---|---:|
| Unigram baseline | 148.32 |
| Bigram baseline | 97.61 |
| Trigram only | 89.14 |
| **Interpolated trigram** | **76.83** |

### Sample Generation — Prompt: `"language models"`

| Model | Generated text |
|---|---|
| Unigram | *use evaluation to language data model that use* |
| Bigram | *can summarize lessons and generate practice questions* |
| **Interpolated trigram** | *can learn patterns from examples but the quality depends* |

---

## 🔍 Research Questions Addressed

1. Does adding context improve next-word prediction over a context-free baseline?
2. Does interpolation reduce the sparsity problem of a pure trigram model?
3. Can qualitative generation examples reveal limitations that perplexity alone cannot?
4. How can visualizations make language model behavior accessible to non-technical audiences?

---

## ⚠️ Limitations

- Trained on a small, NLP-domain corpus (~28 sentences) — generalizes poorly outside this domain
- No subword tokenization; out-of-vocabulary words are mapped to `<unk>`
- Context window capped at 2 previous words (trigram)
- No semantic understanding — the model captures co-occurrence patterns, not meaning

---

## 🔮 Future Improvements

- Train on a larger, more diverse corpus
- Add subword tokenization (BPE or WordPiece)
- Compare against a fine-tuned transformer baseline (GPT-2, DistilBERT)
- Evaluate with human ratings alongside perplexity
- Extend to domain-specific applications (education, healthcare, customer support)

---

## 🧭 Ethical Considerations

| Concern | Notes |
|---|---|
| **Bias** | Models trained on narrow data reproduce narrow language patterns |
| **Privacy** | Real deployments must avoid training on sensitive personal data without consent |
| **Reliability** | Generated text should not be used in high-stakes decisions without human review |
| **Transparency** | Users should know when content is machine-generated |

---

