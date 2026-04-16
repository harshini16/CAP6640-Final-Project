# Evaluating the Effect of Prompting Strategies on Hallucinations in Grounded Question Answering

**Harshini Manimaran — University of Central Florida | CAP6640 Natural Language Processing**

---

## Introduction

Large Language Models are known to generate responses that may be confident yet false. This study seeks to investigate how **prompting strategies** influence the occurrence and form of such hallucinations when the model is tasked with answering questions grounded in a reference context passage.

**Four prompting strategies** are applied on the **FLAN-T5-large model**, evaluated with 100 samples from the SQuAD dataset (400 answers in total). A novel **NLI-based automated annotation pipeline** for classifying the types of hallucinations is developed and validated against human judgments using the DeBERTa-v3 model.

---

## Research Question

> *How do different prompting strategies affect the rate and form of hallucinations in grounded question answering tasks?*

---

## Prompting Strategies Evaluated

| Strategy | Description |
|----------|-------------|
| **Zero-shot** | The model is prompted to answer a question based solely on the context passage |
| **Few-shot** | Two examples of question-answer pairs precede the target question |
| **Chain-of-Thought (CoT)** | The model is instructed to reason step by step before providing an answer |
| **Constrained** | The model is directed to answer based only on the context passage, or reject by saying *"Not found"* |

---

## Key Findings

| Strategy | Exact Match (%) | F1 (%) | Hallucination Rate (%) | Refusal Rate (%) |
|----------|:-:|:-:|:-:|:-:|
| Zero-shot | 82.0 | 65.35 | 17.0 | 0.0 |
| Few-shot | 82.0 | 61.49 | 17.0 | 0.0 |
| Chain-of-Thought | 83.0 | 31.01 | 13.0 | 0.0 |
| Constrained | 76.0 | 60.48 | 14.0 | 10.0 |

**Main takeaways:**
- **CoT yields the highest exact match (83%)** but the lowest F1 (31.01%) as the long-winded answers reduce token overlap with concise gold spans
- **93.4% of hallucinations are extrinsic** — the model rarely contradicts the context but often adds information that is not supported by the passage
- **The constrained strategy presents a trade-off between reliability and coverage:** high accuracy but with a 10% abstention rate
- The zero-shot prompting strategy offers the best overall performance

---

## Automated Annotation Pipeline

Manual annotation of all 400 generated answers would be tedious. An automated NLI-based pipeline is built with `cross-encoder/nli-deberta-v3-base`:

```
1. Refusal phrase detected? -> Label R (refusal)
2. Exact match with gold answer? -> Label C (correct)
3. NLI classification:
     contradiction -> H / intrinsic hallucination
     neutral -> H / extrinsic hallucination
     entailment -> E (extraction error)
```

**Validation with human annotations** on 40 stratified samples:
- Raw agreement: **85%**
- Cohen's κ: **0.407** (moderate)
- NLI obtains perfect hallucination recall (1.00) and low hallucination precision (0.17) — suitable as an upper-bound estimator

---

## Project Structure

```
├── hal_v3.ipynb              # Main notebook — inference, annotation, analysis, plots
├── full_results.csv          # All 400 annotated outputs
├── summary_table.csv         # Accuracy, F1, hallucination, refusal rates by strategy
├── hallucination_type_table.csv # Intrinsic vs extrinsic hallucinations
├── error_analysis_table.csv  # Number of errors per type per strategy
├── validation_sample.csv     # 40-output human validation sample
├── validation_results.csv    # Comparison of NLI vs human labels, Cohen's κ
├── fig1_accuracy.png         # Accuracy bar charts (Exact Match & F1)
├── fig2_error_profile.png    # Stacked error profile bar chart
├── fig3_hall_type.png        # Intrinsic vs extrinsic hallucinations
├── fig4_nli_confidence.png   # Confidence distributions for NLI
├── fig5_f1_dist.png          # Box plots of F1 scores
├── fig6_len_f1.png           # Scatter plot of length vs F1 score
└── pipeline.png              # Full end-to-end evaluation pipeline
```

---

## Running the Code

### 1. Upload to Google Colab
Upload `NLP_hallucinationLLMs.ipynb` to [Google Colab](https://colab.research.google.com).

### 2. Configure Runtime
Navigate to **Runtime → Change runtime type** and choose:
- Accelerator: **GPU T4**
- High RAM: **On**

### 3. Execute cells sequentially
| Section | Description |
|---------|-------------|
| 1-2 | Installation, loading the SQuAD dataset |
| 3-5 | Loading FLAN-T5-large model, defining prompt generation function |
| 6 | Generating 400 answers (8-12 min on T4 GPU) |
| 7 | Computing Exact Match and token F1 score |
| 8 | Releasing VRAM before loading NLI model |
| 9-10 | Loading DeBERTa-v3, generating annotations with NLI pipeline |
| 11 | Manual annotation validation — enter 40 labels, calculate Cohen's κ |
| 12-14 | Tabulating results and analyzing errors |
| 15 | Creating visualizations |
| 16-17 | Examining qualitative examples, exporting all data to CSV |

### 4. Manual Annotation Step (Section 11)
After executing the first cell in Section 11, examine the 40 answers directly in the notebook and assign each a label in the `human_labels` dict:
```python
human_labels = {
    0: "C",  # Correct
    1: "H",  # Hallucination
    2: "R",  # Refusal
    3: "E",  # Extraction error
    #... all 40 samples
}
```

---

## Dependencies

```
transformers
datasets
torch
scikit-learn
matplotlib
seaborn
pandas
```

Installation:
```bash
pip install transformers datasets torch scikit-learn matplotlib seaborn pandas
```

---

## Models

| Model Name | Usage |
|------------|-------|
| google/flan-t5-large | Generation of 400 outputs with different prompts |
| cross-encoder/nli-deberta-v3-base | Automatic annotation of hallucinations with NLI |

---

## Dataset

**SQuAD** (Stanford Question Answering Dataset) by Rajpurkar et al., 2016  
100 examples from the training set, loaded with HuggingFace `datasets`.

---

## Acknowledgments

If you adopt this research, please include the following citation:

```
@article{manimaran2025hallucination,
  title={Evaluating the Impact of Prompting Strategies on Hallucinations in Grounded Question Answering},
  author={Manimaran, Harshini},
  institution={University of Central Florida},
  year={2025}
}
```

---

## Licensing

This project was completed as part of CAP6640 coursework at the University of Central Florida. Academic use only.
