Author Style Change Detection

### Paragraph-Level Author Change Detection Using LUAR and SBERT

This repository contains the implementation and experimental results of my Master's thesis in **Information and Communication Systems Engineering**.

The project investigates **paragraph-level author change detection**: given a document consisting of multiple paragraphs, the system identifies **where a change of author occurs** between consecutive paragraphs.

Rather than identifying the authors themselves, the task focuses on detecting changes in the writing characteristics of the text.

The experiments use **PAN-CLEF authorship analysis datasets from PAN21 through PAN24** and investigate the performance of **LUAR** and **SBERT** representations under different training and classification strategies.

---

## Project Overview

The thesis investigates three main factors:

* **Representation model:** LUAR vs. SBERT
* **Training strategy:** frozen vs. fine-tuned representations
* **Decision mechanism:** cosine similarity vs. concatenation with a classification head

This results in eight evaluated model configurations:

| Model | Training   | Decision Mechanism                  |
| ----- | ---------- | ----------------------------------- |
| SBERT | Frozen     | Concatenation + Classification Head |
| SBERT | Frozen     | Cosine Similarity                   |
| SBERT | Fine-tuned | Concatenation + Classification Head |
| SBERT | Fine-tuned | Cosine Similarity                   |
| LUAR  | Frozen     | Concatenation + Classification Head |
| LUAR  | Frozen     | Cosine Similarity                   |
| LUAR  | Fine-tuned | Concatenation + Classification Head |
| LUAR  | Fine-tuned | Cosine Similarity                   |

The primary evaluation metric is **F1-score**, while **Accuracy, Precision, and Recall** are also reported.

---

## Datasets

The experiments are based on the following PAN datasets:

* **PAN21**
* **PAN22**
* **PAN23**
* **PAN24**

The repository contains the dataset implementations required to load these datasets.

The actual datasets are **not included** in the repository.

Dataset files should therefore be obtained separately and placed in the appropriate local data directory.

---

## Methodology

Each problem consists of a sequence of paragraphs. The model processes the paragraphs and predicts whether the author changes between consecutive paragraphs.

The target labels are:

```text
0 → Same author
1 → Author change
```

Padding positions are represented using `-100` and are ignored during training/evaluation where applicable.

The experiments are performed at the **problem/episode level**, preserving the complete sequence of paragraphs belonging to each problem.

---

## Models

### LUAR

**Learning Universal Authorship Representations (LUAR)** provides representations designed to capture authorship-related characteristics across different domains.

The repository contains LUAR implementations for:

* Frozen representations
* Fine-tuned representations
* Cosine-similarity-based detection
* Concatenation followed by a classification head

### SBERT

**Sentence-BERT (SBERT)** is used as a second representation model for comparison.

The repository contains SBERT implementations for:

* Frozen representations
* Fine-tuned representations
* Cosine-similarity-based detection
* Concatenation followed by a classification head

---

## Repository Structure

The project is organized into several main components:

> **The complete visual representation of the repository structure will be added here.**

### `src/datasets/`

Contains the dataset implementations used in the thesis experiments:

* `pan21_dataset.py`
* `pan22_dataset.py`
* `pan23_dataset.py`
* `pan24_dataset.py`
* `multidomain_dataset.py`
* `utils.py`

### `src/models/`

Contains the model, transformer, training, attention, and classification implementations.

The directory includes the implementations for all LUAR and SBERT configurations evaluated in the thesis.

### `src/evaluation/`

Contains the evaluation helper and the final experimental results:

* `results_luar.xlsx`
* `results_sbert.xlsx`
* `results_total.xlsx`

### `src/utilities/`

Contains supporting utilities used by the training and evaluation pipeline.

### Main scripts

* `main.py` — main entry point for training/testing the different model configurations
* `arguments.py` — command-line and experiment configuration
* `evaluator.py` — evaluation of generated solution files
* `fabricator.py` — auxiliary solution-file processing

---

## Evaluation

The primary metric used throughout the thesis is the **F1-score**.

Additional metrics include:

* Accuracy
* Precision
* Recall

The evaluation is performed on the predicted author-change labels and compares them against the corresponding PAN ground-truth annotations.

The final experimental results are provided in:

```text
src/evaluation/results_luar.xlsx
src/evaluation/results_sbert.xlsx
src/evaluation/results_total.xlsx
```

These files contain the results obtained during the experiments conducted for the thesis.

---

## Experimental Analysis

The experiments investigate the impact of:

### LUAR vs. SBERT

The two representation approaches are compared to determine how effectively they capture information relevant to author changes.

### Frozen vs. Fine-Tuned Representations

The experiments compare using pretrained representations without modification against allowing the representation model to adapt to the author-change detection task.

### Cosine Similarity vs. Classification Head

Two different approaches are evaluated for determining whether an author transition has occurred:

1. **Cosine similarity** between paragraph representations.
2. **Concatenation of representations followed by a classification head.**

This allows the thesis to examine both similarity-based and learned classification approaches.

---

## Results

The experimental results are included in the repository as Excel spreadsheets.

The main conclusions of the experiments indicate that model performance depends on the characteristics of the text and the type of information required to distinguish between authors.

In particular, the experiments suggest that:

* **LUAR can provide an advantage when stylistic information is particularly important.**
* **SBERT can perform well when semantic information contributes strongly to distinguishing the paragraphs.**
* **Fine-tuning can improve the representations for the specific author-change detection task compared with keeping them frozen.**
* The effectiveness of **cosine similarity versus a learned classification head** depends on the representation model and experimental setting.

Detailed results can be found in the Excel files under `src/evaluation/`.

---

## Repository Structure

The repository is organized around the complete experimental pipeline, from dataset loading and representation generation to model training and final evaluation.

### Project Tree

```text
LUAR-classhead/
│
├── src/
│   │
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── multidomain_dataset.py
│   │   ├── pan21_dataset.py
│   │   ├── pan22_dataset.py
│   │   ├── pan23_dataset.py
│   │   ├── pan24_dataset.py
│   │   └── utils.py
│   │
│   ├── models/
│   │   ├── layers.py
│   │   ├── lightning_trainer.py
│   │   ├── trainer_luar_concat.py
│   │   ├── trainer_luar_cosine.py
│   │   ├── trainer_sbert_concat.py
│   │   ├── trainer_sbert_cosine.py
│   │   ├── transformer.py
│   │   ├── transformer_luar_concat.py
│   │   ├── transformer_luar_fn_cosine.py
│   │   ├── transformer_luar_frozen_concat.py
│   │   ├── transformer_luar_frozen_cosine.py
│   │   ├── transformer_sbert_concat.py
│   │   ├── transformer_sbert_cosine.py
│   │   ├── transformer_sbert_finetuned_concat.py
│   │   └── transformer_sbert_finetuned_cosine.py
│   │
│   ├── evaluation/
│   │   ├── helper.py
│   │   ├── results_luar.xlsx
│   │   ├── results_sbert.xlsx
│   │   └── results_total.xlsx
│   │
│   ├── utilities/
│   │   ├── decorators.py
│   │   ├── file_utils.py
│   │   └── metric.py
│   │
│   ├── arguments.py
│   ├── evaluator.py
│   ├── fabricator.py
│   └── main.py
│
├── presentation.pdf
├── README.md
├── LUAR_README.md
├── requirements.txt
├── LICENSE
└── NOTICE
```

### Component Interaction

The main experimental workflow can be summarized as:

```text
                    ┌──────────────────────┐
                    │      PAN21–PAN24     │
                    │       Datasets       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Dataset Loaders    │
                    │   src/datasets/      │
                    │                      │
                    │ PAN21 / PAN22 /      │
                    │ PAN23 / PAN24        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    src/main.py       │
                    │                      │
                    │ Experiment selection │
                    │ & configuration      │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
       ┌──────────────────┐        ┌──────────────────┐
       │      SBERT       │        │       LUAR       │
       │ Representation   │        │ Representation   │
       └────────┬─────────┘        └────────┬─────────┘
                │                           │
                └─────────────┬─────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
      ┌───────────────────┐       ┌────────────────────┐
      │ Cosine Similarity │       │ Concatenation +    │
      │                   │       │ Classification     │
      │                   │       │ Head               │
      └─────────┬─────────┘       └──────────┬─────────┘
                │                            │
                └─────────────┬──────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │   Training / Testing │
                    │    src/models/       │
                    │                      │
                    │ Frozen / Fine-tuned  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Author Change       │
                    │    Predictions       │
                    │                      │
                    │ 0 = Same Author      │
                    │ 1 = Author Change    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    src/evaluator.py  │
                    │                      │
                    │ F1 / Accuracy /      │
                    │ Precision / Recall   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Experimental       │
                    │      Results         │
                    │                      │
                    │ results_luar.xlsx    │
                    │ results_sbert.xlsx   │
                    │ results_total.xlsx   │
                    └──────────────────────┘
```

### How the Components Interact

The main execution flow is centered around `src/main.py`.

1. **Dataset loading**
   `src/datasets/` provides the PAN21–PAN24 dataset implementations. `utils.py` selects the appropriate dataset implementation based on the experiment configuration.

2. **Experiment configuration**
   `arguments.py` defines the available experiment parameters and model configuration. `main.py` uses these parameters to select the required model variant and training configuration.

3. **Representation generation**
   The selected **SBERT** or **LUAR** transformer implementation generates representations for the paragraphs belonging to each problem.

4. **Author-change detection**
   The representations are processed using one of two approaches:

   * **Cosine similarity**, which compares the representations of consecutive paragraphs.
   * **Concatenation + classification head**, which combines the representations and learns a classifier for the author-change decision.

5. **Training**
   The trainer implementations in `src/models/` handle the training and testing process, including the frozen and fine-tuned configurations.

6. **Prediction and evaluation**
   The trained model produces a sequence of author-change predictions. `evaluator.py` compares these predictions with the corresponding PAN ground truth and calculates the evaluation metrics.

7. **Results**
   The final experimental results are organized in the Excel files under `src/evaluation/`.

### Supporting Components

The remaining source files provide functionality used throughout the pipeline:

* `src/models/layers.py` — neural-network and attention-related layers.
* `src/models/transformer.py` — shared transformer functionality.
* `src/models/lightning_trainer.py` — common Lightning training functionality.
* `src/utilities/` — supporting utilities for metrics, file handling, and decorators.
* `src/fabricator.py` — auxiliary processing of generated solution files.
* `src/evaluation/helper.py` — supporting functionality for result evaluation and analysis.


## Reproducibility

The repository contains the source code, evaluation utilities, configuration, and experimental results used for the thesis.

The following files and directories are intentionally **not included**:

* PAN datasets
* Pretrained model weights
* Training checkpoints
* Generated model outputs
* Logs
* Local virtual environments
* Intermediate evaluation files
* Unrelated datasets and implementations from the original LUAR framework

These files are excluded either because of their size, licensing/distribution restrictions, or because they are not required as part of the final thesis source repository.

The Python environment used during development included **Python 3.10.11**.

The corresponding package versions are preserved in:

```text
requirements.txt
```

---

## Thesis Presentation

The presentation accompanying the Master's thesis is included in the repository:

**[View the Thesis Presentation](presentation.pdf)**

---

## Original LUAR Project

This project builds upon the original **Learning Universal Authorship Representations (LUAR)** framework.

The original LUAR repository documentation has been preserved separately as:

```text
LUAR_README.md
```

This file contains the original project's documentation, installation information, acknowledgements, citation, and licensing information.

The original LUAR paper is:

> Rafael A. Rivera Soto, Olivia Miano, Juanita Ordonez, Barry Chen, Aleem Khan, Marcus Bishop, and Nicholas Andrews.
> **Learning Universal Authorship Representations.**
> EMNLP 2021.

The original LUAR work should be cited when using the LUAR framework or representations.

---

## Citation

If you use this repository, the associated Master's thesis should be cited accordingly.

For the underlying LUAR work:

```bibtex
@inproceedings{uar-emnlp2021,
  author    = {Rafael A. Rivera Soto and Olivia Miano and Juanita Ordonez and Barry Chen and Aleem Khan and Marcus Bishop and Nicholas Andrews},
  title     = {Learning Universal Authorship Representations},
  booktitle = {EMNLP},
  year      = {2021},
}
```

---

## License and Attribution

The project retains the licensing and attribution information of the original LUAR codebase.

For the complete licensing information, see:

* [`LICENSE`](LICENSE)
* [`NOTICE`](NOTICE)
* [`LUAR_README.md`](LUAR_README.md)

**SPDX-License-Identifier: Apache-2.0**

---

## Acknowledgements

This project builds upon the work of the authors of the original LUAR framework and the broader research community contributing to authorship analysis and the PAN evaluation campaigns.

The original LUAR documentation and attribution information are preserved in [`LUAR_README.md`](LUAR_README.md).

---

## Author

**Nikos Karagiannis**

Master's Degree in **Information and Communication Systems Engineering**

---

## Repository Contents

The repository provides:

* Thesis source code
* PAN21–PAN24 dataset implementations
* LUAR model implementations
* SBERT model implementations
* Training and evaluation utilities
* Experimental results
* Master's thesis presentation
* Original LUAR documentation and licensing information

The repository is intended to provide a clean and focused representation of the implementation and experimental work carried out as part of the Master's thesis.
