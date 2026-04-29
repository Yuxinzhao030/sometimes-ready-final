# Fake News Detection Using Machine Learning

## Introduction

The rapid spread of misinformation online has made fake news detection an increasingly important problem in machine learning and natural language processing. False or misleading news articles can influence public opinion, amplify social division, and undermine trust in legitimate media sources. As digital platforms continue to become primary channels for news consumption, scalable and automated methods for identifying fake news are becoming increasingly essential.

In this project, we investigate whether machine learning models can accurately distinguish between **real news** and **fake news** based solely on article text. We formulate this as a **binary classification problem**, where each article is labeled as either:

- **0 = Real News**
- **1 = Fake News**

Our goal is to compare multiple machine learning approaches for fake news detection and evaluate their predictive performance, robustness, and practical strengths and limitations.

To accomplish this, we explore a range of modeling strategies, including traditional supervised classification models, transformer-based language models, and anomaly detection methods. By comparing these approaches, we aim to better understand which techniques are most effective for automated fake news detection.

## Dataset

For this project, we use the **WELFake Dataset**, a publicly available dataset for fake news classification.

### Source

- Dataset: WELFake Dataset
- Platform: Kaggle
- Link: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
- Creator: Saurabh Shahane

The WELFake dataset was created by combining several existing fake news and real news datasets into a single large corpus for binary text classification. The goal of the dataset is to support research on automated fake news detection using machine learning and natural language processing techniques.

### Features

The original dataset contains the following columns:

| Column | Description |
|---|---|
| `title` | Headline of the news article |
| `text` | Main body text of the article |
| `label` | Binary label for classification (`0 = Real`, `1 = Fake`) |

For our modeling pipeline, we primarily use the **article text (`text`)** as the predictive feature and the **label** as the target variable.

### Data Size

The raw dataset contains approximately **72,000+ news articles**, making it sufficiently large for supervised learning models.

After preprocessing, our final dataset size is reduced due to:

- removal of missing values
- removal of duplicated articles
- removal of URLs and formatting artifacts
- filtering out extremely short articles
- text normalization and cleaning

This ensures higher data quality and reduces noise in downstream modeling.

### Data Collection Background and Limitations

Because WELFake is an aggregated dataset compiled from multiple news datasets, it contains writing styles, topics, and publication sources from a variety of outlets. This diversity is helpful for building generalized fake news detection models.

However, this also introduces several challenges:

- **Source bias**: certain publishers may have consistent writing patterns that models can overfit to
- **Labeling bias**: article labels depend on the original source datasets and may contain inconsistencies
- **Temporal drift**: writing styles and misinformation strategies evolve over time
- **Dataset artifacts**: formatting tokens, media references, and duplicated content may leak signals unrelated to factual accuracy

These limitations should be considered when interpreting model performance.

## TF-IDF-based Models

### Method

We use **TF-IDF vectorization** to convert cleaned news articles into numerical feature representations suitable for machine learning classification.

Compared to dense embedding methods such as SBERT, TF-IDF produces a sparse representation that emphasizes informative words and short phrases while reducing the influence of overly common terms.

Our analysis evaluates the following classifiers on top of TF-IDF features:

- Logistic Regression
- Multinomial Naive Bayes
- XGBoost

Each model is assessed using **Accuracy**, **Precision**, **Recall**, **F1-score**, and **ROC-AUC**.

## SBERT-based Models

### Method

We use Sentence-BERT (SBERT) to obtain dense semantic representations of news articles. Compared to traditional feature-based methods such as TF-IDF, SBERT captures contextual and semantic information at the sentence level.


We evaluate the following classifiers on top of SBERT embeddings:

- Logistic Regression  
- XGBoost  
- Gaussian Naive Bayes  

---

### Implementation

We use the pre-trained model: sentence-transformers/all-MiniLM-L6-v2 (available at: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 


Key settings:

- Embedding dimension: 384  
- Batch size: 32  

---

### Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **TF-IDF + Logistic** | **0.933** | **0.926** | **0.920** | **0.923** | **0.981** |
| **TF-IDF + MultinomialNB** | **0.864** | **0.831** | **0.863** | **0.847** | **0.938** |
| **SBERT + Logistic** | **0.860** | **0.858** | **0.815** | **0.836** | **0.933** |
| **SBERT + XGBoost** | **0.862** | **0.865** | **0.811** | **0.837** | **0.935** |
| **SBERT + GaussianNB** | **0.727** | **0.673** | **0.723** | **0.698** | **0.803** |

---

### Visualization

**ROC Curve**

![ROC](results/sbert_roc_curve.png)

**Confusion Matrices**

- SBERT + Logistic  
  ![Logistic CM](results/sbert_logistic_confusion.png)

- SBERT + XGBoost  
  ![XGBoost CM](results/sbert_xgboost_confusion.png)
  
- SBERT + GaussianNB  
  ![NB CM](results/sbert_nb_confusion.png)

---

### Analysis

SBERT-based models achieve strong performance, with Logistic Regression and XGBoost reaching accuracy around 86% and ROC-AUC above 0.93.

In contrast, Gaussian Naive Bayes performs substantially worse, with accuracy dropping to around 73%.

This gap can be explained by the underlying assumptions of the model:

- Gaussian Naive Bayes assumes **feature independence**  
- It also assumes each feature follows a **Gaussian distribution**

However, SBERT embeddings are:

- **high-dimensional**
- **dense and correlated across dimensions**
- **not Gaussian-distributed**

As a result, the assumptions of Naive Bayes are strongly violated, leading to degraded performance.

From the confusion matrix, GaussianNB produces:

- significantly more **false positives**
- and more **false negatives**

indicating weaker class separation compared to Logistic Regression and XGBoost.

Overall, this highlights that:

> **model assumptions must align with representation structure** and that not all classifiers are suitable for dense semantic embeddings.
---

### TBD

A comparison with TF-IDF-based models .

---

## Fine-tuned BERT Models

### Method

We fine-tune a transformer-based language model for fake news classification.

Unlike TF-IDF and SBERT pipelines that rely on fixed feature representations, fine-tuned BERT updates model parameters directly on the classification task, allowing the model to learn task-specific semantic patterns from the news text.

This approach captures contextual meaning, long-range dependencies, and subtle linguistic cues useful for distinguishing fake and real news.

---

### Implementation

We use the pre-trained model:

`distilbert-base-uncased`

Key settings:

- Epochs: 2
- Learning rate: 2e-5
- Max length: 256
- Batch size: 8
- Optimizer: AdamW (default HuggingFace Trainer)

---

### Results

| Model | Accuracy | Precision | Recall | F1 Score |
|------|----------|-----------|--------|----------|
| Fine-tuned DistilBERT | 0.989 | 0.992 | 0.984 | 0.988 |

Confusion Matrix:

- True Negative: 6850
- False Positive: 43
- False Negative: 86
- True Positive: 5241

---

### Analysis
Fine-tuned DistilBERT achieves the strongest performance among all tested models, reaching nearly 99% accuracy.

This suggests that end-to-end transformer fine-tuning substantially outperforms feature-based pipelines such as TF-IDF and SBERT on this dataset.

The result indicates that contextual language understanding and task-specific adaptation are highly effective for fake news detection.
---

## Appendix

### How to Reproduce the Results

#### 1. Install dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

#### 2. Download the dataset

Download the **WELFake Dataset** from Kaggle:

https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

After downloading, place the CSV file in:

```text
data/raw/WELFake_Dataset.csv
```

The project expects the following directory structure:

```text
project/
├── code/
│   ├── data_cleaning.py
│   ├── evaluate.py
│   ├── logistic_regression.py
│   ├── main.py
│   ├── naive_bayes.py
│   └── tfidf_features.py
│
├── data/
│   ├── raw/
│   │   └── WELFake_Dataset.csv
│   └── processed/
│
├── results/
│   ├── csv/
│   ├── figures/
│   └── models/
│
├── README.md
└── requirements.txt
```

#### 3. Run the full pipeline

```bash
python code/main.py
```

#### 3. Run the full pipeline

```bash
python code/main.py
```

This script will:

- clean and preprocess the raw dataset
- generate train/test splits
- create feature representations
- train machine learning models
- evaluate model performance
- generate visualizations
- save model outputs and summary results

#### Output

Results generated by `main.py` will be saved in:

```text
results/
├── csv/
├── figures/
└── models/
```

#### Reproducibility Note

We fix data splitting and model randomness using `random_state=42` where applicable to improve reproducibility.

However, slight numerical differences may still occur across different machines due to differences in operating systems, package versions, floating-point computation, hardware, and parallel computation in some model implementations.

These differences are expected to be very small and should not materially affect the overall conclusions.
