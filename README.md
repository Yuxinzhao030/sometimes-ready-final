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

## Experiment Design

This project adopts a modular pipeline where each team member contributes an independent component. The overall workflow is:

```
Raw Data → Preprocessing & TF-IDF (Person 1)
                ↓
        ┌───────┴────────┐
   XGBoost (Person 2)   Logistic Regression (Person 1)
        │                        │
        └────────┬───────────────┘
                 ↓
          SBERT Embeddings (Person 3) → Ensemble Comparison
                 ↓
          Fine-tuned BERT (Person 4) → Final Evaluation
```

All models consume the same 80/20 train-test split (48,880 / 12,220 samples) produced by Person 1's preprocessing script, ensuring a fair and consistent comparison across methods.

---

## TF-IDF-based Models

### Method

We use **TF-IDF vectorization** to convert cleaned news articles into numerical feature representations suitable for machine learning classification.

Compared to dense embedding methods such as SBERT, TF-IDF produces a sparse representation that emphasizes informative words and short phrases while reducing the influence of overly common terms.

Our analysis evaluates the following classifiers on top of TF-IDF features:

- Logistic Regression
- Multinomial Naive Bayes
- XGBoost

Each model is assessed using **Accuracy**, **Precision**, **Recall**, **F1-score**, and **ROC-AUC**.

## XGBoost

### Implementation

We use gradient-boosted decision trees (XGBoost) on TF-IDF vectors as a strong non-linear baseline.

The input features are 5,000-dimensional TF-IDF vectors with unigrams and bigrams, constructed by Person 1's `build_tfidf()` function.

Key settings:

| Parameter | Value |
|-----------|-------|
| n_estimators | 200 |
| max_depth | 6 |
| learning_rate | 0.1 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| tree_method | hist |

The `hist` tree method was selected for memory efficiency, as it uses histogram-based splitting rather than exact greedy enumeration.

---

### Results

| Metric | Score |
|--------|-------|
| Accuracy | 0.9158 |
| Precision | 0.8962 |
| Recall | 0.9125 |
| F1-Score | 0.9043 |
| ROC-AUC | 0.9733 |

Confusion Matrix:

|  | Predicted Real | Predicted Fake |
|--|---------------|---------------|
| **Actual Real** | 6,330 | 563 |
| **Actual Fake** | 466 | 4,861 |

Per-class Performance:

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Real (0) | 0.93 | 0.92 | 0.92 | 6,893 |
| Fake (1) | 0.90 | 0.91 | 0.90 | 5,327 |

**Visualization**

- ROC Curve

![XGBoost ROC](results/xgboost_roc_curve.png)

- Confusion Matrix

![XGBoost CM](results/xgboost_confusion_matrix.png)

- Feature Importance (Top 30)

![XGBoost FI](results/xgboost_feature_importance.png)

---

### Analysis

XGBoost achieves an accuracy of 91.6% and a ROC-AUC of 0.9733, indicating strong discriminative ability on TF-IDF features.

The model performs consistently across both classes, with slightly higher precision for real news (0.93 vs 0.90). This suggests the model is marginally more conservative in labeling articles as fake — a desirable property that reduces false accusations.

Compared to Logistic Regression (accuracy 0.933), XGBoost performs slightly lower in accuracy but provides a complementary non-linear perspective. The two models may capture different patterns in the feature space, which is useful for downstream ensemble methods explored by Person 3.

---

## Anomaly Detection

### Method

Beyond supervised classification, we investigate whether fake news articles exhibit anomalous linguistic patterns that unsupervised methods can detect without labels. Two complementary approaches are used:

- **Isolation Forest (IF):** Builds 200 random trees that isolate observations through recursive splits. Points that require fewer splits to isolate receive lower anomaly scores, as they are structurally distinct from the majority.

- **One-Class SVM (OC-SVM):** Learns a decision boundary around the training distribution in a reduced feature space. To handle the high dimensionality of TF-IDF vectors (5,000 features), we first apply Truncated SVD to project into 100 dimensions before fitting the RBF kernel.

Both methods use a contamination rate of 10%.

---

### Results

| Method | Anomalies Detected | Fake Ratio in Anomalies | Enrichment | Mann-Whitney p |
|--------|-------------------|------------------------|------------|----------------|
| Isolation Forest | 1,255 (10.3%) | 35.94% | 0.82x | 2.06e-17 *** |
| One-Class SVM | 1,171 (9.6%) | 29.89% | 0.69x | 3.39e-133 *** |
| Overall baseline | — | 43.59% | 1.00x | — |

Method agreement: 82.18% — the two methods largely agree on which samples are normal vs anomalous, with 124 samples flagged by both.

**Visualization**

- Score Distributions

![IF Scores](results/anomaly_if_score_distribution.png)
![SVM Scores](results/anomaly_svm_score_distribution.png)

- Fake Ratio Comparison

![Fake Ratio](results/anomaly_fake_ratio_comparison.png)

- Method Agreement

![Agreement](results/anomaly_agreement_pie.png)

- PR Curves

![IF PR](results/anomaly_if_pr_curve.png)
![SVM PR](results/anomaly_svm_pr_curve.png)

---

### Analysis

Both anomaly detection methods produce statistically significant results (Mann-Whitney p < 0.001), confirming that anomaly scores carry meaningful information about article authenticity.

However, the direction of the finding is notable: anomalies are disproportionately **real news**, not fake. Both methods show enrichment below 1.0x, meaning flagged anomalies contain a *lower* proportion of fake news than the overall dataset.

This suggests that fake news tends to follow more predictable, formulaic language patterns, while genuine journalism — with its diverse topics, writing styles, and source-specific conventions — produces more structural outliers in the TF-IDF feature space.

The two methods agree on 82.18% of samples. Among the 124 articles flagged as anomalous by both methods, the fake ratio drops further to just 20.16%, reinforcing the pattern that the most structurally unusual articles are overwhelmingly real.

From a practical standpoint, these results complement the supervised XGBoost classifier by offering an alternative lens: rather than asking "is this article fake?", anomaly detection reveals that fake news occupies a surprisingly narrow region of the feature space — a useful insight for understanding the nature of misinformation.

---

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
