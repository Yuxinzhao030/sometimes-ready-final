# eco395m-FinalProject



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

| Model              | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------------------|---------|----------|--------|---------|--------|
| SBERT + Logistic | 0.860   | 0.832    | 0.851  | 0.841   | 0.932 |
| SBERT + XGBoost  | 0.862   | 0.865    | 0.811  | 0.837   | 0.935 |
| SBERT + GaussianNB | 0.763 | 0.673    | 0.723  | 0.697   | — |

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

In contrast, Gaussian Naive Bayes performs substantially worse, with accuracy dropping to around 76%.

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

> **model assumptions must align with representation structure**

and that not all classifiers are suitable for dense semantic embeddings.
---

### TBD

A comparison with TF-IDF-based models .

