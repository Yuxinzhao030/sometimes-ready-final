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
| SBERT + GaussianNB | (to be added) | — | — | — | — |

---

### Visualization

**ROC Curve**

![ROC](results/sbert_roc_curve.png)

**Confusion Matrices**

- SBERT + Logistic  
  ![Logistic CM](results/sbert_logistic_confusion.png)

- SBERT + XGBoost  
  ![XGBoost CM](results/sbert_xgboost_confusion.png)

---

### Analysis

SBERT-based models achieve strong performance, with accuracy around **86%** and ROC-AUC above **0.93**, indicating that semantic representations significantly improve separability.

Logistic Regression and XGBoost yield very similar overall performance, suggesting that:

> **representation quality plays a more critical role than model complexity**

A trade-off is observed:

- Logistic Regression achieves higher recall (fewer false negatives)  
- XGBoost achieves higher precision (fewer false positives)  

Gaussian Naive Bayes is expected to perform worse, as it assumes feature independence and Gaussian-distributed inputs—assumptions that do not hold well for high-dimensional semantic embeddings.

---

### Note

A comparison with TF-IDF-based models and a full evaluation of Gaussian Naive Bayes will be provided in subsequent sections.

