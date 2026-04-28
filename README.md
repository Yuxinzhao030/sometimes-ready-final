# Fake News Detection Using Machine Learning

## Introduction

The rapid spread of misinformation online has made fake news detection an increasingly important problem in machine learning and natural language processing. False or misleading news articles can influence public opinion, amplify social division, and undermine trust in legitimate media sources. As digital platforms continue to become primary channels for news consumption, scalable and automated methods for identifying fake news are becoming essential.

In this project, we investigate whether machine learning models can accurately distinguish between **real news** and **fake news** based solely on article text. We formulate this as a **binary classification problem**, where each article is labeled as either:

- **0 = Real News**
- **1 = Fake News**

Our goal is to compare multiple modeling approaches for fake news detection and determine which methods provide the best balance between predictive performance, robustness, and interpretability.

To study this problem, we explore three different machine learning perspectives:

### 1. Traditional Supervised Classification
We first establish strong baseline models using classical text feature engineering methods such as **TF-IDF**, combined with classifiers including:

- **Logistic Regression**
- **Naive Bayes**
- **XGBoost**

These models are computationally efficient, interpretable, and serve as strong benchmarks for text classification.

### 2. Deep Learning and Transformer-Based Models
We then investigate modern NLP approaches using **Sentence-BERT embeddings** and **fine-tuned BERT models**, which can capture contextual meaning, semantic relationships, and deeper linguistic patterns beyond simple word frequency representations.

These methods allow us to test whether pretrained language models significantly improve fake news detection performance over traditional approaches.

### 3. Anomaly Detection
In addition to supervised classification, we explore **anomaly detection methods** as an alternative framework for fake news detection.

Rather than directly learning to classify fake versus real news, anomaly detection models learn the distribution of legitimate news articles and identify fake news as statistically unusual or out-of-distribution observations. This approach may be especially useful when labeled fake news examples are scarce or when new forms of misinformation emerge that differ from historical training data.

By comparing supervised classification, transformer-based methods, and anomaly detection, we aim to better understand:

- how fake news differs from legitimate news in textual patterns,
- whether deep contextual models outperform classical NLP pipelines,
- and whether fake news can be effectively detected as a form of statistical anomaly.

We evaluate all approaches using standard classification metrics including **Accuracy**, **Precision**, **Recall**, **F1-score**, and **ROC-AUC**, and discuss their strengths, weaknesses, and practical implications for real-world misinformation detection systems.

