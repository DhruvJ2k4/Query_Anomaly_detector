# Query Anomaly Detector
---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Concept & Methodology](#concept--methodology)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Results](#results)
- [License](#license)

---

## Overview

**Query Anomaly Detector** is a data-driven project implemented in Jupyter Notebooks for the detection of anomalies in query patterns using advanced machine learning and statistical techniques. The project is designed for data scientists, ML engineers, and researchers aiming to identify unusual or suspicious queries in large datasets.

---

## Features

- 📊 **Exploratory Data Analysis (EDA)** for query datasets
- 🛠️ **Preprocessing**: Cleansing, normalization, and feature extraction
- 🤖 **Machine Learning Models**: Used Clustering Technique for anomaly detection
- 📈 **Visualization**: Detailed plots and dashboards for result interpretation
- 🧩 **Notebook-based workflow**: Easy to modify, rerun, and experiment
- 🚀 **Extensible & Modular**: Add your own data, models, or metrics

---

## Concept & Methodology

### Problem Statement

Anomaly detection in queries aims to identify query records that deviate significantly from the norm, which may indicate data issues, fraud, or system misuse.

### Approach

1. **Data Acquisition**: Load and inspect query data.
2. **Preprocessing**: Clean, transform, and engineer features relevant to anomaly detection.
3. **Modeling**: Apply unsupervised ML models for anomaly detection.
4. **Evaluation**: Use metrics and visualization to interpret model performance.
5. **Inference**: Apply the trained model to new data to flag anomalies.

#### Algorithms Used

- K-Means Clustering
- Principal Component Analysis (PCA)
- TF - IDF vectorisation

### Methodology
This project implements a query anomaly detection pipeline using Natural Language Processing (NLP) and machine learning. The process includes:

### Preprocessing

All queries are converted to lowercase.
Punctuation and numbers are removed.
Stopwords are filtered out with NLTK.
Lemmatization is performed using spaCy.
Words with ≤2 characters are discarded.
Boost Words

A set of "boost words" (e.g., hacked, phishing, breach, bomb, vpn, alert) is used to highlight queries likely to be anomalous or security-relevant.
TF-IDF Vectorization

Preprocessed queries are tokenized and converted into TF-IDF vectors.
This encodes the importance and uniqueness of each word in the context of the entire dataset.
Anomaly Scoring

Queries are scored using their TF-IDF representation, with additional weight given to queries containing boost words.
These scores can be used for downstream tasks like clustering, thresholding, or visualization.
Results
Data Summary:

Total queries: 12,613
Unique words (vocabulary): 3,549
TF-IDF matrix shape: (12613, 3549)
Sample Preprocessed Queries:

---

## Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab
- Recommended: [Anaconda](https://www.anaconda.com/products/distribution)

### Usage

1. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
2. **Open the main notebook (e.g., `Query_Anomaly_Detector.ipynb`) and follow the instructions in the cells.**
3. **Replace or update input data as needed.**
4. **Run all cells to preprocess data, train the model, and view results.**

---

## Results

- **Performance Metrics:** [Accuracy, Precision, Recall, F1-score, ROC-AUC, etc.]
- **Visualization:** Anomaly score distribution, confusion matrix, example flagged queries.


---

## License

This project is licensed under the [MIT License](LICENSE).

---
