

# 🦠 COVID-19 Tweet Classification – Updated Solution

This repository contains a **state-of-the-art solution** for the [Zindi COVID-19 Tweet Classification Challenge](https://zindi.africa/competitions/covid-19-tweet-classification).
The model **identifies tweets related to COVID-19** without relying solely on keywords like "covid" or "coronavirus".

---

## ✨ Features

* ✅ **Text Preprocessing:** Cleans URLs, mentions, hashtags, punctuation, and extra spaces
* ✅ **Meta Features:** Tweet length, hashtags, mentions, uppercase words, sentiment scores
* ✅ **TF-IDF Features:** Word-level and character-level n-grams
* ✅ **BERT Embeddings:** DistilBERT mean-pooled embeddings for contextual understanding
* ✅ **Ensemble Modeling:** Combines TF-IDF, BERT, and meta features
* ✅ **LightGBM with Stratified K-Fold:** Robust cross-validation and early stopping
* ✅ **Easy Submission:** Generates CSV for Zindi evaluation

---

⚡ How It Works

This solution follows a robust NLP and machine learning pipeline to classify COVID-19 related tweets:

Load Data

Reads the training and test CSV files containing tweet text and target labels.

Text Cleaning

Converts text to lowercase and removes URLs, mentions, hashtags, punctuation, and extra whitespace.

Meta Feature Extraction

Calculates features such as tweet length, number of hashtags, mentions, uppercase words, digits, and sentiment (polarity and subjectivity).

TF-IDF Features

Word-level and character-level TF-IDF vectors capture syntactic and semantic patterns.

BERT Embeddings

DistilBERT embeddings (mean-pooled from the last 4 layers) capture contextual meaning beyond simple token frequencies.

Feature Combination

Merges TF-IDF, BERT embeddings, and meta features into a single feature matrix for training.

LightGBM Training

Stratified K-Fold cross-validation ensures robust evaluation.

Early stopping prevents overfitting and handles class imbalance.

Submission Generation

Produces a CSV file with ID and predicted target probabilities ready for Zindi evaluation.

---

🎯 Performance

This updated solution demonstrates strong predictive performance for the COVID-19 Tweet Classification challenge:

Cross-validated ROC-AUC Score: 0.242230032 

Robust Predictions: Effectively identifies COVID-19 related tweets without relying solely on keywords.
