

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



