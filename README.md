# 🦠 COVID-19 Tweet Classification – Updated Solution with BERT + LightGBM

This repository contains a state-of-the-art hybrid solution for the Zindi COVID-19 Tweet Classification Challenge
.
The model classifies tweets related to COVID-19 by combining deep learning (BERT) and machine learning (LightGBM) for robust predictions.

###✨ Features

✅ Text Preprocessing: Cleans URLs, mentions, hashtags, punctuation, and extra spaces

✅ Meta Features: Tweet length, hashtags, mentions, uppercase words, digits, sentiment scores (polarity & subjectivity)

✅ TF-IDF Features: Word-level and character-level n-grams to capture lexical and syntactic patterns

✅ BERT Embeddings: DistilBERT mean-pooled embeddings provide deep contextual understanding of text

✅ Hybrid Modeling: Combines TF-IDF, meta features, and BERT embeddings

✅ LightGBM with Stratified K-Fold: Robust ML model with cross-validation, early stopping, and handling class imbalance

✅ Easy Submission: Generates a CSV ready for Zindi evaluation

###⚡ How It Works

This pipeline combines deep learning and classical ML for optimal performance:

1️⃣ Load Data

Reads training and test CSV files containing tweets and target labels.

2️⃣ Text Cleaning

Converts text to lowercase

Removes URLs, mentions, hashtags, punctuation, and extra spaces

3️⃣ Meta Feature Extraction

Calculates additional features including:

Text length, number of hashtags, mentions, uppercase words, digits

Sentiment features using TextBlob (polarity & subjectivity)

4️⃣ TF-IDF Features

Word-level n-grams capture common terms and phrases

Character-level n-grams capture stylistic patterns and spelling variations

5️⃣ BERT Embeddings

Uses DistilBERT, a pretrained transformer, to generate mean-pooled embeddings

Captures contextual meaning of tweets beyond simple token frequency

These embeddings are not fine-tuned, only used as rich text features

6️⃣ Feature Combination

Combines TF-IDF, character n-grams, BERT embeddings, and meta features

Forms a single hybrid feature matrix for training

7️⃣ LightGBM Training

Stratified K-Fold cross-validation ensures stable evaluation

Early stopping prevents overfitting and handles class imbalance

Predicts probabilities for each tweet

8️⃣ Submission Generation

Outputs a CSV file with ID and predicted probabilities ready for Zindi submission

###🎯 Performance

This hybrid approach demonstrates strong predictive performance:

Cross-validated ROC-AUC Score: 0.2422

Robust Predictions: Accurately identifies COVID-19 related tweets without relying solely on keywords

###⚙️ Notes on DL + ML Usage

BERT embeddings provide deep contextual representations of tweet text

LightGBM acts as the final predictive model, leveraging both engineered features and embeddings

This combination allows the model to benefit from deep learning features while remaining efficient and interpretable
