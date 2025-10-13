import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.sparse import hstack, csr_matrix
from textblob import TextBlob

# =====================
# 1️⃣ Load Data
# =====================
df = pd.read_csv('/content/Train (5).csv')
test = pd.read_csv('/content/Test (4).csv')

y = df["target"]
df_id = df["ID"].copy()
test_id = test["ID"].copy()

# Ensure 'text' column exists
if "text" not in df.columns:
    df.columns = ["ID", "text", "target"]
if "text" not in test.columns:
    test.columns = ["ID", "text"]

# =====================
# 2️⃣ Text Cleaning
# =====================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text"] = df["text"].apply(clean_text)
test["text"] = test["text"].apply(clean_text)

# =====================
# 3️⃣ Meta Features
# =====================
def add_meta_features(df):
    df['text_len'] = df['text'].apply(len)
    df['num_hashtags'] = df['text'].apply(lambda x: x.count('#'))
    df['num_mentions'] = df['text'].apply(lambda x: x.count('@'))
    df['num_exclamations'] = df['text'].apply(lambda x: x.count('!'))
    df['num_questions'] = df['text'].apply(lambda x: x.count('?'))
    df['num_uppercase'] = df['text'].apply(lambda x: sum(1 for w in x.split() if w.isupper()))
    df['num_digits'] = df['text'].apply(lambda x: sum(c.isdigit() for c in x))
    df['hashtag_ratio'] = df['num_hashtags'] / (df['text_len'] + 1)
    df['mention_ratio'] = df['num_mentions'] / (df['text_len'] + 1)
    df['polarity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    return df

df = add_meta_features(df)
test = add_meta_features(test)

meta_cols = ['text_len','num_hashtags','num_mentions','num_exclamations','num_questions',
             'num_uppercase','num_digits','hashtag_ratio','mention_ratio','polarity','subjectivity']

# =====================
# 4️⃣ TF-IDF Features
# =====================
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    max_df=0.9,
    stop_words='english'
)

char_vectorizer = TfidfVectorizer(
    analyzer='char_wb', ngram_range=(2,5), max_features=5000
)

all_text = pd.concat([df['text'], test['text']], axis=0)
X_all_tfidf = vectorizer.fit_transform(all_text)
X_all_char = char_vectorizer.fit_transform(all_text)

X_train_tfidf = X_all_tfidf[:len(df)]
X_test_tfidf = X_all_tfidf[len(df):]

X_train_char = X_all_char[:len(df)]
X_test_char = X_all_char[len(df):]

# =====================
# 5️⃣ DistilBERT Embeddings (Mean Pooling)
# =====================
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = AutoModel.from_pretrained("distilbert-base-uncased")
bert_model.eval()

def get_bert_mean_pooling(text_list, batch_size=16):
    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch_text = text_list[i:i+batch_size]
        inputs = tokenizer(batch_text, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = bert_model(**inputs, output_hidden_states=True)
        last_4 = torch.stack(outputs.hidden_states[-4:]).mean(0)
        batch_embeddings = last_4.mean(dim=1).numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

X_train_bert = get_bert_mean_pooling(df['text'].tolist())
X_test_bert = get_bert_mean_pooling(test['text'].tolist())

# =====================
# 6️⃣ Combine Features
# =====================
meta_features_train = df[meta_cols].values
meta_features_test = test[meta_cols].values

X_train = hstack([X_train_tfidf, X_train_char, csr_matrix(X_train_bert), csr_matrix(meta_features_train)])
X_test = hstack([X_test_tfidf, X_test_char, csr_matrix(X_test_bert), csr_matrix(meta_features_test)])

# =====================
# 7️⃣ LightGBM Training with Stratified K-Fold
# =====================
NFOLDS = 5
skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(df))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    print(f"\n🔹 Fold {fold + 1}")
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "is_unbalance": True,
        "seed": 42
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )

    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / NFOLDS

auc_score = roc_auc_score(y, oof_preds)
print(f"\n✅ CV AUC Score: {auc_score:.4f}")

# =====================
# 8️⃣ Save Submission
# =====================
submission = pd.DataFrame({
    "ID": test_id,
    "target": test_preds
})
submission.to_csv("submission_lightgbm_bert_full_features.csv", index=False)
print("\n✅ submission_lightgbm_bert_full_features.csv saved successfully!")
