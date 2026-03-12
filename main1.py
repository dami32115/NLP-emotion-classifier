# DG3NLP – Emotion Classification (Tweets)
# Pipeline :
# clean text → TF-IDF features → a few extra signals → Logistic Regression baseline → tuned Linear SVM


# Imports
# Using the standard libraries :
# - pandas/numpy for data handling
# - scikit-learn for splitting, TF-IDF, models and evaluation
# - scipy.sparse to combine sparse and dense features
# - matplotlib for plotting the confusion matrix

import re
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt


# Reproducibility
# Using a fixed random seed so the splits and model results stay the same each run.
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Folder where the confusion matrix will be saved.
OUT_DIR = Path("reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

label_names = ["anger", "fear", "joy", "love", "sadness", "surprise"]


# Load the dataset
# Keeping the dataset path relative to the project folder so it works in scripts
DATA_PATH = "data/Option 1-Training Dataset.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print(df.head())

# checks class balance before modelling.
print("\nLabel counts:")
print(df["label"].value_counts().sort_index())



# Date preprocessing
# Kept simple on purpose:
# - lowercase everything
# - remove URLs and @mentions
# - keep hashtag words but remove "#"
# - squeeze extra spaces
#
# This avoids removing useful emotional cues.

URL_RE = re.compile(r"http\S+|www\S+")
MENTION_RE = re.compile(r"@\w+")
MULTISPACE_RE = re.compile(r"\s+")

def clean_text(t):
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = URL_RE.sub("", t)
    t = MENTION_RE.sub("", t)
    t = t.replace("#", "")
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t

df["text_clean"] = df["text"].apply(clean_text)

print("\nCleaned text examples:")
print(df[["text", "text_clean"]].head())



# Simple extra features
# These are small binary signals that sometimes help:
# - emoji present
# - elongated letters like “soooo”
# - negation words
# They’re cheap to compute and easy to interpret.

X_text = df["text_clean"]
y = df["label"]

def extra_feats(s):
    if not isinstance(s, str):
        s = ""
    has_emoji = 1 if re.search(r"[\U00010000-\U0010FFFF]", s) else 0
    has_elong = 1 if re.search(r"(.)\1{2,}", s) else 0
    has_neg   = 1 if re.search(r"\b(no|not|never|n't)\b", s) else 0
    return [has_emoji, has_elong, has_neg]

extra = X_text.apply(extra_feats).apply(pd.Series)
extra.columns = ["has_emoji", "has_elong", "has_neg"]


# Train / Val / Test split
# Using a 70/15/15 split with stratify so the label proportions stay consistent.

X_tr, X_tmp, y_tr, y_tmp, extra_tr, extra_tmp = train_test_split(
    X_text, y, extra, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
X_va, X_te, y_va, y_te, extra_va, extra_te = train_test_split(
    X_tmp, y_tmp, extra_tmp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_tmp
)

print(f"\nSplit sizes: train={len(X_tr)}, val={len(X_va)}, test={len(X_te)}")



# TF-IDF vectoriser
# TF-IDF converts text into numeric features.
# Settings used:
# - unigrams + bigrams (captures things like “not good”)
# - drop rare one-off terms (min_df=2)
# - drop extremely common ones (max_df=0.90)
# - sublinear_tf helps balance very frequent words

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.90,
    lowercase=True,
    strip_accents="unicode",
    sublinear_tf=True
)

Xtr_tfidf = vectorizer.fit_transform(X_tr)
Xva_tfidf = vectorizer.transform(X_va)
Xte_tfidf = vectorizer.transform(X_te)

print("\nTF-IDF feature count:", Xtr_tfidf.shape[1])


# Add the extra features
# We simply stack the 3 binary features next to the TF-IDF matrix.

def add_extra(X_sparse, df_extra):
    return hstack([X_sparse, csr_matrix(df_extra.values)], format="csr")

Xtr = add_extra(Xtr_tfidf, extra_tr)
Xva = add_extra(Xva_tfidf, extra_va)
Xte = add_extra(Xte_tfidf, extra_te)



# Baseline: Logistic Regression
# class_weight="balanced" helps because some emotion classes appear less often.

lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
lr.fit(Xtr, y_tr)
pred_va_lr = lr.predict(Xva)

print("\n[LR] Val Accuracy:", round(accuracy_score(y_va, pred_va_lr), 4))
print("[LR] Val Macro-F1:", round(f1_score(y_va, pred_va_lr, average="macro"), 4))

# LinearSVC (main model)
# Here I'm using a Linear Support Vector Machine (LinearSVC),
# which is known to work really well with text data + TF-IDF.
#
# Why LinearSVC?
# - It's fast even with large sparse matrices (like TF-IDF).
# - It's good at separating high-dimensional data.
# - Works well for short texts like tweets.
#
# What “C” means:
# - C is the regularisation strength.
# - Smaller C = more regularisation (simpler model).
# - Larger C = less regularisation (model fits training harder).
#
# Since we don’t know the best C value beforehand,
# I test a few options (0.25, 0.5, 1.0, 2.0) and pick the one
# that gives the highest macro-F1 score on the validation set.
#
# Macro-F1 matters because:
# - It gives equal weight to every class.
# - Our dataset isn’t perfectly balanced, so accuracy alone
#   could be misleading.

# Variables to store the best model + best performance
best_svm = None     # this will hold the actual SVM model that performs best
best_f1 = -1        # start at -1 so anything is better
best_c = None       # will store the C value that works best

# Loop through a few C values to see which performs best
for C in [0.25, 0.5, 1.0, 2.0]:
    
    # Create an SVM model with this specific C value
    # class_weight="balanced" helps because some emotion labels
    # appear more often than others.
    svm = LinearSVC(C=C, class_weight="balanced", random_state=RANDOM_STATE)
    
    # Train the SVM model on our combined TF-IDF + extra-features training set
    svm.fit(Xtr, y_tr)
    
    # Makes predictions on the validation set
    pred = svm.predict(Xva)

    # Calculate accuracy and macro-F1 for this C
    acc = accuracy_score(y_va, pred)
    f1m = f1_score(y_va, pred, average="macro")

    # Prints out how this C value performed
    # Helps compare them directly in the notebook output.
    print(f"[SVM C={C}] Val Acc: {acc:.4f} | Val Macro-F1: {f1m:.4f}")

    # If this model is currently the best one, it saves this 
    if f1m > best_f1:
        best_f1 = f1m        # update best score
        best_svm = svm       # store this trained model
        best_c = C           # store which C produced this result

# After trying all C values, print which one worked best
print(f"\nChosen SVM C = {best_c} (best macro-F1 = {best_f1:.4f})")



# Final test results
# Only evaluating on the test set once to keep it fair.

test_pred = best_svm.predict(Xte)
test_acc = accuracy_score(y_te, test_pred)
test_f1m = f1_score(y_te, test_pred, average="macro")

print("\nFINAL TEST - Accuracy:", round(test_acc, 4))
print("FINAL TEST - Macro-F1:", round(test_f1m, 4))
print("\nFinal Test Classification Report:")
print(classification_report(y_te, test_pred, target_names=label_names, digits=3))


# Confusion Matrix
# A confusion matrix shows:
# - rows = true classes
# - columns = predicted classes
# - each cell = how many times the model predicted that class
# Normalising by row makes it easier to compare classes of different sizes.
# I also printed raw counts inside each cell to see how many mistakes there are.

cm = confusion_matrix(y_te, test_pred, labels=list(range(6)))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(6, 5))
im = plt.imshow(cm_norm, cmap="Blues")
plt.title("Confusion Matrix (Test) - Normalised")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(np.arange(6), label_names, rotation=45)
plt.yticks(np.arange(6), label_names)
plt.colorbar(im)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)

plt.tight_layout()
save_path = OUT_DIR / "confusion_matrix.png"
plt.savefig(save_path, dpi=200)
plt.close()

print(f"\nSaved confusion matrix to: {save_path}")
