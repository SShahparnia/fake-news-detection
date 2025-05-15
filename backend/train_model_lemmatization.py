import re
import pandas as pd
import joblib
from nltk.stem import WordNetLemmatizer
from nltk import download
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

download('wordnet')
download('omw-1.4')
download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(doc):
    doc = re.sub(r"<[^>]+>", " ", doc)
    tokens = re.findall(r"\b\w+\b", doc.lower())
    lemmed = [lemmatizer.lemmatize(t) for t in tokens]
    filtered = [t for t in lemmed if t not in stop_words]
    return " ".join(filtered)

df = pd.read_csv("fake_news_dataset.csv")
df = df.dropna(subset=["text", "label"])
df["label_bin"] = df["label"].map({"real": 0, "fake": 1})
X = df[["text", "title", "source", "category"]]
y = df["label_bin"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

body_tfidf = TfidfVectorizer(preprocessor=clean_text, max_df=0.8, ngram_range=(1,2))
title_tfidf = TfidfVectorizer(preprocessor=clean_text, max_df=0.9, ngram_range=(1,1))
meta_ohe = OneHotEncoder(handle_unknown="ignore")
preprocessor = ColumnTransformer([
    ("body", body_tfidf, "text"),
    ("title", title_tfidf, "title"),
    ("meta", meta_ohe, ["source", "category"])
])

pipeline = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(solver="saga", class_weight="balanced", max_iter=2000))
])

param_grid = {"pre__body__max_df": [0.7, 0.8], "clf__C": [0.1, 1, 10]}
grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
preds = grid.predict(X_test)
print(grid.best_params_)
print(classification_report(y_test, preds, target_names=["real", "fake"]))
joblib.dump(grid.best_estimator_, "fake_news_clf_lemmatized.joblib")


# Compute confusion matrix
cm = confusion_matrix(y_test, preds)

# Create display
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["real", "fake"])

# Plot and save
plt.figure(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
plt.title("Confusion Matrix: Real vs Fake")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()