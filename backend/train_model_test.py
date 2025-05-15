import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv("fake_news_dataset.csv")
df = df.dropna(subset=["text", "label"])
df["label_bin"] = df["label"].map({"real": 0, "fake": 1})

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label_bin"],
    test_size=0.2, random_state=42,
    stratify=df["label_bin"]
)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.8, ngram_range=(1,2))),
    ("clf", LogisticRegression(solver="saga", max_iter=2000))
])

param_grid = {
    "tfidf__max_df": [0.7, 0.8],
    "clf__C": [0.5, 1, 2]
}
grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
preds = grid.predict(X_test)
print(classification_report(y_test, preds, target_names=["real", "fake"]))

joblib.dump(grid.best_estimator_, "fake_news_clf.joblib")
