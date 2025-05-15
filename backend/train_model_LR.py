import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Load & label-encode
df = pd.read_csv("fake_news_dataset.csv")
df = df.dropna(subset=["text","label"])
df["label_bin"] = df["label"].map({"real":0,"fake":1})

# 2. Split
X = df[["title","text","source","category"]]
y = df["label_bin"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Build transformers
text_body = ("text", TfidfVectorizer(
                 stop_words="english",
                 max_df=0.8,
                 ngram_range=(1,2)
             ), "text")
text_title = ("title", TfidfVectorizer(
                  stop_words="english",
                  max_df=0.9,
                  ngram_range=(1,1)
              ), "title")
meta_ohe = ("meta", OneHotEncoder(handle_unknown="ignore"), ["source","category"])

pre = ColumnTransformer([text_body, text_title, meta_ohe])

# 4. Pipeline + grid
pipe = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(
                solver="saga",
                class_weight="balanced",
                max_iter=2000
           ))
])

param_grid = {
    "pre__text__max_df": [0.7,0.8],
    "clf__C": [0.1, 1, 10]
}

grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
preds = grid.predict(X_test)
print(classification_report(y_test, preds, target_names=["real","fake"]))

joblib.dump(grid.best_estimator_, "fake_news_clf.joblib")
