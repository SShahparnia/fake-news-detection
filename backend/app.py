import os
import re
import joblib
import pandas as pd
import __main__
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import download

# 1) Ensure NLTK data is available
download('wordnet')
download('omw-1.4')
download('stopwords')

# 2) Re-define and monkey-patch clean_text so joblib can unpickle
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(doc):
    doc = re.sub(r"<[^>]+>", " ", doc)
    tokens = re.findall(r"\b\w+\b", doc.lower())
    lemmed = [lemmatizer.lemmatize(t) for t in tokens]
    filtered = [t for t in lemmed if t not in stop_words]
    return " ".join(filtered)

__main__.clean_text = clean_text

# 3) Load your trained pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fake_news_clf_lemmatized.joblib")
model = joblib.load(MODEL_PATH)

# 4) Define your request body
class Article(BaseModel):
    text: str
    title: str = ""
    source: str = ""
    category: str = ""

# 5) Create the FastAPI app
app = FastAPI()

# 6) Mount the frontend folder (assumes ../frontend contains index.html + any assets)
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# 7) ML endpoint
@app.post("/predict")
def predict(item: Article):
    try:
        X_new = pd.DataFrame([{
            "text":     item.text,
            "title":    item.title,
            "source":   item.source,
            "category": item.category
        }])
        pred = model.predict(X_new)[0]
        prob = model.predict_proba(X_new)[0][pred]
        label = "fake" if pred == 1 else "real"
        return {"label": label, "confidence": round(float(prob), 4)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 8) Serve index.html on root & any unmatched path (SPA fallback)
@app.get("/{full_path:path}")
def spa_fallback(full_path: str):
    html_path = os.path.join(frontend_dir, "index.html")
    return FileResponse(html_path)
