from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("fake_news_clf.joblib")

class Article(BaseModel):
    title: str
    text: str

@app.post("/predict")
def predict(item: Article):
    pred = model.predict([item.text])[0]
    prob = model.predict_proba([item.text])[0][pred]
    label = "fake" if pred == 1 else "real"
    return {"label": label, "confidence": round(float(prob), 4)}
