# Fake News Detection

An end-to-end fake-news classification system:  
- **Backend**: model training pipelines + FastAPI `/predict` endpoint  
- **Frontend**: interactive Streamlit app for users to paste articles and get real-vs-fake predictions  

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Tech Stack](#tech-stack)  
- [Project Structure](#project-structure)  
- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Data](#data)  
- [Training the Model](#training-the-model)  
- [Backend API](#backend-api)  
- [Frontend UI](#frontend-ui)  
- [Evaluation & Visualization](#evaluation--visualization)  
- [Deployment](#deployment)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Overview

This project demonstrates a complete workflow for detecting fake news with machine learning:

1. **Data preprocessing** (lemmatization, stop-word removal)  
2. **Feature engineering** (TF-IDF + metadata)  
3. **Model training** (Logistic Regression pipeline)  
4. **API serving** (FastAPI)  
5. **User interface** (Streamlit)  

We chose the **lemmatized + metadata** pipeline over a vanilla TF-IDF+LR or simple XGBoost because it yielded the best F1-score in cross-validation by combining clean text normalization with contextual signals (`title`, `source`, `category`).

---

## Features

- **Robust text cleaning** with NLTK lemmatization  
- **Metadata fusion**: incorporates article title, source, and category  
- **Balanced classification** using `class_weight="balanced"`  
- **Real-time API**: FastAPI `/predict` endpoint  
- **Interactive UI**: Streamlit app for instant feedback  
- **Visualization**: confusion matrix PNG for model evaluation  

---

## Tech Stack

- **Python** 3.8+  
- **scikit-learn** for pipelines and modeling  
- **NLTK** for text preprocessing  
- **FastAPI** + **Uvicorn** for the backend  
- **Streamlit** for the frontend  
- **joblib** to serialize pipelines  

---

## Project Structure

```text
FAKE-NEWS-DETECTION/
│
├── backend/
│   ├── app.py                         # FastAPI server exposing /predict
│   ├── fake_news_clf.joblib           # baseline TF-IDF + LR model
│   ├── fake_news_clf_lemmatized.joblib# final lemmatized+metadata pipeline
│   ├── requirements.txt               # backend dependencies
│   ├── train_model_LR.py              # trains baseline TF-IDF + LR
│   ├── train_model_test.py            # scratchpad for quick tests
│   └── train_model_lemmatization.py   # trains final lemmatized+metadata model
│
├── frontend/
│   ├── requirements.txt               # frontend dependencies
│   └── ui.py                          # Streamlit interface (paste, predict)
│
├── .gitignore                         # ignores __pycache__, dataset, joblibs
├── confusion_matrix.png               # saved confusion matrix for evaluation
├── fake_news_dataset.csv              # synthetic dataset (gitignored)
└── README.md                          # this file
```

## 📦 Dataset

- **Source**: [Fake News Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/mahdimashayekhi/fake-news-detection-dataset)  
- **Records**: 20,000 synthetic news articles  
- **Columns**: `title`, `text`, `date`, `source`, `author`, `category`, `label`  
- ~5% of entries have missing `source` and `author` fields to mimic real-world conditions  
- The file is named `fake_news_dataset.csv` and **excluded from GitHub** via `.gitignore`  

---

## 🛠️ Prerequisites

- Python 3.8+  
- Git  
- (Optional) Conda or `venv` for managing virtual environments  

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<USERNAME>/fake-news-detection.git
   cd fake-news-detection
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r backend/requirements.txt -r frontend/requirements.txt
   ```

### 📁 Requirements

**backend/requirements.txt**
```
pandas scikit-learn joblib fastapi uvicorn datasets transformers matplotlib numpy scipy pydantic nltk
```

**frontend/requirements.txt**
```
streamlit requests
```

---

## 🧠 Model Training

1. **Baseline Model**
   ```bash
   python backend/train_model_LR.py
   ```
   * TF-IDF → Logistic Regression
   * Output: `backend/fake_news_clf.joblib`

2. **Final Model (lemmatized + metadata)**
   ```bash
   python backend/train_model_lemmatization.py
   ```
   * Uses:
      * NLTK lemmatization + stop-word filtering
      * `ColumnTransformer` to fuse:
         * TF-IDF on `text`
         * TF-IDF on `title`
         * One-hot encoding of `source`, `category`
   * Output: `backend/fake_news_clf_lemmatized.joblib`

### ✅ Why This Model?

* Lemmatization reduces noise by collapsing similar word forms
* Stop-word removal improves signal-to-noise
* Article `source` and `category` add valuable context
* Outperformed plain TF-IDF+LR and simple XGBoost in cross-validation

---

## 🧩 Backend API

**Start the FastAPI server**
```bash
uvicorn backend.app:app --reload
```

**API Endpoint**
`POST /predict`

**Input JSON:**
```json
{
  "text": "Article body text...",
  "title": "Optional title",
  "source": "Optional source",
  "category": "Optional category"
}
```

**Response JSON:**
```json
{
  "label": "real" | "fake",
  "confidence": 0.87
}
```

---

## 🎯 Frontend UI

**Run the Streamlit app**
```bash
streamlit run frontend/ui.py
```

**Usage**
* Paste the article into the text box
* Click **Check**
* View real/fake prediction and confidence score

---

## 📊 Evaluation

* Confusion matrix generated and saved to: `confusion_matrix.png`
* Use it to analyze model performance and tweak features

---

## 🚀 Deployment

### Backend (Render or Heroku)

**Build Command:**
```bash
pip install -r backend/requirements.txt && python backend/train_model_lemmatization.py
```

**Start Command:**
```bash
uvicorn backend.app:app --host 0.0.0.0 --port $PORT
```

### Frontend (Streamlit Cloud)

* Link GitHub repo to Streamlit Cloud
* Set entry point to `frontend/ui.py`
* No additional configuration needed

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to your fork:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a pull request!
