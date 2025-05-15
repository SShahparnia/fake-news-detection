import streamlit as st
import requests

st.title("📰 Fake News Detector")
text = st.text_area("Paste article text here", height=300)

if st.button("Check"):
    payload = {
        "title": "",   # <— required by your FastAPI model
        "text": text
    }
    resp = requests.post("http://localhost:8000/predict", json=payload)
    if resp.status_code == 200:
        data = resp.json()
        st.markdown(
            f"**Prediction:** {data['label'].upper()}  \n"
            f"**Confidence:** {data['confidence']:.2%}"
        )
    else:
        st.error(f"API error {resp.status_code}: {resp.text}")
