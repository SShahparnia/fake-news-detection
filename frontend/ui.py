import streamlit as st
import requests

st.title("📰 Fake News Detector")
text = st.text_area("Paste article text here", height=300)

if st.button("Check"):
    resp = requests.post(
        "http://localhost:8000/predict",
        json={"text": text}
    )
    data = resp.json()
    st.markdown(
        f"**Prediction:** {data['label'].upper()}  \n"
        f"**Confidence:** {data['confidence']:.2%}"
    )
