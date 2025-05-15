import os
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detector")
st.write("Paste an article below and click **Check** to see if it's real or fake.")

text = st.text_area("Article text", height=300)

if st.button("Check"):
    if not text.strip():
        st.warning("Please paste some text first.")
    else:
        try:
            resp = requests.post(
                f"{API_URL}/predict",
                json={"text": text}
            )
            resp.raise_for_status()
            data = resp.json()
            st.markdown(f"**Prediction:** `{data['label'].upper()}`")
            st.markdown(f"**Confidence:** `{data['confidence'] * 100:.1f}%`")
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")
        except KeyError:
            st.error(f"Unexpected response: {resp.text}")
