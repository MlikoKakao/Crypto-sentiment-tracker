import os
import streamlit as st
import logging
from streamlit.errors import StreamlitSecretNotFoundError

from src.presentation.pages import render_app

EXPECTED_SECRET_KEYS = {
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
    "REDDIT_USER_AGENT",
    "YOUTUBE_API_KEY",
    "DATABASE_PATH",
    "DEMO",
    "HF_DEVICE",
}

try:
    for key in EXPECTED_SECRET_KEYS:
        if key in st.secrets:
            os.environ[str(key)] = str(st.secrets[key])
except StreamlitSecretNotFoundError:
    logging.info(
        "No .streamlit/secrets.toml found; continuing with environment variables only."
    )

demo_mode = os.getenv("DEMO", "0") == "1"


# Page header
st.set_page_config(page_title="Crypto Sentiment Tracker", layout="wide")


render_app(demo_mode=demo_mode)
