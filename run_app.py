import os
import streamlit as st
import logging


try:
    for key, value in st.secrets.items():
        os.environ[str(key)] = str(value)
except Exception:
    logging.info(
        "No .streamlit/secrets.toml found; continuing with environment variables only."
    )
from config.settings import DEMO_MODE
from src.presentation.pages import render_app

# Page header
st.set_page_config(page_title="Crypto Sentiment Tracker", layout="wide")


render_app(demo_mode=DEMO_MODE)
