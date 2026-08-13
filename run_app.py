import os
import streamlit as st

from src.app.secrets import load_app_secrets

load_app_secrets()

from src.presentation.pages import render_app


# Page header
st.set_page_config(page_title="Coin Sentiment", layout="wide")

demo_mode = os.getenv("DEMO", "0") == "1"

render_app(demo_mode=demo_mode)
