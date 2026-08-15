import streamlit as st

from src.app.secrets import load_app_secrets

load_app_secrets()

from src.presentation.pages import render_app


# Page header
st.set_page_config(page_title="Coin Sentiment / 加密貨幣情緒", layout="wide")

render_app()
