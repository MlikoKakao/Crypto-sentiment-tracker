import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


EXPECTED_SECRET_KEYS = {
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
    "REDDIT_USER_AGENT",
    "YOUTUBE_API_KEY",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
    "DATABASE_URL",
    "ADMIN_API_KEY",
    "DEMO",
    "HF_DEVICE",
}


def load_app_secrets() -> None:
    load_dotenv()

    try:
        import streamlit as st
        from streamlit.errors import StreamlitSecretNotFoundError

        try:
            for key in EXPECTED_SECRET_KEYS:
                if key in st.secrets:
                    os.environ[key] = str(st.secrets[key])
        except StreamlitSecretNotFoundError:
            logger.info("No Streamlit secrets found; using environment variables.")
    except ImportError:
        logger.info("Streamlit not installed; using environment variables.")
