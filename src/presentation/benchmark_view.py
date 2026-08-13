import streamlit as st

from src.benchmark.analyzer_eval import run_fixed_benchmark
from src.benchmark.benchmark_plot import accuracy_figure, confusion_matrices
from src.presentation.translations import TEXT


def show_benchmark_data(language: str) -> None:
    text = TEXT[language]
    st.header(text["analyzer_benchmark"])

    results, table = run_fixed_benchmark()
    unavailable = {
        name: result
        for name, result in results.items()
        if not result.get("available", True)
    }

    for name, result in unavailable.items():
        st.info(f"{name} {text['unavailable']}: {result['reason']}")

    st.dataframe(table, use_container_width=True)
    accuracy_figure(table)
    confusion_matrices(results)
