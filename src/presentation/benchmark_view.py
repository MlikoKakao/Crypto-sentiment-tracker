import streamlit as st

from src.benchmark.analyzer_eval import run_fixed_benchmark
from src.benchmark.benchmark_plot import accuracy_figure, confusion_matrices


def show_benchmark_data() -> None:
    st.header("Analyzer benchmark")

    results, table = run_fixed_benchmark()

    st.dataframe(table, use_container_width=True)
    accuracy_figure(table)
    confusion_matrices(results)
