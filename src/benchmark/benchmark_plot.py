import pandas as pd
import plotly.express as px
import numpy as np
from typing import List, Dict, Any
import streamlit as st


CANONICAL = ("negative", "neutral", "positive")


def to_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        rows.append(
            {
                "Model": name,
                "Accuracy": r["accuracy"],
                "F1 (macro)": r["f1_macro"],
                "Throughput (texts/s)": r.get("throughput_txt_per_s", np.nan),
                "Latency (ms/text)": (
                    1000 * r["time_sec"] / max(r.get("n_texts", 1), 1)
                )
                if r.get("time_sec")
                else np.nan,
            }
        )
    df = (
        pd.DataFrame(rows)
        .sort_values(["F1 (macro)", "Accuracy"], ascending=False)
        .reset_index(drop=True)
    )
    df.index = np.arange(1, len(df) + 1)
    df.index.name = "Rank"
    return df


def confusion_figure(cm: np.ndarray, labels=CANONICAL, title: str = "Confusion matrix"):
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    fig = px.imshow(
        df_cm,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="True", color="Count"),
        title=title,
        aspect="equal",
    )
    return fig


def accuracy_figure(table: pd.DataFrame):
    fig_acc = px.bar(
        table,
        x="Model",
        y="Accuracy",
        title="Model Accuracy on benchmark_labeled.csv",
        text=table["Accuracy"].map(lambda v: f"{v:.3f}"),
    )
    fig_acc.update_traces(textposition="outside")
    fig_acc.update_layout(yaxis=dict(range=[0, 1]), margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_acc, use_container_width=True)


def confusion_matrices(results):
    cols = st.columns(2)
    names = list(results.keys())
    for i, name in enumerate(names):
        fig_cm = confusion_figure(
            results[name]["confusion"], title=f"{name} - Confusion"
        )
        cols[i % 2].plotly_chart(fig_cm, use_container_width=True)

    st.markdown("#### Misclassified Examples")
    model_for_examples = st.selectbox("Choose model", names, index=0, key="bench_model")
    examples = results[model_for_examples].get("examples", [])
    if not examples:
        st.info("No misclassified examples or dataset too small")
    else:
        for t, yt, yp in examples:
            st.write(f"- **true:** '{yt}' **predicted:** '{yp}' - {t}")

