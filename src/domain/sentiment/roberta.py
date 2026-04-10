from __future__ import annotations
import os
from typing import Any, Optional
try:
    from transformers import pipeline  # type: ignore[import]
except Exception:  # pragma: no cover - fallback when transformers not installed
    pipeline = None  # type: ignore

pipeline_fn: Any = pipeline  # type: ignore
_roberta: Any = None
_device: int = int(os.environ.get("HF_DEVICE", "-1"))
    

def roberta_analyze(text: Optional[str]) -> float:
    global _roberta
    if _roberta is None:
        if pipeline_fn is None:
            raise RuntimeError("transformers.pipeline not available")
        _roberta = pipeline_fn(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
            truncation=True,
            max_length=512,
            padding=True,
            device=_device 
        )
    if text is None:
        return 0.0
    short_text = str(text)[:1000]
    result = _roberta(short_text)[0]
    label = result["label"].lower()
    score = result["score"]
    return -score if label == "negative" else (score if label == "positive" else 0.0)