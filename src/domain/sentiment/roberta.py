from __future__ import annotations
import os
from typing import Any, Optional, TypedDict

try:
    from transformers import pipeline  # type: ignore[import]
except ImportError:  # pragma: no cover - fallback when transformers not installed
    pipeline = None  # type: ignore

pipeline_fn: Any = pipeline  # type: ignore
_roberta: Any = None
_device: int = int(os.environ.get("HF_DEVICE", "-1"))


class RobertaAnalyzer:
    def __init__(self, batch_size: int = 16):
        self.batch_size = batch_size
        self.pipe = None
        
    def _load(self):
        if self.pipe is None:
            if pipeline_fn is None:
                raise RuntimeError("transformers.pipeline not available")
            self.pipe = pipeline_fn(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
                truncation=True,
                max_length=512,
                padding=True,
                device=_device,
            )
        return self.pipe

    def analyze(self, text: Optional[str]) -> float:
        return self.analyze_many([text])[0]

    def analyze_many(self, texts: list[str | None]) -> list[float]:
        pipe = self._load()
        short_texts = ["" if text is None else str(text)[:1000] for text in texts]
        results = pipe(short_texts, batch_size=self.batch_size)
        return [_score_roberta_result(result) for result in results]
    
    def __call__(self, text: str | None) -> float:
        return self.analyze(text)
    
class RobertaResult(TypedDict):
    label: str
    score: float
    
    
def _score_roberta_result(result: RobertaResult) -> float:
    label = str(result["label"]).lower()
    score = float(result["score"])

    if label == "negative":
        return -score
    if label == "positive":
        return score
    return 0.0