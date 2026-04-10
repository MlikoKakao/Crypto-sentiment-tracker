from __future__ import annotations
from typing import Any
import os


try:
    import torch.nn.functional as f
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception:  # fallback when transformers or torch not installed
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    torch = None
    f = None

_finbert_model: Any = None
_finbert_tokenizer: Any = None
_hf_device: int = int(os.environ.get("HF_DEVICE", "-1"))

# Load the FinBERT model and tokenizer if not already loaded, and return them.
def _load_finbert() -> tuple[Any, Any]:
    global _finbert_model, _finbert_tokenizer
    if _finbert_model is None or _finbert_tokenizer is None:
        if AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise RuntimeError("transformers not available")
        _finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        _finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        if _hf_device >= 0 and torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            _finbert_model = _finbert_model.to(f"cuda:{_hf_device}")
    return _finbert_model, _finbert_tokenizer

def finbert_analyze(text: str) -> float:
    model, tok = _load_finbert()
    if model is None or tok is None or torch is None or f is None:
        raise RuntimeError("FinBERT model/tokenizer/torch/f not available")
    short = str(text)[:1000]
    enc = tok(short, truncation=True, padding=True, max_length=512, return_tensors="pt")

    # CUDA support
    if _hf_device >= 0 and torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        enc = {k: v.to(f"cuda:{_hf_device}") for k, v in enc.items()}

    # no_grad - not for training, logits = model output before softmax
    with torch.no_grad():
        logits = model(**enc).logits[0]

    probs = f.softmax(logits, dim=-1).detach().cpu().numpy()
    # id2label maps class indices to labels, e.g. {0: "negative", 1: "neutral", 2: "positive"}
    id2label = model.config.id2label
    # Create a dict mapping labels to their probabilities, e.g. {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
    pdict = {id2label[i].lower(): float(probs[i]) for i in range(len(probs))}
    
    return pdict.get("positive", 0.0) - pdict.get("negative", 0.0)