from __future__ import annotations
from typing import Any, Dict, Tuple

HASH_KEYS_BY_DATASET: Dict[str, Tuple[str, ...]] = {
    "posts_news": ("dataset","coin", "source", "query","start_date","end_date","num_posts"),
    "posts_reddit":  ("dataset", "coin", "source", "query", "start_date", "end_date", "num_posts"),
    "posts_twitter": ("dataset", "coin", "source", "query", "start_date", "end_date", "num_posts"),
    "price":     ("dataset", "coin", "days"),
    "sentiment": ("dataset", "coin", "source", "analyzer", "input_sha1"),
    "merged":    ("dataset", "coin", "days", "analyzer", "posts_choice", "depends_on"),
    "features":  ("dataset", "coin", "days", "analyzer", "posts_choice", "depends_on", "lag_min_s", "lag_max_s", "lag_step_s", "metric")
} 
#Each dict entry maps a dataset type to the exact tuple of setting keys that define that file's bytes


DEFAULTS: Dict[str, Any] = { #Defaults - unused for now, could implement later but dont really see use

}

#Checks for type of values inside dictionary, if its not date makes it lower, if its list, tuple or set, sorts them - so order doesnt change hash, same values will have same hash, returns values
def _norm_value(key: str, v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str):
        return v if key in ("start_date", "end_date") else v.lower()#if its not date makes it lower
    if isinstance(v,(list,tuple,set)):
        return sorted([str(x).lower() for x in v]) or None #if its list, tuple or set, sorts them (what's the use for this?
    return v #Returns values

#Checks if [] has all required keys to be settings
def _canonicalize_settings(raw: Dict[str, Any]) -> Dict[str, Any]:
    if "dataset" not in raw:
        raise ValueError("settings missing required key:dataset") #If no dataset = error
    ds = raw["dataset"] #Takes in only keys required to be dataset
    if ds not in HASH_KEYS_BY_DATASET: #Checks if used valid dataset name (like cant pass "prices" bcs dataset is "price")
        raise ValueError(f"Unknown dataset {ds}")
    
    out: Dict[str, Any] = {} #Assigns empty dict
    for k in HASH_KEYS_BY_DATASET[ds]:
        v = _norm_value(k, raw.get(k, DEFAULTS.get(k))) #normalize values inside the dictionary?
        if v in (None, [], {}):#Treats empty values as invalid for required keys
            raise ValueError(f"settings for dataset {ds} missing key: {k}")
        out[k] = v
    return out