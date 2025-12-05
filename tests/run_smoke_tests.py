import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import traceback

print('Starting smoke tests')

# Test cache read/write
try:
    import pandas as pd
    from src.utils import cache as cache_mod
    print('cache module imported')
    # clear cache dir
    info = cache_mod.clear_cache_dir()
    print('clear_cache_dir ->', info)
    df = pd.DataFrame({'text': ['hello world', 'test'], 'timestamp': [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02')]})
    settings = {'dataset': 'price', 'coin': 'bitcoin', 'days': '1'}
    p = cache_mod.cache_csv(df, settings)
    print('cached to', p)
    loaded = cache_mod.load_cached_csv(settings)
    print('loaded type:', type(loaded))
    if loaded is None:
        print('ERROR: loaded is None')
    else:
        print('loaded shape:', getattr(loaded, 'shape', None))
        print(loaded.head().to_string())
except Exception:
    print('Cache test failed:')
    traceback.print_exc()

# Test sentiment analyzer import and a light call
try:
    import src.sentiment.analyzer as analyzer
    print('analyzer imported')
    try:
        # call a safe analyzer if available
        if hasattr(analyzer, 'textblob_analyze'):
            print('textblob_analyze ->', analyzer.textblob_analyze('I love crypto'))
        if hasattr(analyzer, 'vader_analyze'):
            print('vader_analyze ->', analyzer.vader_analyze('I love crypto'))
    except Exception as e:
        print('Analyzer function call raised:', repr(e))
except Exception:
    print('Analyzer import failed:')
    traceback.print_exc()

print('Smoke tests finished')
