# src

Reusable code imported by the notebooks (`sys.path.append('../src')`).

| File | What it does |
|---|---|
| `data.py` | `load_hour_data`, `chronological_split` (cutoff 2012-07-01), `get_xy` (removes `instant`, `dteday`, `casual`, `registered`, `cnt`) |
| `preprocessing.py` | `build_preprocessor(kind)` - `'linear'` (one-hot hour x working day, no `atemp`) for Linear/Ridge/Lasso, `'forest'` (cyclic hour/month) for tree models |
| `baselines.py` | `persistence_forecast` - "same hour last week" reference forecast |

All learned transformations (scalers, encoders) are fitted inside a scikit-learn `Pipeline`, so they only ever see training data.
