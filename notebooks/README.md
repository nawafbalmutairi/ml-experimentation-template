# Notebooks

Exploration and analysis live here. Import from `src/` rather than copying logic into a cell:

```python
from src.config import load_config
from src.data_processor import clean_dataset, load_dataset

config = load_config()
frame = clean_dataset(
    load_dataset(config.data.raw_path), config.features.all_features, config.data.target
)
```

Anything a notebook proves useful should move into `src/` and get a test.
