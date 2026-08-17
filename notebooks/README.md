# Notebooks

Exploration and analysis live here. Import from `ml_template/` rather than copying logic into a cell:

```python
from ml_template.config import load_config
from ml_template.data_processor import clean_dataset, load_dataset

config = load_config()
frame = clean_dataset(
    load_dataset(config.data.raw_path),
    config.features.all_features,
    config.data.target,
    config.validation.value_ranges,
)
```

Anything a notebook proves useful should move into `ml_template/` and get a test.
