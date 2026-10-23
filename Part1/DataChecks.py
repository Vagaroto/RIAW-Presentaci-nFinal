"""Figures about the preprocessing that are quoted in the report.

The results are saved as CSV files in data_checks_results/ to be referenced in the report.
"""
from pathlib import Path

import pandas as pd

from DataPreparation import db_fashion, preprocess

OUTPUT_DIR = Path(__file__).parent / "data_checks_results"
docs = preprocess(db_fashion)


def numeric_fields():
    """Missing values and range of the numeric values kept for ranking."""
    rows = []
    for field in ["selling_price", "actual_price", "discount", "average_rating", "out_of_stock"]:
        values = [doc["numeric"][field] for doc in docs]
        present = [value for value in values if value is not None]
        numbers = [value for value in present if not isinstance(value, bool)]
        rows.append({
            "field": field,
            "missing": len(values) - len(present),
            "missing_pct": 100 * (len(values) - len(present)) / len(values),
            "min": min(numbers) if numbers else None,
            "max": max(numbers) if numbers else None,
            "true_count": sum(present) if field == "out_of_stock" else None,
        })
    return pd.DataFrame(rows)


def preprocessing_checks():
    """Effect of joining t-shirt into one term."""
    return pd.DataFrame([
        {"check": "products", "value": len(docs)},
        {"check": "titles_with_tshirt_token", "value": sum("tshirt" in doc["processed"]["title"] for doc in docs)},
    ])


OUTPUT_DIR.mkdir(exist_ok=True)
tables = {"numeric_fields": numeric_fields(), "preprocessing_checks": preprocessing_checks()}
for name, table in tables.items():
    table.round(3).to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

print(f"Results saved in {OUTPUT_DIR}: {', '.join(f'{name}.csv' for name in tables)}")
