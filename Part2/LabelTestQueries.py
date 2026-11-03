import csv

from part1.DataPreparation import db_fashion
from part2.Evaluation import TEST_LABELS

# Relevance criteria of the expert judges, applied to the fields of the original product (not to the index).
# The 20 products of each query are the ones in test_labels.csv.

def details(product: dict) -> dict:
    """product_details as one dict {name: value}."""
    return {name: value for item in product.get("product_details") or [] for name, value in item.items()}

def is_tshirt(product: dict) -> bool:
    return "t-shirt" in product["title"].lower()

def is_men(product: dict) -> bool:
    return "men" in product["title"].lower().split()

CRITERIA = {
    # men's t-shirt with a polo neck
    3: lambda p: is_men(p) and is_tshirt(p) and details(p).get("Neck Type") == "Polo Neck",
    # men's t-shirt, round neck, half sleeve (the dataset has a separate "Short Sleeve" value), cotton in the fabric
    4: lambda p: (is_men(p) and is_tshirt(p) and details(p).get("Neck Type") == "Round Neck"
                  and details(p).get("Sleeve") == "Half Sleeve" and "cotton" in details(p).get("Fabric", "").lower()),
    # Reebok men's t-shirt (the brand field is cut to REEB / REEBOK CLASSI, the full name is in the description)
    5: lambda p: str(p.get("brand")).upper().startswith("REEB") and is_men(p) and is_tshirt(p),
    # raincoat for men
    6: lambda p: p.get("sub_category") == "Raincoats" and is_men(p),
    # shirt (not t-shirt) whose colour is white and whose fabric contains linen (blends count)
    7: lambda p: (details(p).get("Color", "").lower() in ("white", "off white") and "linen" in details(p).get("Fabric", "").lower()
                  and "shirt" in p["title"].lower() and not is_tshirt(p)),
}


if __name__ == "__main__":
    products = {product["pid"]: product for product in db_fashion}   # `pid` of each original record
    with open(TEST_LABELS, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["labels"] = int(CRITERIA[int(row["query_id"])](products[row["pid"]]))
    with open(TEST_LABELS, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for query_id in CRITERIA:
        labels = [row["labels"] for row in rows if int(row["query_id"]) == query_id]
        print(f"query {query_id}: {sum(labels)} relevant of {len(labels)}")
