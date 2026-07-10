"""Create a product-disjoint train/test split for AmazonHistoryPrice.

The original AHP release ships one JSON file per category with no canonical
split, so training and evaluation previously drew from the same product pool.
This script pools all products, shuffles with a fixed seed, and writes an
80/20 product-level split to data/ahp/train/ and data/ahp/test/. The category
field is preserved on each product, and the split is stratified by category so
both pools cover the full category distribution.

Usage: python scripts/split_ahp.py
"""

import json
import random
from pathlib import Path

SEED = 42
TEST_FRACTION = 0.2
AHP_DIR = Path("data/ahp")


def main():
    rng = random.Random(SEED)
    train_products: list[dict] = []
    test_products: list[dict] = []

    category_files = sorted(
        f for f in AHP_DIR.glob("*.json") if f.parent == AHP_DIR
    )
    if not category_files:
        raise SystemExit(f"No category JSON files found in {AHP_DIR}")

    for path in category_files:
        with open(path, encoding="utf-8") as f:
            products = json.load(f)
        rng.shuffle(products)
        n_test = max(1, round(len(products) * TEST_FRACTION))
        test_products.extend(products[:n_test])
        train_products.extend(products[n_test:])
        print(f"{path.name}: {len(products)} products -> "
              f"{len(products) - n_test} train / {n_test} test")

    for name, products in [("train", train_products), ("test", test_products)]:
        out_dir = AHP_DIR / name
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"ahp.{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(products, f, indent=1)
        print(f"Wrote {len(products)} products to {out_path}")


if __name__ == "__main__":
    main()
