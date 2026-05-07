from datasets import Dataset, Features, Value, Image
import pandas as pd
from pathlib import Path
from PIL import Image as PILImage
import gc

BASE_DIR = "output"
OUTPUT_DIR = "parquet_output"

BATCH_SIZE = 400

caps = {
    "Latn": 4000,
    "Cyrl": 400,
    "Arab": 400,
    "Hani": 400,
    "Deva": 400,
    "Japn": 400,
}

scripts = [p.name for p in Path(BASE_DIR).iterdir() if p.is_dir()]

Path(OUTPUT_DIR).mkdir(exist_ok=True)

features = Features({
    "id": Value("string"),
    "language": Value("string"),
    "text": Value("string"),
    "img_old_document": Image(),
    "img_plain": Image(),
    "source": Value("string"),
    "script": Value("string"),
})

for script in scripts:

    max_cap = caps.get(script, 100)

    script_dir = Path(BASE_DIR) / script
    meta = pd.read_csv(script_dir / "metadata.csv")

    shard = 0
    rows = []
    count = 0

    for _, r in meta.iterrows():

        if count >= max_cap:
            break

        idx = str(r["id"])

        old_img = script_dir / "img_OLD_DOCUMENT" / f"{idx}.png"
        plain_img = script_dir / "img_PLAIN" / f"{idx}.png"

        if not old_img.exists() or not plain_img.exists():
            continue

        rows.append({
            "id": idx,
            "language": r["language"],
            "text": r["text"],
            "img_old_document": PILImage.open(old_img).convert("RGB"),
            "img_plain": PILImage.open(plain_img).convert("RGB"),
            "source": r["source"],
            "script": script
        })

        count += 1

        if len(rows) >= BATCH_SIZE:

            ds = Dataset.from_list(rows, features=features)

            out = Path(OUTPUT_DIR) / f"{script}_{shard:03d}.parquet"
            ds.to_parquet(out)

            rows.clear()
            del ds
            gc.collect()

            shard += 1

    if rows:

        ds = Dataset.from_list(rows, features=features)
        out = Path(OUTPUT_DIR) / f"{script}_{shard:03d}.parquet"
        ds.to_parquet(out)

        rows.clear()
        del ds
        gc.collect()

    print(script, count)
