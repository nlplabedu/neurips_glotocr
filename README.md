# GlotOCR-bench


A multilingual OCR benchmark covering a wide range of writing scripts, designed to evaluate OCR models across hundreds of languages.

---

## Dataset

All dataset-related code is in the `dataset/` folder.

### 1. Fonts

Download and organize Google Fonts by script by running:

```bash
python dataset/fonts/get_fonts.py
```

### 2. Seed Text

Place per-script sentence CSVs in `dataset/seed/seed_data/`. You can generate them from the GlotLID corpus by running:

```bash
python dataset/seed/get_seed.py
```

The GlotLID corpus is available at:
[cis-lmu/glotlid-corpus](https://huggingface.co/datasets/cis-lmu/glotlid-corpus)

### 3. Image Generation

We provide two rendering profiles (`PLAIN` and `OLD_DOCUMENT`). You can adjust parameters in `dataset/ocr_generator/config.py` and the rendering logic in `dataset/ocr_generator/engine.py`, then generate images by running:

```bash
python dataset/ocr_generator/main.py
```

To export the generated images to Parquet format:

```bash
python dataset/ocr_generator/export.py
```

---

## Evaluation

### 1. Run OCR Models

Run the OCR models on the dataset using the scripts provided at:
[uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr)

### 2. Compute Metrics

Once model outputs are ready, compute CER, Acc@k, and ScriptAcc metrics by running:

```bash
cd evaluation/metrics
python main.py
```

Results are saved per model under `evaluation/res_v1.0/`, including per-script, per-language, and tier-level (high / mid / low resource) breakdowns.
