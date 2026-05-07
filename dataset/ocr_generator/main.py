import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from engine import load_fonts, render_sentence, render_face
from augmentations import apply_augmentations
from config import CONFIGS

CSV_DIR = "../seed/seed_data"
FONT_BASE_DIR = "../fonts/fonts_data"
OUTPUT_DIR = "output"

SELECTED_PROFILES = ["PLAIN", "OLD_DOCUMENT"]


def main():
    csv_files = [f for f in Path(CSV_DIR).glob("*.csv") if f.stat().st_size > 0]
    csv_files = [f for f in csv_files if any(k.lower() in f.name.lower() for k in ['Gujr'])]  # filter
    # csv_files = [f for f in csv_files if not any(k.lower() in f.name.lower() for k in ['Latn', 'Gujr'])] # filter
    for csv_file in csv_files:

        script = csv_file.stem
        font_dir = os.path.join(FONT_BASE_DIR, script)

        if not os.path.isdir(font_dir):
            print(f"⚠️ No fonts for script {script}, loading all fonts")
            font_dir = FONT_BASE_DIR

        print(f"\nProcessing script: {script}")

        fonts = load_fonts(font_dir)

        try:
            df = pd.read_csv(csv_file)
        except pd.errors.EmptyDataError:
            print(f"⚠️ Empty CSV for script {script}, skipping")
            continue

        if df.empty:
            print(f"⚠️ No rows for script {script}, skipping")
            continue

        # -----------------------------
        # Output folders
        # -----------------------------

        script_dir = os.path.join(OUTPUT_DIR, script)

        img_dirs = [os.path.join(script_dir, f"img_{p}") for p in SELECTED_PROFILES]

        for img_dir in img_dirs:
            os.makedirs(img_dir, exist_ok=True)

        metadata_rows = []

        for idx, row in tqdm(
                df.iterrows(),
                total=len(df),
                desc=f"Rendering {script}"
        ):

            sentence = row["text"]
            language = row["language"]
            source = row["source"]

            rendered_text, rendered_face, rendered_lines, selected_font_path = render_sentence(sentence, fonts)

            if rendered_text is None or rendered_face is None or rendered_lines is None or selected_font_path is None:
                continue

            imgs = {}
            valid = True

            # -----------------------------
            # Render all profiles first
            # -----------------------------
            for p in SELECTED_PROFILES:
                profile = CONFIGS[p]
                img = render_face(rendered_face, rendered_lines, selected_font_path, profile)

                if img is None:
                    valid = False
                    break

                img = apply_augmentations(img, profile)
                imgs[p] = img

            # -----------------------------
            # Skip if any failed
            # -----------------------------
            if not valid:
                continue

            # -----------------------------
            # Save images
            # -----------------------------
            for p, img in imgs.items():
                img_dir = os.path.join(script_dir, f"img_{p}")

                img_name = f"{idx}.png"
                img_path = os.path.join(img_dir, img_name)

                img.save(img_path)

            # -----------------------------
            # Save metadata once
            # -----------------------------
            metadata_rows.append({
                "id": idx,
                "language": language,
                "text": rendered_text,
                "source": source
            })

        # -----------------------------
        # Save metadata
        # -----------------------------

        metadata_df = pd.DataFrame(metadata_rows)

        metadata_path = os.path.join(script_dir, "metadata.csv")
        metadata_df.to_csv(metadata_path, index=False)

        print(f"Saved {len(metadata_rows)} samples for {script}")

    print("\nAll scripts processed.")


if __name__ == "__main__":
    main()
