import random
from pathlib import Path
import pandas as pd
import regex as re

MIN_CHARS = 30
MAX_CHARS = 100
LONG_CHAR_THRESHOLD = 60
LEAST_CHAR_LEN = 2
MAX_PER_FILE_PER_ROUND = 5


# ---------------------------------------------------------
# Generate overlapping chunks
# ---------------------------------------------------------
def generate_chunks(text, min_chars, max_chars):

    chunks = []

    if " " in text:
        tokens = text.split()
        joiner = " "
    else:
        tokens = re.findall(r"\X", text)
        joiner = ""

    if len(tokens) < 2:
        return chunks

    avg_len = sum(len(t) for t in tokens[:10]) / min(len(tokens), 10)
    approx_tokens = max(2, int(min_chars / max(avg_len, 1)))

    window_size = approx_tokens
    step_size = max(1, int(window_size * 0.8))

    for i in range(0, len(tokens) - window_size + 1, step_size):

        chunk_tokens = tokens[i:i + window_size]
        chunk = joiner.join(chunk_tokens)

        if len(chunk) < min_chars:
            j = i + window_size
            while len(chunk) < min_chars and j < len(tokens):
                chunk_tokens.append(tokens[j])
                chunk = joiner.join(chunk_tokens)
                j += 1

        if len(chunk) <= max_chars:
            chunks.append(chunk)

    return chunks


# ---------------------------------------------------------
# Sampling function
# ---------------------------------------------------------
def sample_script(base_paths, script, seed=42):

    random.seed(seed)
    base_paths = [Path(p) for p in base_paths]

    script_folders = []

    for base_path in base_paths:
        if not base_path.exists():
            continue

        for f in base_path.iterdir():
            if f.is_dir() and f.name.endswith(f"_{script}"):
                script_folders.append(f)

    if not script_folders:
        return None

    print(f"\nProcessing script: {script}")

    caps = {
        "Latn": 4000,
        "Cyrl": 400,
        "Arab": 400,
        "Hani": 400,
        "Deva": 400,
        "Japn": 400,
    }

    max_cap = caps.get(script, 110)

    # ---------------------------------------
    # Collect txt files
    # ---------------------------------------
    all_files = []

    for folder in script_folders:
        for txt_file in folder.glob("*.txt"):
            source = txt_file.stem.split("_")[-1]
            all_files.append((folder.name, txt_file, source))

    if not all_files:
        return None

    random.shuffle(all_files)

    normal_priority = []
    low_priority = []

    for item in all_files:

        _, txt_file, _ = item
        name = txt_file.name.lower()

        if any(tag in name for tag in ["_lyrics", "_pbc", "_cbc", "_jw"]):
            low_priority.append(item)

        else:
            normal_priority.append(item)

    all_files = normal_priority + low_priority

    collected = []
    seen_texts = set()

    outside_bucket = []
    long_bucket = []
    jw_bucket = []

    cap_reached = False

    # ---------------------------------------
    # PASS 1 — Round robin sampling
    # ---------------------------------------
    file_iters = []

    for lang_folder, txt_file, source in all_files:

        try:
            f = open(txt_file, "r", encoding="utf-8")
            file_iters.append((lang_folder, txt_file, source, f))
        except:
            continue

    active = True

    while active and len(collected) < max_cap:

        active = False

        for lang_folder, txt_file, source, f in file_iters:

            taken = 0

            while taken < MAX_PER_FILE_PER_ROUND:

                line = f.readline()

                if not line:
                    break

                active = True

                text = line.strip()

                if not text or text in seen_texts:
                    continue

                char_len = len(text)
                is_jw = "_jw" in txt_file.name.lower()

                if char_len < LEAST_CHAR_LEN:
                    continue

                # ideal sample
                if MIN_CHARS <= char_len <= MAX_CHARS and not is_jw:

                    collected.append({
                        "language": lang_folder,
                        "text": text,
                        "source": source
                    })

                    seen_texts.add(text)
                    taken += 1

                    if len(collected) >= max_cap:
                        cap_reached = True
                        break

                else:

                    if is_jw:
                        jw_bucket.append((lang_folder, text, source))
                    else:
                        outside_bucket.append((lang_folder, text, char_len, source))

                    if char_len > LONG_CHAR_THRESHOLD and not is_jw:
                        long_bucket.append((lang_folder, text, source))

            if cap_reached:
                break

    for _, _, _, f in file_iters:
        f.close()

    # ---------------------------------------
    # PASS 2 — Long sentence chunks
    # ---------------------------------------
    if len(collected) < max_cap:

        all_chunks = []

        for lang_folder, text, source in long_bucket:

            chunks = generate_chunks(text, MIN_CHARS, MAX_CHARS)

            for chunk in chunks:
                if chunk not in seen_texts:
                    all_chunks.append((lang_folder, chunk, source))

        random.shuffle(all_chunks)

        for lang_folder, chunk, source in all_chunks:

            if chunk in seen_texts:
                continue

            collected.append({
                "language": lang_folder,
                "text": chunk,
                "source": source
            })

            seen_texts.add(chunk)

            if len(collected) >= max_cap:
                cap_reached = True
                break

    # ---------------------------------------
    # PASS 3 — Outside length
    # ---------------------------------------
    if len(collected) < max_cap:

        outside_bucket.sort(
            key=lambda x: (
                0 if x[2] > MAX_CHARS else 2 if x[2] < MIN_CHARS else 1,
                -x[2] if x[2] < MIN_CHARS else x[2]
            )
        )

        for lang_folder, text, _, source in outside_bucket:

            if text in seen_texts:
                continue

            collected.append({
                "language": lang_folder,
                "text": text,
                "source": source
            })

            seen_texts.add(text)

            if len(collected) >= max_cap:
                cap_reached = True
                break

    # ---------------------------------------
    # PASS 4 — JW fallback
    # ---------------------------------------
    if len(collected) < max_cap:

        for lang_folder, text, source in jw_bucket:

            if text in seen_texts:
                continue

            collected.append({
                "language": lang_folder,
                "text": text,
                "source": source
            })

            seen_texts.add(text)

            if len(collected) >= max_cap:
                cap_reached = True
                break

    if cap_reached:
        print(f"Cap reached ({max_cap}) for {script}: {len(collected)}.")
    else:
        print(f"Warning: Only {len(collected)} samples collected (cap={max_cap}) for {script}.")

    return pd.DataFrame(collected)


# ---------------------------------------------------------
# Find scripts
# ---------------------------------------------------------
def find_all_scripts(base_paths):

    base_paths = [Path(p) for p in base_paths]
    scripts = set()

    for base_path in base_paths:

        if not base_path.exists():
            continue

        for folder in base_path.iterdir():

            if folder.is_dir() and "_" in folder.name:

                parts = folder.name.split("_")

                if len(parts) == 2:
                    scripts.add(parts[1])

    return sorted(scripts)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":

    base_paths = [
        "/path/to/glotlid-corpus/v3.1",
        "/path/to/glotlid-corpus/v4.0-ext"
    ]

    seed_value = 42

    output_dir = Path("seed_data")
    output_dir.mkdir(exist_ok=True)

    scripts = find_all_scripts(base_paths)

    print(f"Detected scripts: {scripts}")

    for script in scripts:

        df = sample_script(base_paths, script, seed=seed_value)

        if df is not None:

            output_file = output_dir / f"{script}.csv"

            df.to_csv(output_file, index=False)

            print(f"Saved {output_file}")

    print("\nAll scripts processed.")