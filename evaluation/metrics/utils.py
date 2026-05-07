import re
import json
import unicodedata
from GlotScript import sp


def normalize_text(text):
    if text is None:
        return ""

    return text.strip()


def remove_accents(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        c for c in text
        if unicodedata.category(c) != "Mn"
    )
    return unicodedata.normalize("NFC", text)


def levenshtein(a, b):
    # remove all whitespace characters
    a = re.sub(r"\s+", "", a)
    b = re.sub(r"\s+", "", b)

    n, m = len(a), len(b)

    if n > m:
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))

    for i in range(1, m + 1):

        previous, current = current, [i] + [0] * n

        for j in range(1, n + 1):

            add = previous[j] + 1
            delete = current[j - 1] + 1
            change = previous[j - 1]

            if a[j - 1] != b[i - 1]:
                change += 1

            current[j] = min(add, delete, change)

    return current[n]


def cer(pred, gt):

    gt_len = max(len(gt), 1)

    candidates = [
        (pred, gt),
        (pred, gt[::-1]),

        (pred.lower(), gt.lower()),
        (pred.lower(), gt[::-1].lower()),

        (remove_accents(pred), gt),
        (remove_accents(pred.lower()), gt.lower()),
    ]

    dists = [levenshtein(a, b) for a, b in candidates]

    dist = min(dists)
    cer_score = min(dist / gt_len, 1.0)

    return cer_score


def detect_script(text):
    if not text:
        return "None"
    try:
        s_pred = sp(text)[0]
        if s_pred:
            return s_pred
        else:
            "None"
    except Exception:
        return "None"


def save_json(path, obj):
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
