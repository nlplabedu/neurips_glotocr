import os
import re

DATASET_ROOT = "../../../uniOCR.bench-v1.0-results"
# DATASET_ROOT = "../../../uniOCR-results-with-hint"

RESULT_DIR = "../res_v1.0"

MODELS = [
    # "gpt-4.1+img_plain_hint",
    "gpt-4.1+img_plain",
    # "dots-ocr+img_old_document",
    # "dots-ocr-1.5+img_old_document",
    # "glm-ocr-v2+img_old_document",
    # "gpt-4.1+img_old_document",
    # "hunyuan-ocr+img_old_document",
    # "olmocr2-vllm+img_old_document",
    "glm-ocr-v2+img_plain",
    "dots-ocr-1.5+img_plain",
    "paddleocr-vl-1.5+img_plain",
    "gemini-3.1-flash-lite-preview+img_plain",
    "dots-ocr+img_plain",
    "deepseek-ocr2-vllm+img_plain",
    # "paddleocr-vl+img_plain",
    "olmocr2-vllm+img_plain",
    "nanonets-ocr2+img_plain",
    "firered-ocr+img_plain",
    "lighton-ocr2+img_plain",
    "rolm-ocr+img_plain",
    "hunyuan-ocr+img_plain",
    "qwen3-vl-8b+img_plain",
]

# ---------- firered extractor ----------

REASONING_PATTERNS = [
    "converting the text",
    "converting to markdown",
    "converting the image",
    "generating the markdown",
    "markdown output",
    "i'm now",
    "i am now",
    "i've been working",
    "i have been working",
    "i'm focusing",
    "i'm satisfied",
    "the final step",
    "ready to provide",
    "reversing the text",
    "reversing the order",
    "transcribing the text",
    "translating the text",
]


def extract_firered(markdown):
    if markdown is None:
        return ""

    text = str(markdown)

    cleaned_lines = []
    for line in text.splitlines():
        lower = line.lower()
        if not any(p in lower for p in REASONING_PATTERNS):
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.M)
    text = text.replace("**", "").replace("*", "")

    return text.strip()


# ---------- lighton extractor ----------

def extract_lighton(markdown):
    if markdown is None:
        return ""

    text = str(markdown)

    # ---- remove math wrappers ----
    text = text.replace("$", "")

    # ---- unwrap \text{...} and ext{...} ----
    text = re.sub(r"(?:\\text|ext)\s*\{([^}]*)\}", r"\1", text)

    # ---- Greek letters ----
    greek = {
        r"\alpha": "α", r"\beta": "β",
        r"\gamma": "γ", r"\Gamma": "Γ",
        r"\delta": "δ", r"\Delta": "Δ",
        r"\epsilon": "ε", r"\varepsilon": "ε",
        r"\zeta": "ζ",
        r"\eta": "η",
        r"\theta": "θ", r"\Theta": "Θ",
        r"\vartheta": "θ",
        r"\iota": "ι",
        r"\kappa": "κ",
        r"\lambda": "λ", r"\Lambda": "Λ",
        r"\mu": "μ",
        r"\nu": "ν",
        r"\xi": "ξ", r"\Xi": "Ξ",
        r"\pi": "π", r"\Pi": "Π",
        r"\rho": "ρ", r"\varrho": "ρ",
        r"\sigma": "σ", r"\Sigma": "Σ",
        r"\varsigma": "ς",
        r"\tau": "τ",
        r"\upsilon": "υ", r"\Upsilon": "Υ",
        r"\phi": "φ", r"\Phi": "Φ",
        r"\varphi": "φ",
        r"\chi": "χ",
        r"\psi": "ψ", r"\Psi": "Ψ",
        r"\omega": "ω", r"\Omega": "Ω",
    }

    for k, v in greek.items():
        text = text.replace(k, v)

    # ---- unwrap math font macros ----
    text = re.sub(r"\\mathcal\s*\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathfrak\s*\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathbb\s*\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathbf\s*\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathit\s*\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathsf\s*\{([^}]*)\}", r"\1", text)

    # ---- unwrap generic latex commands ----
    text = re.sub(r"\\[a-zA-Z]+\s*\{([^}]*)\}", r"\1", text)

    # ---- remove remaining latex commands ----
    text = re.sub(r"\\[a-zA-Z]+", "", text)

    return text.strip()


# ---------- paddleocr extractor ----------

def extract_paddleocr(markdown):
    if markdown is None:
        return ""

    text = str(markdown)

    cleaned_lines = []
    for line in text.splitlines():
        lower = line.lower()
        if "provided image is a graphic design" not in lower:
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    return text.strip()


# ---------- qwen3 extractor ----------

def extract_qwen3(text):
    if text is None:
        return ""

    text = str(text)
    text = extract_lighton(text)
    lines = text.splitlines()

    inside = False
    found_block = False
    collected = []

    for line in lines:

        stripped = line.strip()
        # start fence
        if stripped.startswith("```"):

            tag = stripped[3:].strip()

            if not inside and (tag == "" or tag.lower() == "markdown"):
                inside = True
                found_block = True
                continue

            if inside:
                inside = False
                continue

        if inside:
            collected.append(line)

    if found_block:
        return "\n".join(collected).strip()

    # ---- no markdown blocks -> remove reasoning lines ----

    reasoning_patterns = [
        "the provided image",
        "the image contains",
        "cannot provide",
        "cannot interpret",
        "convert the text",
        "convert into markdown",
        "markdown format",
        "since the text",
        "it appears to be",
        "optical character recognition",
        "without additional context",
        "here is the text",
        "text from the image"
    ]

    kept = []

    for line in lines:

        l = line.strip()
        if not l:
            continue

        low = l.lower()

        if any(p in low for p in reasoning_patterns):
            continue

        kept.append(l)

    text = "\n".join(kept).strip()

    text = text.replace("\(", "").replace("\)", "")
    text = text.replace("\\", "")
    text = text.replace("\\\\", "")
    return text


# ---------- deepseek extractor ----------

def extract_deepseek(markdown):
    if markdown is None:
        return ""

    text = str(markdown)
    text = re.sub(r"<\|ref\|>.*?<\|/ref\|>", "", text)
    text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text)

    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.M)

    return text.strip()


def extract_glm(markdown):
    if markdown is None:
        return ""

    text = str(markdown)
    """Remove ```markdown ... ``` and plain ``` ... ``` fences, return inner content."""
    text = re.sub(r"```markdown\s*\n(.*?)```", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"```\s*\n(.*?)```", r"\1", text, flags=re.DOTALL)

    return text.strip()


# ---------- default extractor ----------

def extract_default(markdown):
    if markdown is None:
        return ""

    text = str(markdown)
    return text.strip()


# ---------- Register extractors ----------

EXTRACTORS = {
    "firered-ocr+img_plain": extract_firered,
    "lighton-ocr2+img_plain": extract_lighton,
    # "paddleocr-vl+img_plain": extract_paddleocr,
    "qwen3-vl-8b+img_plain": extract_qwen3,
    "deepseek-ocr2-vllm+img_plain": extract_deepseek,
    "glm-ocr-v2+img_plain": extract_glm
}

for model in MODELS:
    if model not in EXTRACTORS:
        EXTRACTORS[model] = extract_default
