import os
import shutil
import re
import requests
import zipfile
from tqdm import tqdm

ZIP_URL = "https://github.com/google/fonts/archive/refs/heads/main.zip"
ZIP_NAME = "google_fonts_main.zip"
EXTRACTED_FOLDER_NAME = "fonts_main"
SOURCE_ROOT = os.path.join(EXTRACTED_FOLDER_NAME, "ofl")
DEST_ROOT = "fonts_data"

SUBSET_TO_SCRIPT = {
    "adlam": "Adlm",
    "ahom": "Ahom",
    "anatolian-hieroglyphs": "Hluw",
    "arabic": "Arab",
    "armenian": "Armn",
    "avestan": "Avst",
    "balinese": "Bali",
    "bamum": "Bamu",
    "bassa-vah": "Bass",
    "batak": "Batk",
    "bengali": "Beng",
    "beria-erfe": "Berf",
    "bhaiksuki": "Bhks",
    "brahmi": "Brah",
    "braille": "Brai",
    "buginese": "Bugi",
    "buhid": "Buhd",
    "canadian-aboriginal": "Cans",
    "carian": "Cari",
    "caucasian-albanian": "Aghb",
    "chakma": "Cakm",
    "cham": "Cham",
    "cherokee": "Cher",
    "chorasmian": "Chrs",
    "coptic": "Copt",
    "cuneiform": "Xsux",
    "cypriot": "Cprt",
    "cypro-minoan": "Cpmn",
    "cyrillic": "Cyrl",
    "cyrillic-ext": "Cyrl",
    "deseret": "Dsrt",
    "devanagari": "Deva",
    "dives-akuru": "Diak",
    "dogra": "Dogr",
    "duployan": "Dupl",
    "egyptian-hieroglyphs": "Egyp",
    "elbasan": "Elba",
    "elymaic": "Elym",
    "ethiopic": "Ethi",
    "georgian": "Geor",
    "glagolitic": "Glag",
    "gothic": "Goth",
    "grantha": "Gran",
    "greek": "Grek",
    "greek-ext": "Grek",
    "gujarati": "Gujr",
    "gunjala-gondi": "Gong",
    "gurmukhi": "Guru",
    "hanifi-rohingya": "Rohg",
    "hanunoo": "Hano",
    "hatran": "Hatr",
    "hebrew": "Hebr",
    "imperial-aramaic": "Armi",
    "indic-siyaq-numbers": "Zyyy",
    "inscriptional-pahlavi": "Phli",
    "inscriptional-parthian": "Prti",
    "japanese": "Jpan",
    "javanese": "Java",
    "kaithi": "Kthi",
    "kana-extended": "Kana",
    "kannada": "Knda",
    "kawi": "Kawi",
    "kayah-li": "Kali",
    "kharoshthi": "Khar",
    "khitan-small-script": "Kits",
    "khmer": "Khmr",
    "khojki": "Khoj",
    "khudawadi": "Sind",
    "kirat-rai": "Krai",
    "korean": "Hang",
    "lao": "Laoo",
    "latin": "Latn",
    "latin-ext": "Latn",
    "lepcha": "Lepc",
    "limbu": "Limb",
    "linear-a": "Lina",
    "linear-b": "Linb",
    "lisu": "Lisu",
    "lycian": "Lyci",
    "lydian": "Lydi",
    "mahajani": "Mahj",
    "makasar": "Maka",
    "malayalam": "Mlym",
    "mandaic": "Mand",
    "manichaean": "Mani",
    "marchen": "Marc",
    "masaram-gondi": "Gonm",
    "math": "Zyyy",
    "mayan-numerals": "Zyyy",
    "medefaidrin": "Medf",
    "meetei-mayek": "Mtei",
    "mende-kikakui": "Mend",
    "meroitic": "Merc",
    "meroitic-cursive": "Merc",
    "meroitic-hieroglyphs": "Mero",
    "miao": "Plrd",
    "modi": "Modi",
    "mongolian": "Mong",
    "mro": "Mroo",
    "multani": "Mult",
    "music": "Zyyy",
    "myanmar": "Mymr",
    "nabataean": "Nbat",
    "nag-mundari": "Nagm",
    "nandinagari": "Nand",
    "new-tai-lue": "Talu",
    "newa": "Newa",
    "nko": "Nkoo",
    "nushu": "Nshu",
    "nyiakeng-puachue-hmong": "Hmnp",
    "ogham": "Ogam",
    "ol-chiki": "Olck",
    "old-hungarian": "Hung",
    "old-italic": "Ital",
    "old-north-arabian": "Narb",
    "old-permic": "Perm",
    "old-persian": "Xpeo",
    "old-sogdian": "Sogo",
    "old-south-arabian": "Sarb",
    "old-turkic": "Orkh",
    "old-uyghur": "Ougr",
    "oriya": "Orya",
    "osage": "Osge",
    "osmanya": "Osma",
    "ottoman-siyaq-numbers": "Zyyy",
    "pahawh-hmong": "Hmng",
    "palmyrene": "Palm",
    "pau-cin-hau": "Pauc",
    "phags-pa": "Phag",
    "phoenician": "Phnx",
    "psalter-pahlavi": "Phlp",
    "rejang": "Rjng",
    "runic": "Runr",
    "samaritan": "Samr",
    "saurashtra": "Saur",
    "sharada": "Shrd",
    "shavian": "Shaw",
    "siddham": "Sidd",
    "signwriting": "Sgnw",
    "sinhala": "Sinh",
    "sogdian": "Sogd",
    "sora-sompeng": "Sora",
    "soyombo": "Soyo",
    "sundanese": "Sund",
    "sunuwar": "Sunu",
    "syloti-nagri": "Sylo",
    "symbols": "Zyyy",
    "symbols2": "Zyyy",
    "syriac": "Syrc",
    "tagalog": "Tglg",
    "tagbanwa": "Tagb",
    "tai-le": "Tale",
    "tai-tham": "Lana",
    "tai-viet": "Tavt",
    "takri": "Takr",
    "tamil": "Taml",
    "tamil-supplement": "Taml",
    "tangsa": "Tnsa",
    "tangut": "Tang",
    "telugu": "Telu",
    "thaana": "Thaa",
    "thai": "Thai",
    "tibetan": "Tibt",
    "tifinagh": "Tfng",
    "tirhuta": "Tirh",
    "todhri": "Todr",
    "toto": "Toto",
    "ugaritic": "Ugar",
    "vai": "Vaii",
    "vietnamese": "Latn",
    "vithkuqi": "Vith",
    "wancho": "Wcho",
    "warang-citi": "Wara",
    "yezidi": "Yezi",
    "yi": "Yiii",
    "zanabazar-square": "Zanb",
    "znamenny": "Zyyy"
}


def download_and_extract():
    if not os.path.exists(ZIP_NAME):
        print("Downloading Google Fonts repository...")
        response = requests.get(ZIP_URL, stream=True)
        response.raise_for_status()

        with open(ZIP_NAME, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    if not os.path.exists(EXTRACTED_FOLDER_NAME):
        print("Extracting archive...")
        with zipfile.ZipFile(ZIP_NAME, "r") as zip_ref:
            zip_ref.extractall()
        if os.path.exists("fonts-main"):
            os.rename("fonts-main", EXTRACTED_FOLDER_NAME)


def extract_scripts(metadata_path):
    scripts = set()

    with open(metadata_path, "r", encoding="utf-8") as f:
        content = f.read()

    # primary_script
    primary = re.findall(r'primary_script:\s*"([^"]+)"', content)
    scripts.update(primary)

    # primary_language
    primary_languages = re.findall(r'primary_language:\s*"([^"]+)"', content)
    for lang in primary_languages:
        parts = lang.split("_")
        if len(parts) == 2:
            scripts.add(parts[1])

    # languages
    languages = re.findall(r'languages:\s*"([^"]+)"', content)
    for lang in languages:
        parts = lang.split("_")
        if len(parts) == 2:
            scripts.add(parts[1])

    # subsets
    subsets = re.findall(r'subsets:\s*"([^"]+)"', content)
    for subset in subsets:
        subset = subset.lower()
        if subset in SUBSET_TO_SCRIPT:
            scripts.add(SUBSET_TO_SCRIPT[subset])

    return list(scripts)


def organize_fonts():
    os.makedirs(DEST_ROOT, exist_ok=True)
    folders = [
        f for f in os.listdir(SOURCE_ROOT)
        if os.path.isdir(os.path.join(SOURCE_ROOT, f))
    ]

    for folder_name in tqdm(folders, desc="Processing fonts"):
        folder_path = os.path.join(SOURCE_ROOT, folder_name)
        metadata_path = os.path.join(folder_path, "METADATA.pb")

        if not os.path.exists(metadata_path):
            continue

        scripts = extract_scripts(metadata_path)

        if not scripts:
            continue

        for script in scripts:
            script_dir = os.path.join(DEST_ROOT, script)
            os.makedirs(script_dir, exist_ok=True)
            destination_path = os.path.join(script_dir, folder_name)

            if not os.path.exists(destination_path):
                shutil.copytree(folder_path, destination_path)


def merge_scripts():
    merges = {
        "Kore": "Hang",
        "Hant": "Hani",
        "Hans": "Hani",
    }

    for source_script, target_script in merges.items():
        source_dir = os.path.join(DEST_ROOT, source_script)
        target_dir = os.path.join(DEST_ROOT, target_script)
        if not os.path.exists(source_dir):
            continue

        os.makedirs(target_dir, exist_ok=True)
        for item in os.listdir(source_dir):
            source_path = os.path.join(source_dir, item)
            target_path = os.path.join(target_dir, item)

            if os.path.exists(target_path):
                continue

            shutil.move(source_path, target_path)

        if not os.listdir(source_dir):
            os.rmdir(source_dir)


def duplicate_script_folders():
    SCRIPT_DUPLICATES = {
        "Hani": "Bopo",
        "Latn": "Zyyy",
        "Kana": "Jpan",
        "Hira": "Jpan",
    }

    for source_script, target_script in SCRIPT_DUPLICATES.items():
        source_dir = os.path.join(DEST_ROOT, source_script)

        if not os.path.exists(source_dir):
            continue

        target_dir = os.path.join(DEST_ROOT, target_script)
        os.makedirs(target_dir, exist_ok=True)

        for item in os.listdir(source_dir):
            source_path = os.path.join(source_dir, item)
            target_path = os.path.join(target_dir, item)

            if os.path.exists(target_path):
                continue

            shutil.copytree(source_path, target_path)


def main():
    download_and_extract()
    organize_fonts()
    merge_scripts()
    duplicate_script_folders()


if __name__ == "__main__":
    main()
