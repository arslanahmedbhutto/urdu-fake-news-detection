"""Loading and normalising the Bend The Truth Urdu fake news corpus."""
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "data" / "raw" / "1.Corpus"
MODEL_PATH = REPO_ROOT / "model" / "fake_news_model.joblib"

CATEGORIES = {"bus": "Business", "hlth": "Health", "sbz": "Entertainment", "sp": "Sports", "tch": "Technology"}

# Arabic-script diacritics (zer, zabar, pesh, ...) are used inconsistently, so they are removed.
_DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭ]")
_NON_URDU = re.compile(r"[^؀-ۿ\s]")
_SPACES = re.compile(r"\s+")
# Unify letter variants that look the same but have different code points.
_VARIANTS = str.maketrans({"ي": "ی", "ك": "ک", "ى": "ی", "ۀ": "ہ"})


def raw_text(text: str) -> str:
    """Only collapses whitespace: keeps every character exactly as typed."""
    return _SPACES.sub(" ", text).strip()


def normalize(text: str) -> str:
    """Removes typing artifacts: unifies Arabic/Urdu letter variants (e.g. Arabic yeh U+064A -> Urdu yeh
    U+06CC), drops diacritics and collapses whitespace. Used by the deployed model."""
    text = text.translate(_VARIANTS)
    text = _DIACRITICS.sub("", text)
    return _SPACES.sub(" ", text).strip()


def urdu_letters_only(text: str) -> str:
    """Stricter version that keeps Urdu letters only (used to measure how much formatting cues matter)."""
    return _SPACES.sub(" ", _NON_URDU.sub(" ", normalize(text))).strip()


DATASET_URL = ("https://github.com/MaazAmjad/Datasets-for-Urdu-news/raw/master/"
               "Urdu%20Fake%20News%20Dataset.zip")


def download_corpus(dest: Path = CORPUS_DIR.parent) -> Path:
    """Downloads and unpacks the Bend The Truth corpus from the authors' GitHub repository (about 1 MB)."""
    import io
    import urllib.request
    import zipfile

    if (Path(dest) / "1.Corpus").exists():
        return Path(dest) / "1.Corpus"
    Path(dest).mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATASET_URL, timeout=60) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            if name.startswith("/") or ".." in name:
                raise ValueError(f"Unsafe path in archive: {name}")
        z.extractall(dest)
    return Path(dest) / "1.Corpus"


def _parse_name(filename: str):
    """'hlth12.txt' -> ('hlth', 'hlth12'). The stem identifies a real article and its fake rewrite."""
    stem = Path(filename).stem
    m = re.match(r"([a-z]+)(\d+)", stem)
    code = m.group(1) if m else stem
    return code, stem


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> pd.DataFrame:
    """One row per article with its split, label, category and pair id."""
    rows = []
    for split in ("Train", "Test"):
        for label in ("Real", "Fake"):
            folder = Path(corpus_dir) / split / label
            for f in sorted(folder.glob("*.txt")):
                code, stem = _parse_name(f.name)
                text = f.read_text(encoding="utf-8", errors="ignore")
                rows.append({
                    "split": split.lower(),
                    "label": label.lower(),
                    "category": CATEGORIES.get(code, code),
                    "pair_id": f"{split.lower()}-{stem}",
                    "file": f"{split}/{label}/{f.name}",
                    "text": text,
                })
    df = pd.DataFrame(rows)
    df["raw"] = df["text"].map(raw_text)
    df["clean"] = df["text"].map(normalize)
    df["n_words"] = df["clean"].str.split().str.len()
    df["y"] = (df["label"] == "fake").astype(int)
    return df
