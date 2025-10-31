#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from pathlib import Path
import re, html, unicodedata
from ftfy import fix_text

# -------------------------------
# LOWERCASE (Azerbaijani specific)
# -------------------------------
def lower_az(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("I", "ı").replace("İ", "i")
    s = s.lower().replace("i̇", "i")
    return s

# -------------------------------
# REGEX + NORMALIZATION PATTERNS
# -------------------------------
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\+?\d[\d\-\s\(\)]{6,}\d")
USER_RE = re.compile(r"@\w+")
MULTI_PUNCT = re.compile(r"([!?.,;:])\1{1,}")
MULTI_SPACE = re.compile(r"\s+")
REPEAT_CHARS = re.compile(r"(.)\1{2,}", flags=re.UNICODE)

EMO_MAP = {
    # Positive
    "🙂": "EMO_POS", "😀": "EMO_POS", "😃": "EMO_POS", "😄": "EMO_POS", "😁": "EMO_POS",
    "😆": "EMO_POS", "😂": "EMO_POS", "🤣": "EMO_POS", "😍": "EMO_POS", "😊": "EMO_POS",
    "😉": "EMO_POS", "😎": "EMO_POS", "👍": "EMO_POS", "❤️": "EMO_POS", "💪": "EMO_POS",
    # Negative
    "☹": "EMO_NEG", "🙁": "EMO_NEG", "😞": "EMO_NEG", "😢": "EMO_NEG", "😭": "EMO_NEG",
    "😠": "EMO_NEG", "😡": "EMO_NEG", "👎": "EMO_NEG", "💔": "EMO_NEG", "😣": "EMO_NEG",
    "😤": "EMO_NEG"
}


SLANG_MAP = {"slm":"salam","tmm":"tamam","sagol":"sağol","cox":"çox","yaxsi":"yaxşı"}
NEGATORS = {"yox","deyil","heç","qətiyyən","yoxdur"}

def normalize_text_az(s: str) -> str:
    if not isinstance(s, str): return ""
    s = fix_text(html.unescape(s))
    s = HTML_TAG_RE.sub(" ", s)
    s = URL_RE.sub(" URL ", s)
    s = EMAIL_RE.sub(" EMAIL ", s)
    s = PHONE_RE.sub(" PHONE ", s)
    s = USER_RE.sub(" USER ", s)

    for emo, tag in EMO_MAP.items(): s = s.replace(emo, f" {tag} ")

    s = lower_az(s)
    s = MULTI_PUNCT.sub(r"\1", s)
    s = re.sub(r"\d+", " <NUM> ", s)
    s = re.sub(r"[^\w\s<>'əğıöşüçƏĞIİÖŞÜÇxqXQ]", " ", s)
    s = MULTI_SPACE.sub(" ", s).strip()

    toks, norm, mark_neg = s.split(), [], 0
    for t in toks:
        t = REPEAT_CHARS.sub(r"\1\1", t)
        t = SLANG_MAP.get(t, t)
        if t in NEGATORS:
            norm.append(t)
            mark_neg = 3
            continue
        if mark_neg > 0:
            norm.append(t + "_NEG")
            mark_neg -= 1
        else:
            norm.append(t)
    return " ".join(norm)

# -------------------------------
# DOMAIN DETECTION
# -------------------------------
NEWS_HINTS   = re.compile(r"\b(apa|trend|azertac|reuters|bloomberg|dha|aa)\b", re.I)
SOCIAL_HINTS = re.compile(r"\b(rt)\b|@|#|(?:😂|😍|😊|👍|👎|😡|🙂|🤣)")
REV_HINTS    = re.compile(r"\b(azn|manat|qiymət|aldım|ulduz|çox yaxşı|çox pis)\b", re.I)

def detect_domain(text: str) -> str:
    s = text.lower()
    if NEWS_HINTS.search(s): return "news"
    if SOCIAL_HINTS.search(s): return "social"
    if REV_HINTS.search(s): return "reviews"
    return "general"

PRICE_RE = re.compile(r"\b\d+\s*(azn|manat)\b", re.I)
STARS_RE = re.compile(r"\b([1-5])\s*ulduz\b", re.I)
POS_RATE = re.compile(r"\bçox yaxşı\b")
NEG_RATE = re.compile(r"\bçox pis\b")

def domain_specific_normalize(cleaned, domain):
    s = cleaned
    if domain == "reviews":
        s = PRICE_RE.sub(" <PRICE> ", s)
        s = STARS_RE.sub(lambda m: f" <STARS_{m.group(1)}> ", s)
        s = POS_RATE.sub(" <RATING_POS> ", s)
        s = NEG_RATE.sub(" <RATING_NEG> ", s)
        s = re.sub(r"\s+", " ", s).strip()
    return s

def add_domain_tag(line, domain):
    return f"dom{domain} " + line

def map_sentiment_value(v, scheme: str):
    if scheme == "binary":
        try: return 1.0 if int(v)==1 else 0.0
        except: return None
    s = str(v).strip().lower()
    if s in {"pos","positive","1","müsbət","good","pozitiv"}: return 1.0
    if s in {"neu","neutral","2","neytral"}: return 0.5
    if s in {"neg","negative","0","mənfi","bad","neqativ"}: return 0.0
    return None

def process_file(in_path, text_col, label_col, scheme, out_two_col_path, remove_stopwords=False):
    df = pd.read_excel(in_path)
    for c in ["Unnamed: 0", "index"]:
        if c in df.columns: df.drop(columns=[c], inplace=True)

    df = df.dropna(subset=[text_col])
    df = df[df[text_col].astype(str).str.strip().str.len() > 0]
    df = df.drop_duplicates(subset=[text_col])

    df["cleaned_text"] = df[text_col].astype(str).apply(normalize_text_az)
    df["__domain__"] = df[text_col].astype(str).apply(detect_domain)
    df["cleaned_text"] = df.apply(lambda r: domain_specific_normalize(r["cleaned_text"], r["__domain__"]), axis=1)

    if remove_stopwords:
        sw = set(["və","ilə","amma","ancaq","lakin","ya","həm","ki","bu","bir","o","biz","siz","mən","sən",
            "orada","burada","bütün","hər","artıq","çox","az","ən","də","da","üçün",
            "belə","ha","axı","yenə","onsuzda","demək","ammaa","vallah","bax","hə","aha"])
        for keep in ["deyil","yox","heç","qətiyyən","yoxdur"]:
            sw.discard(keep)
        df["cleaned_text"] = df["cleaned_text"].apply(lambda s: " ".join([t for t in s.split() if t not in sw]))

    df["sentiment_value"] = df[label_col].apply(lambda v: map_sentiment_value(v, scheme))
    df = df.dropna(subset=["sentiment_value"])
    df["sentiment_value"] = df["sentiment_value"].astype(float)

    domain_counts = df["__domain__"].value_counts(normalize=True).round(3)*100
    print(f"\nDomain distribution for {Path(in_path).name}:")
    for dom, pct in domain_counts.items():
        print(f"  - {dom:<8}: {pct:.1f}%")

    out_df = df[["cleaned_text", "sentiment_value"]]
    Path(out_two_col_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(out_two_col_path, index=False)
    print(f"Saved: {out_two_col_path} (rows={len(out_df)})")

def build_corpus_txt(input_dir="outputs", out_path="corpus_all.txt"):
    input_dir = Path(input_dir)
    files = list(input_dir.glob("*_2col.xlsx"))
    all_lines = []
    for f in files:
        df = pd.read_excel(f)
        df["__domain__"] = df["cleaned_text"].astype(str).apply(detect_domain)
        lines = [add_domain_tag(t, d) for t, d in zip(df["cleaned_text"], df["__domain__"]) if isinstance(t, str) and len(t.strip())>0]
        all_lines.extend(lines)
    Path(out_path).write_text("\n".join(all_lines), encoding="utf-8")
    print(f"Corpus saved as {out_path} ({len(all_lines)} lines)")

if __name__ == "__main__":
    data_dir = Path("data")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    CFG = [
        ("labeled-sentiment.xlsx", "text", "sentiment", "tri"),
        ("test__1_.xlsx", "text", "label", "binary"),
        ("train__3_.xlsx", "text", "label", "binary"),
        ("train-00000-of-00001.xlsx", "text", "labels", "tri"),
        ("merged_dataset_CSV__1_.xlsx", "text", "labels", "binary"),
    ]

    for fname, tcol, lcol, scheme in CFG:
        in_path = data_dir / fname
        out_path = output_dir / f"{Path(fname).stem}_2col.xlsx"
        process_file(in_path, tcol, lcol, scheme, out_path, remove_stopwords=True)

    build_corpus_txt("outputs", "corpus_all.txt")
