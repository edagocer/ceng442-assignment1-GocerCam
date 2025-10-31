import pandas as pd, numpy as np
from gensim.models import Word2Vec, FastText
from pathlib import Path
import matplotlib.pyplot as plt


def lexical_coverage(model, tokens):
    vocab = model.wv.key_to_index
    return sum(1 for t in tokens if t in vocab) / max(1, len(tokens))

def read_tokens(fpath):
    df = pd.read_excel(fpath, usecols=["cleaned_text"])
    return [t for row in df["cleaned_text"].astype(str) for t in row.split()]

def pair_sim(model, pairs):
    vals=[]
    for a,b in pairs:
        try: vals.append(model.wv.similarity(a,b))
        except KeyError: pass
    return np.mean(vals) if vals else float("nan")

def neighbors(model, word, k=5):
    try: return [w for w,_ in model.wv.most_similar(word, topn=k)]
    except KeyError: return []

if __name__ == "__main__":
    w2v = Word2Vec.load("embeddings/word2vec.model")
    ft  = FastText.load("embeddings/fasttext.model")

    seed_words = ["yaxşı", "pis", "çox", "bahalı", "ucuz", "əla", "<PRICE>", "<RATING_POS>"]
    syn_pairs = [("yaxşı","əla"),("bahalı","qiymətli"),("ucuz","sərfəli"),("çox","mükəmməl")]
    ant_pairs = [("yaxşı","pis"),("bahalı","ucuz"),("çox","az"),("mükəmməl","dəhşət")]

    files = [
        "outputs/labeled-sentiment_2col.xlsx",
        "outputs/test__1__2col.xlsx",
        "outputs/train__3__2col.xlsx",
        "outputs/train-00000-of-00001_2col.xlsx",
        "outputs/merged_dataset_CSV__1__2col.xlsx"
    ]

    print("== Lexical coverage ==")
    for f in files:
        toks = read_tokens(f)
        cov_w2v = lexical_coverage(w2v, toks)
        cov_ft  = lexical_coverage(ft, toks)
        print(f"{Path(f).name}: W2V={cov_w2v:.3f}, FT={cov_ft:.3f}")

    syn_w2v, syn_ft = pair_sim(w2v, syn_pairs), pair_sim(ft, syn_pairs)
    ant_w2v, ant_ft = pair_sim(w2v, ant_pairs), pair_sim(ft, ant_pairs)

    print("\n== Similarity ==")
    print(f"Synonyms: W2V={syn_w2v:.3f}, FT={syn_ft:.3f}")
    print(f"Antonyms: W2V={ant_w2v:.3f}, FT={ant_ft:.3f}")
    print(f"Separation: W2V={syn_w2v-ant_w2v:.3f}, FT={syn_ft-ant_ft:.3f}")

    print("\n== Nearest neighbors ==")
    for w in seed_words:
        print(f" W2V NN for '{w}': {neighbors(w2v,w)}")
        print(f"  FT NN for '{w}': {neighbors(ft,w)}")
        print("-"*50)

    # --- Visualization: Synonym vs Antonym Similarity ---
    labels = ["Word2Vec", "FastText"]
    syn_vals = [syn_w2v, syn_ft]
    ant_vals = [ant_w2v, ant_ft]

    plt.figure(figsize=(6, 4))
    x = range(len(labels))
    plt.bar(x, syn_vals, width=0.4, label="Synonyms", align="center")
    plt.bar(x, ant_vals, width=0.4, label="Antonyms", align="edge")
    plt.xticks(x, labels)
    plt.ylabel("Cosine Similarity")
    plt.title("Synonym vs Antonym Similarity")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Visualization: Lexical Coverage per Dataset ---
    coverages_w2v, coverages_ft = [], []
    for f in files:
        toks = read_tokens(f)
        coverages_w2v.append(lexical_coverage(w2v, toks))
        coverages_ft.append(lexical_coverage(ft, toks))

    plt.figure(figsize=(9, 4))
    x = range(len(files))
    plt.bar(x, coverages_w2v, width=0.4, label="Word2Vec", align="center")
    plt.bar(x, coverages_ft, width=0.4, label="FastText", align="edge")
    plt.xticks(x, [Path(f).stem for f in files], rotation=40, ha="right")
    plt.ylabel("Coverage")
    plt.title("Lexical Coverage per Dataset")
    plt.legend()
    plt.tight_layout()
    plt.show()
