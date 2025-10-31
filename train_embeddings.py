from gensim.models import Word2Vec, FastText
from gensim.utils import simple_preprocess
from pathlib import Path
from time import time
from collections import Counter
import numpy as np

def read_corpus(corpus_path):
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            yield simple_preprocess(line, min_len=2, max_len=30)

def train_embeddings(corpus_path="corpus_all.txt", out_dir="embeddings"):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    sentences = list(read_corpus(corpus_path))
    print(f"Loaded {len(sentences)} sentences.")

    # Corpus stats
    all_tokens = [t for s in sentences for t in s]
    print(f"Total tokens: {len(all_tokens):,}")
    print(f"Unique tokens: {len(set(all_tokens)):,}")

    # Flatten all tokens
    all_tokens = [t for sent in sentences for t in sent]
    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens))
    avg_len = np.mean([len(s) for s in sentences])

    print(f"Total sentences: {len(sentences):,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Unique tokens: {unique_tokens:,}")
    print(f"Avg sentence length: {avg_len:.2f} tokens")

    # Top 20 most frequent tokens
    top_tokens = Counter(all_tokens).most_common(20)
    print("\nTop 20 most frequent tokens:")
    for tok, freq in top_tokens:
        print(f"  {tok:<15} {freq:,}")

    # Save corpus summary
    corpus_report = Path(out_dir) / "corpus_summary.txt"
    with open(corpus_report, "w", encoding="utf-8") as f:
        f.write("Corpus Statistics\n")
        f.write(f"Sentences: {len(sentences):,}\n")
        f.write(f"Total tokens: {total_tokens:,}\n")
        f.write(f"Unique tokens: {unique_tokens:,}\n")
        f.write(f"Avg sentence length: {avg_len:.2f}\n\n")
        f.write("Top 20 tokens:\n")
        for tok, freq in top_tokens:
            f.write(f"{tok:<15} {freq:,}\n")

    print(f"\nCorpus summary saved → {corpus_report}")

    # --- Train Word2Vec ---
    print("\nTraining Word2Vec...")
    t0 = time()
    w2v = Word2Vec(sentences, vector_size=300, window=7, min_count=3, workers=4, sg=1, epochs=30)
    print(f"✅ Word2Vec trained in {time()-t0:.2f}s")
    w2v.save(str(out_dir / "word2vec.model"))

    # --- Train FastText ---
    print("\nTraining FastText...")
    t0 = time()
    ft = FastText(sentences, vector_size=300, window=7, min_count=3, workers=4, sg=1, epochs=30)
    print(f"✅ FastText trained in {time()-t0:.2f}s")
    ft.save(str(out_dir / "fasttext.model"))

    print("\nModels saved to:", out_dir)

if __name__ == "__main__":
    train_embeddings("corpus_all.txt", "embeddings")
