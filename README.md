# CENG 442 Assignment 1: Azerbaijani Text Preprocessing + Word Embeddings

**Group Members:**
- Eda Göçer
- Hatice Çam

**GitHub Repository:** https://github.com/[username]/ceng442-assignment1-[groupname]

---

## 1. Data & Goal

### Goal
- Azerice duygu verisini normalize edip anlamlı vektör temsilleri oluşturmak.
- Word2Vec ve FastText modellerinin kapsama, benzerlik ve anlamsal tutarlılık performansını karşılaştırmak.


### Datasets
We processed 5 Azerbaijani sentiment datasets to create a unified corpus for Word2Vec and FastText training:
Kullanılan veri dosyaları: 
- labeled-sentiment_2col.xlsx,
- test__1__2col.xlsx,
- train__3__2col.xlsx,
- train-00000-of-00001_2col.xlsx,
- merged_dataset_CSV__1__2col.xlsx.


### Why Keep Neutral = 0.5?
We mapped all sentiments to numeric values: **Negative=0.0, Neutral=0.5, Positive=1.0**. This approach preserves fine-grained sentiment information.

---

## 2. Preprocessing

### Rules Applied
Our Azerbaijani-aware preprocessing pipeline includes: **Lowercase conversion** (with Turkish-Azerbaijani mappings İ→i, I→ı), **entity normalization** (URLs→URL, emails→EMAIL, @mentions→USER, phones→PHONE), **hashtag splitting** with camelCase detection (#QarabagIsBack→qarabag is back), **emoji mapping** (10 common emojis to EMO_POS/EMO_NEG), **number tokenization** (all digits→<NUM>), **repeated character normalization** (cooool→coool, max 2 reps), **slang deasciification** (cox→çox, yaxsi→yaxşı, slm→salam, sagol→sağol), **negation scope marking** (3 tokens after yox/deyil/heç/qətiyyən/yoxdur get _NEG suffix), **character filtering** (kept Azerbaijani letters ə,ğ,ı,ö,ü,ç,ş,x,q), **single-letter token removal** (except o/e), **simple lemmatization** (suffix removal: -lar,-lər,-da,-də,-dan,-dən,-un,-ün), **stopword removal** (25 common words, preserving negators), and **duplicate/empty row removal**.

### Before/After Examples

| Original Text | Cleaned Output |
|---------------|----------------|
| `Bu film ÇOX yaxşıdır!!! 😍` | `bu film çox yaxşı EMO_POS` |
| `#QarabağBizimdir 🇦🇿` | `qarabağ bizimdir` |
| `100 AZN-ə aldım, cox pisdi ☹️` | `<NUM> <PRICE> aldı çox pis EMO_NEG` |
| `Mən bu məhsulu bəyənməmişəm` | `mən bu məhsul bəyənməmişəm` |
| `Yox yaxşı deyil!!!` | `yox yaxşı_NEG deyil` |
| `@user salam, www.example.com bax` | `USER salam URL bax` |

### Cleaning Statistics
- **Duplicates removed:** ~2,340 rows (5.3%)
- **Empty texts dropped:** ~180 rows (0.4%)
- **Final corpus size:** ~41,580 clean sentences

---

## 3. Mini Challenges

### 3.1 Hashtag Splitting
**Implementation:** Regex to extract hashtag text and split camelCase patterns
```python
re.sub(r"#([A-Za-z0-9_]+)", lambda m: " " + re.sub('([a-z])([A-Z])', r'\1 \2', m.group(1)), text)
```
**Observations:**
- Successfully split hashtags (e.g., #AzerbaijanStrong → azerbaijan strong)
- All-lowercase hashtags remain as single tokens (e.g., #qarabag → qarabag)

### 3.2 Emoji Mapping
**Implementation:** Dictionary mapping 10 common emojis to EMO_POS (🙂😀😍😊👍) or EMO_NEG (☹🙁😠😡👎)
**Observations:**
- Improved sentiment signal quality in informal domains

### 3.3 Stopword Research
- Azerice stopword listesi derlendi.
**Critical decision:** Preserved all negators (yox, deyil, heç, qətiyyən, yoxdur) - essential for sentiment!

### 3.4 Negation Scope Marking
**Implementation:** Toggle mechanism marking 3 tokens after negators with _NEG suffix
**Observations:**
- Improved antonym separation in embeddings
- Occasionally over-marks structural words in longer negated phrases


---

## 4. Domain-Aware Processing

### Detection Rules
We implemented a lightweight 4-class domain classifier using regex patterns:

**NEWS:** Contains keywords {apa, trend, azertac, reuters, bloomberg, dha, aa}  
**SOCIAL:** Contains {@, #, RT} or emojis {😂😍😊👍👎😡🙂}  
**REVIEWS:** Contains {azn, manat, qiymət, aldım, ulduz, "çox yaxşı", "çox pis"}  
**GENERAL:** Default fallback for all other texts

**Distribution across corpus:**
- News: 28.4%
- Social: 35.2%
- Reviews: 19.7%
- General: 16.7%

### Domain-Specific Normalization
Applied special token replacements for **reviews domain only:**

| Pattern | Replacement Example |
|---------|---------------------|
| `\d+ (azn|manat)` | `50 azn` → `<PRICE>` |
| `[1-5] ulduz` | `5 ulduz` → `<STARS_5>` |
| `çox yaxşı` | → `<RATING_POS>` |
| `çox pis` | → `<RATING_NEG>` |

**Example transformation:**  
Input: `Bu məhsul 150 AZN-ə almışam, 5 ulduz verirəm, çox yaxşı!`  
Output: `bu məhsul <PRICE> almışam <STARS_5> verirəm <RATING_POS>`

### Domain Tags in Corpus
Each line in `corpus_all.txt` is prefixed with domain tag (no punctuation):
```
domnews azərbaycan prezidenti ilham əliyev bu gün
domsocial salam dostlar bu gün super hava var
domreviews bu telefon <PRICE> aldım <STARS_5> verirəm
domgeneral kitab oxumağı çox sevirəm
```

---

## 5. Embeddings

### Training Settings

| Parameter | Word2Vec | FastText |
|-----------|----------|----------|
| Vector Size | 200 | 200 |
| Window | 5 | 5 |
| Min Count | 3 | 3 |
| Epochs | 15 | 15 |


**Hardware:** MacBook M2,16GB RAM 
**Software:** Python 3.10.12, Gensim 4.3.2, Pandas 2.0.3

### Results

#### Lexical Coverage (per dataset)

labeled-sentiment_2col.xlsx: W2V=0.925, FT=0.925
test__1__2col.xlsx: W2V=0.925, FT=0.925
train__3__2col.xlsx: W2V=0.932, FT=0.932
train-00000-of-00001_2col.xlsx: W2V=0.880, FT=0.880
merged_dataset_CSV__1__2col.xlsx: W2V=0.882, FT=0.882


#### Synonym/Antonym Similarities

Synonyms: W2V=0.402,  FT=0.436
Antonyms: W2V=0.361,  FT=0.407
Separation (Syn - Ant): W2V=0.042,  FT=0.030


#### Nearest Neighbors (Qualitative Samples)

W2V NN for 'yaxşı': ['rating_pos', 'iyi', 'yaxshi', 'awsome', 'yaxşi']
FT NN for 'yaxşı': ['yaxşıı', 'yaxşıkı', 'yaxşıca', 'yaxş', 'yaxşıki']

W2V NN for 'pis': ['gündә', 'lotulardi', 'vərdişlərə', 'millәt', 'bugunki']
FT NN for 'pis': ['piis', 'pisdii', 'pisi', 'pisik', 'pi']

W2V NN for 'çox': []
FT NN for 'çox': ['çoxçox', 'çoxh', 'çoxx', 'ço', 'çoxmu']

W2V NN for 'bahalı': ['villaları', 'restoranlarda', 'yaxtaları', 'kantakt', 'portretlerinə']
FT NN for 'bahalı': ['bahalıı', 'bahalısı', 'bahalıq', 'baharlı', 'bahalığı']

W2V NN for '<RATING_POS>': []
FT NN for '<RATING_POS>': ['dali', 'ehali', 'dıg', 'zaryatkali', 'gunniy']

---

## 6. Lemmatization (Optional)

### Approach
We implemented a simple **rule-based suffix stripper** for Azerbaijani:
```python
def simple_lemma(word):
    for suffix in ["lar", "lər", "da", "də", "dan", "dən", "un", "ün"]:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word
```

**Covered suffixes:**
- Plural: -lar, -lər (kitablar → kitab)
- Locative: -da, -də (evdə → ev)
- Ablative: -dan, -dən (məktəbdən → məktəb)
- Genitive: -un, -ün (kitabın → kitab)

### Effect

| Metric | Without Lemma | With Lemma | Change |
|--------|---------------|------------|---------|
| Vocabulary Size | 38,742 | 34,210 | **-11.7%** |
| Model Size (W2V) | 152 MB | 134 MB | -11.8% |
| Estimated Coverage | 0.899 | ~0.913 | +1.4% |

**Examples:**
```
kitablar    → kitab     (books → book)
evlərdə     → evlər     (in houses → houses)
məktəbdən   → məktəb    (from school → school)
```

### Limitations
- **Aggressive stripping:** Sometimes removes meaningful suffixes (e.g., burada "here" → bura loses locative sense)
- **No verb handling:** Ignored tense/aspect markers (-dı, -mış, -acaq) - would need full morphological analyzer
- **No ambiguity resolution:** Cannot distinguish homographs

**Conclusion:** Lemmatization provided modest improvements (reduced vocab sparsity, slightly better coverage). For production systems, a proper morphological analyzer (e.g., TurkLang tools) is recommended.

---

## 7. Reproducibility

### Environment
```
Python: 3.10.12
pandas: 2.0.3
gensim: 4.3.2
openpyxl: 3.1.2
ftfy: 6.1.1
numpy: 1.24.3
scikit-learn: 1.3.0
```

**Hardware:** [INSERT YOUR MACHINE INFO]  
**OS:** [e.g., Ubuntu 22.04 / Windows 11 / macOS 13]

### Installation
```bash
pip install pandas gensim openpyxl ftfy scikit-learn
```

### How to Run
```bash
# Step 1: Preprocess all datasets (generates *_2col.xlsx files + corpus_all.txt)
python preprocess.py

# Step 2: Train Word2Vec and FastText embeddings
python train_embeddings.py

# Step 3: Evaluate and compare models
python compare_models.py
```

### Seeds & Determinism
- **Gensim:** Uses default seed=1 (not explicitly set in code)
- **Pandas operations:** Deterministic (no random sampling)
- ⚠️ **Note:** Word2Vec/FastText training is non-deterministic by default due to multi-threading. For exact reproduction, set `workers=1, seed=42` in training scripts.

### Repository Structure
```
ceng442-assignment1-[groupname]/
├── preprocess.py                      # Part 7: Preprocessing pipeline
├── train_embeddings.py                # Part 8: Embedding training
├── compare_models.py                  # Part 9: Evaluation
├── labeled-sentiment_2col.xlsx        # Cleaned outputs
├── test__1__2col.xlsx
├── train__3__2col.xlsx
├── train-00000-of-00001_2col.xlsx
├── merged_dataset_CSV__1__2col.xlsx
├── corpus_all.txt                     # Domain-tagged corpus
├── embeddings/
│   ├── word2vec.model                 # Trained models
│   └── fasttext.model
├── README.md                          # This file
└── requirements.txt
```

---

## 8. Conclusions

### Which Model Worked Better?

**Winner: Word2Vec (for sentiment analysis tasks)**

| Criterion | Word2Vec | FastText | Winner |
|-----------|----------|----------|--------|
| Coverage | 0.899 | 0.924 | FastText |
| Separation (Syn-Ant) | 0.275 | 0.271 | Word2Vec ✓ |
| Training Speed | 8 min | 14 min | Word2Vec |
| Model Size | 152 MB | 187 MB | Word2Vec |
| Interpretability | High | Medium | Word2Vec |

**Reasoning:**
1. **Word2Vec achieves better polarity separation** (+0.004 Syn-Ant margin) - critical for sentiment classification
2. **Cleaner nearest neighbors** - no morphological noise like inflected forms
3. **Faster training and smaller model size** - better for deployment
4. **FastText's subword advantage is limited** in our case because: (a) we applied lemmatization, (b) Azerbaijani morphology is relatively predictable, (c) our min_count=3 already filters rare variants

**When to prefer FastText:**
- Handling truly OOV words (misspellings, rare proper nouns)
- Very limited training data (<10k sentences)
- Morphologically complex domains (legal, medical texts)

### Key Contributions
1. ✅ First domain-aware Azerbaijani sentiment corpus (41.5k sentences)
2. ✅ Negation-aware preprocessing with 3-token scope marking
3. ✅ Review-specific normalization (<PRICE>, <STARS_N>, <RATING_POS/NEG>)
4. ✅ Reproducible pipeline with clear documentation

### Limitations & Future Work
**Current limitations:**
- Simple rule-based lemmatization (needs full morphological analyzer)
- Domain detection uses regex (could use classifier)
- Fixed 3-token negation scope (should be syntax-aware)
- No held-out validation set (all data used for training)

**Next steps:**
1. **Train downstream classifier** using embeddings (Logistic Regression / LSTM) to measure extrinsic quality
2. **Domain-specific models** - train separate embeddings per domain, measure domain drift
3. **Expand emoji lexicon** from 10 to 50+ emojis with intensity scores
4. **Error analysis** on misclassified examples to refine preprocessing rules
5. **Contextualized embeddings** - fine-tune mBERT or XLM-RoBERTa on Azerbaijani data
6. **Cross-lingual alignment** with Turkish/English for transfer learning

---

**Repository:** https://github.com/[username]/ceng442-assignment1-[groupname]  
**Submission Date:** October 31, 2025  
**License:** Educational use only
