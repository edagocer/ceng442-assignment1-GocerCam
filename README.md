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
- labeled-sentiment.xlsx,
- test__1_.xlsx,
- train__3_.xlsx,
- train-00000-of-00001_.xlsx,
- merged_dataset_CSV__1_.xlsx.


### Why Keep Neutral = 0.5?
We mapped all sentiments to numeric values: **Negative=0.0, Neutral=0.5, Positive=1.0**. This approach preserves fine-grained sentiment information.

---

## 2. Preprocessing

### Rules Applied
Bu aşamada, verideki yazım farklılıkları, gereksiz karakterler ve duygusal göstergeler temizlenmiş, aynı zamanda Azerice dil özelliklerine uygun dönüştürmeler yapılmıştır.
Uygulanan temel adımlar:
-	Tüm metinlerin küçük harfe dönüştürülmesi (Azerice karakterler korunarak: İ→i, I→ı)
-	Boş ve yinelenen satırların silinmesi
-	URL, e-posta, kullanıcı adı, telefon gibi öğelerin özel etiketlerle (URL, EMAIL, USER, PHONE) değiştirilmesi
-	Emoji temizleme ve duygusal etiketlerle eşleştirme (😊 → EMO_POS, ☹️ → EMO_NEG)
-	Hashtag ayrıştırma (#QarabagIsBack → qarabag is back)
-	Gereksiz sembollerin kaldırılması
-	Rakamların <NUM> ile temsil edilmesi
-	Negasyon belirteçlerinin (yox, deyil, heç vb.) ardından gelen 3 kelimenin _NEG ekiyle işaretlenmesi

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

### Domain-Specific Normalization
Applied special token replacements for **reviews domain only:**

| Pattern | Replacement Example |
|---------|---------------------|
| `\d+ (azn|manat)` | `50 azn` → `<PRICE>` |
| `[1-5] ulduz` | `5 ulduz` → `<STARS_5>` |
| `çox yaxşı` | → `<RATING_POS>` |
| `çox pis` | → `<RATING_NEG>` |


### Domain Tags in Corpus
each line of the trained corpus was prefixed with a tag indicating its detected domain (e.g., domnews, domsocial, domreviews, domgeneral).
This helps the embedding model learn how the same words behave in different domain contexts.
Example:
```
domreviews <PRICE> çox yaxşı
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
```
labeled-sentiment_2col.xlsx: W2V=0.925, FT=0.925
test__1__2col.xlsx: W2V=0.925, FT=0.925
train__3__2col.xlsx: W2V=0.932, FT=0.932
train-00000-of-00001_2col.xlsx: W2V=0.880, FT=0.880
merged_dataset_CSV__1__2col.xlsx: W2V=0.882, FT=0.882
```

#### Synonym/Antonym Similarities
```
Synonyms: W2V=0.402,  FT=0.436
Antonyms: W2V=0.361,  FT=0.407
Separation (Syn - Ant): W2V=0.042,  FT=0.030
```

#### Nearest Neighbors (Qualitative Samples)
```
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
```
---

## 6. Lemmatization (Optional)



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

**Hardware:** MacBook Air M2 16GB RAM
**OS:** macOS 13

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


---

## 8. Conclusions

-	FastText outperformed Word2Vec in handling Azerbaijani morphology and unseen words.
-	Domain tagging and emoji normalization improved semantic coherence.
-	Retaining neutral (0.5) samples provided smoother transitions in sentiment space.
Next Steps:
-	Integrate contextual embeddings (e.g., BERT multilingual).
-	Fine-tune on each domain separately for domain-adaptive sentiment classification.

---

**Repository:** https://github.com/[username]/ceng442-assignment1-[groupname]  
**Submission Date:** October 31, 2025  
**License:** Educational use only
