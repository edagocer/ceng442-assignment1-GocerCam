# CENG 442 Assignment 1: Azerbaijani Text Preprocessing + Word Embeddings

**Group Members:**
- Eda Göçer
- Hatice Çam

---

## 1. Data & Goal

### Goal
Build a domain-aware sentiment embedding space that represents both general and emotional contexts in Azerbaijani, and compare the coverage, similarity, and semantic consistency performance of Word2Vec and FastText models.

### Datasets
We processed 5 Azerbaijani sentiment datasets to create a unified corpus for Word2Vec and FastText training:
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
The preprocessing pipeline performs a series of Azerbaijani-specific normalization and cleaning steps to prepare text for sentiment analysis and embedding training.
-	Lowercasing & Normalization
  Unicode normalization (NFC) and Azerbaijani-specific casing (İ→i, I→ı).
  Fixes encoding glitches and HTML entities.
-	Noise Removal
  Removes HTML tags, URLs, emails, phone numbers, and user mentions.
  Replaces them with placeholders: <URL>, <EMAIL>, <PHONE>, <USER>.
-	Emoji and Hashtag Handling
  Emojis are mapped to sentiment placeholders:
    😊 😍 ❤️ 👍 → EMO_POS, 😡 😭 👎 💔 → EMO_NEG.
  Hashtags are split into words (e.g. #QarabagIsBack → qarabag is back).
-	Text Normalization
  Expands slang forms (e.g. slm → salam, yaxsi → yaxşı).
  Replaces numbers with <NUM>.
  Reduces repeated characters (çooox → çox).
  Keeps Azerbaijani letters (ə, ğ, ı, ö, ü, ç, ş) and removes unnecessary punctuation.
-	Negation Handling
  Detects negators (yox, deyil, heç, qətiyyən, yoxdur) and marks the next 3 tokens with _NEG.
-	Stopword Removal
  Removes frequent functional words (e.g., və, ilə, amma, ki, bu, biz, etc.)
  Keeps negators to preserve polarity (yox, deyil, heç, etc.).
-	Domain Detection & Normalization
  Simple rule-based domain detection:
    News → mentions apa, trend, reuters, etc.
  Social → emojis, hashtags, mentions.
    Reviews → words like qiymət, ulduz, çox yaxşı.
 	Adds placeholders for domain-specific patterns:
    <PRICE>, <STARS_4>, <RATING_POS>, <RATING_NEG>.
  Each line is prefixed with a tag such as domnews, domsocial, or domreviews.
-	Output & Cleaning
  Removes duplicates and empty rows.
 	Saves as two-column Excel files:
 	  (cleaned_text, sentiment_value) where sentiment_value ∈ {0.0, 0.5, 1.0} for Negative, Neutral, Positive.

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
**Implementation:** Dictionary mapping 26 common emojis to EMO_POS (🙂😀😍😊👍 and more) or EMO_NEG (☹🙁😠😡👎 and more)
**Observations:**
- Improved sentiment signal quality in informal domains

### 3.3 Stopword Research
- An Azerbaijani stopword list was compiled.
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
domreviews <POS_EMOJI> yemək çox dadlı idi
domgeneral prezident nitq söylədi

```

---

## 5. Embeddings

### Training Settings

| Parameter | Word2Vec | FastText |
|-----------|----------|----------|
| Vector Size | 300 | 300 |
| Window | 7 |75 |
| Min Count | 3 | 3 |
| Epochs | 30 | 30 |


**Hardware:** MacBook M2,16GB RAM 
**Software:** Python 3.10.12, Gensim 4.3.2, Pandas 2.0.3

### Results

#### Lexical Coverage (per dataset)
```
labeled-sentiment_2col.xlsx: W2V=0.925, FT=0.925
test__1__2col.xlsx: W2V=0.925, FT=0.925
train__3__2col.xlsx: W2V=0.932, FT=0.932
train-00000-of-00001_2col.xlsx: W2V=0.880, FT=0.880
merged_dataset_CSV__1__2col.xlsx: W2V=0.881, FT=0.881
```

#### Synonym/Antonym Similarities
```
Synonyms: W2V=0.220,  FT=0.324
Antonyms: W2V=0.246,  FT=0.310
Separation (Syn - Ant): W2V=-0.026,  FT=0.014
```

#### Nearest Neighbors (Qualitative Samples)
```
W2V NN for 'yaxşı': ['iyi', 'yaxwi', 'awsome', 'calxalamaq', 'yaxshi']
  FT NN for 'yaxşı': ['yaxşıı', 'yaxş', 'yaxşıkı', 'uaxşı', 'yaxwı']

 W2V NN for 'pis': ['gedər_neg', 'baktelecom_neg', 'gpon_neg', 'baxmaqa_neg', 'sakız_neg']
  FT NN for 'pis': ['pisə', 'piis', 'pisi', 'pis_neg', 'pisəm']

 W2V NN for 'çox': []
  FT NN for 'çox': ['çoxçox', 'çoxh', 'çoxx', 'ço', 'çoxmu']

 W2V NN for 'bahalı': ['kantakt', 'yaxtaları', 'təchizata', 'şanslısan', 'birləşmədir']
  FT NN for 'bahalı': ['bahalıı', 'bahalıq', 'pahalı', 'bahalısı', 'bahalığı']

 W2V NN for 'ucuz': ['sududu', 'satmalidi', 'satsaq', 'şeytanbazardan', 'yaşananların']
  FT NN for 'ucuz': ['ucuza', 'ucuzu', 'ucuzlawib', 'qiymətə', 'qiymətdi']

 W2V NN for 'əla': ['ela', 'proqramdır', 'kısaca', 'qəşəg', 'kamfort']
  FT NN for 'əla': ['ela', 'əlaa', 'əlaçi', 'əladıı', 'əladı']

 W2V NN for '<PRICE>': []
  FT NN for '<PRICE>': ['rinqdə', 'haunebu', 'reeceep', 'erddoogaann', 'niels']

 W2V NN for '<RATING_POS>': []
  FT NN for '<RATING_POS>': ['dali', 'hali', 'riali', 'bahali', 'kefim']
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
python3 -m venv .venv 
source .venv/bin/activate
pip install -r requirements.txt
```

### How to Run
```bash
# Step 1: Preprocess all datasets (generates *_2col.xlsx files + corpus_all.txt)
python preprocess.py

# Step 2: Train Word2Vec and FastText embeddings
python train_embeddings.py

# Step 3: Evaluate and compare models
python evaluate_embeddings.py
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

**Submission Date:** October 31, 2025  
**License:** Educational use only
