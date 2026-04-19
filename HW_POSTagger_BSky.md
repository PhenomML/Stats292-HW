# Stats 292 — Statistical Models of Text and Language
## Homework: Part-of-Speech Tagging via Hidden Markov Models

> **Course:** Stats 292, Prof. David Donoho, Spring 2026
> **Based on:** Manning & Schütze, *Foundations of Statistical Natural Language Processing*, Ch. 10 (§10.1–10.2); Jurafsky & Martin, *Speech and Language Processing*, 3rd ed., Ch. 8
> **Data sources:** NLTK Brown Corpus (Francis & Kučera 1964); BSky2GBQ Bluesky firehose archive

---

## About the Data

**Brown Corpus** (Henry Kučera & W. Nelson Francis, Brown University 1964): One million words of American English sampled from 500 texts across 15 genres — news, fiction, academic writing, government documents, and others. Each word is annotated with a part-of-speech tag. The corpus ships with NLTK and is the oldest and most widely studied English tagged corpus. We use the **Universal tagset** (Petrov, Das & McDonald 2012), which collapses the original 87 Brown tags into 12 coarse categories, making the transition matrix small enough to inspect and interpret.

| Tag | Meaning | Examples |
|---|---|---|
| NOUN | Noun | *dog, city, happiness* |
| VERB | Verb | *run, is, gone* |
| ADJ | Adjective | *red, big, happy* |
| ADV | Adverb | *quickly, never, well* |
| PRON | Pronoun | *I, he, they, who* |
| DET | Determiner | *the, a, this, each* |
| ADP | Adposition (preposition) | *in, of, on, for* |
| NUM | Number | *one, 42, first* |
| CONJ | Conjunction | *and, but, or* |
| PRT | Particle | *not, up, 's* |
| . | Punctuation | *. , : ?* |
| X | Other / foreign / unknown | *£, etc.* |

**Bluesky skeets:** The same 1% sample of English skeets from November 18–24, 2024 (~300,000 posts) used in previous homeworks. Bluesky text is informal and contains hashtags, @-mentions, emoji, neologisms, abbreviations, and deliberate non-standard spellings — a challenging test of a tagger trained on edited formal text.

---

## Overview

Part-of-speech (POS) tagging is the problem of assigning a syntactic category to each word in a sentence. It is the canonical application of Hidden Markov Models in NLP.

The HMM frame (Manning & Schütze §10.1; Jurafsky & Martin §8.2) has three components:

**States (hidden):** POS tags $t_1, t_2, \dots, t_n$.

**Observations (visible):** Words $w_1, w_2, \dots, w_n$.

**Three parameter matrices** estimated from a labeled corpus:

$$\pi_t = P(t_1 = t) \quad \text{(initial distribution)}$$

$$A_{t' \to t} = P(t_i = t \mid t_{i-1} = t') \quad \text{(transition matrix)}$$

$$B_{t \to w} = P(w_i = w \mid t_i = t) \quad \text{(emission matrix)}$$

Given an observed word sequence $\mathbf{w}$, the tagger finds the tag sequence that maximizes the joint probability:

$$\hat{\mathbf{t}} = \arg\max_{\mathbf{t}} P(\mathbf{t}, \mathbf{w}) = \arg\max_{\mathbf{t}} \pi_{t_1} \prod_{i=1}^{n} A_{t_{i-1} \to t_i} \cdot B_{t_i \to w_i}$$

In this homework you will:

1. Load the Brown Corpus and estimate $\pi$, $A$, and $B$ from training data (Manning & Schütze Figure 10.1).
2. Implement the Viterbi algorithm in log space to find $\hat{\mathbf{t}}$ (Manning & Schütze Figure 10.2).
3. Evaluate your tagger on held-out Brown sentences by per-tag accuracy and confusion matrix.
4. Add suffix-based OOV heuristics to handle words not seen during training.
5. Apply your tagger to Bluesky skeets, analyze the OOV rate, and characterize failure modes.

---

## Environment Setup

This homework runs in Jupyter under the `stats292` conda environment.
All packages — including `nltk` — are managed via `environment.yml`.

```bash
conda activate stats292
jupyter lab
```

The two NLTK corpora below are downloaded once to `~/nltk_data/` and cached for subsequent sessions.

```python
import nltk
nltk.download('brown',            quiet=True)
nltk.download('universal_tagset', quiet=True)
```

---

## Part 1 — Training the HMM (Manning & Schütze Figure 10.1)

### 1a. Load the Brown Corpus

```python
import nltk
from collections import defaultdict, Counter
import math

tagged_sentences = list(nltk.corpus.brown.tagged_sents(tagset='universal'))

print(f"Total sentences: {len(tagged_sentences):,}")
print(f"Total tokens:    {sum(len(s) for s in tagged_sentences):,}")
print(f"\nFirst sentence:")
for word, tag in tagged_sentences[0]:
    print(f"  {word:20s} {tag}")
```

### 1b. Train / Test Split

Hold out the last 20% of sentences for evaluation. Do not touch the test set until Part 3.

```python
import random
random.seed(42)

n = len(tagged_sentences)
split = int(0.8 * n)
train_sents = tagged_sentences[:split]
test_sents  = tagged_sentences[split:]

print(f"Train: {len(train_sents):,} sentences")
print(f"Test:  {len(test_sents):,} sentences")
```

### 1c. Count Frequencies

Accumulate the four counts needed to estimate $\pi$, $A$, and $B$.

```python
TAGS = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON',
        'DET',  'ADP',  'NUM', 'CONJ', 'PRT', '.', 'X']

tag_counts      = Counter()          # C(t)
start_counts    = Counter()          # C(t at position 0)
bigram_counts   = defaultdict(Counter)  # C(t' → t)
emission_counts = defaultdict(Counter)  # C(w | t)

for sent in train_sents:
    if not sent:
        continue
    words, tags = zip(*sent)
    words = [w.lower() for w in words]

    start_counts[tags[0]] += 1

    for i, (word, tag) in enumerate(zip(words, tags)):
        tag_counts[tag] += 1
        emission_counts[tag][word] += 1
        if i > 0:
            bigram_counts[tags[i-1]][tag] += 1

# Build training vocabulary
train_vocab = set(w.lower() for s in train_sents for w, t in s)

print(f"Training vocabulary: {len(train_vocab):,} word types")
print(f"\nTag frequencies:")
for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
    print(f"  {tag:6s}  {count:7,}")
```

### 1d. Estimate $\pi$, $A$, $B$ with Laplace Smoothing

Add-1 smoothing ensures no probability is zero, which prevents Viterbi paths from being killed by a single unseen event.

```python
num_sentences = len(train_sents)
num_tags      = len(TAGS)

# Initial probabilities π_t = P(t at position 0)
# Smoothed: (C(start=t) + 1) / (total_sentences + |T|)
start_p = {
    t: (start_counts[t] + 1) / (num_sentences + num_tags)
    for t in TAGS
}

# Transition probabilities A[t'][t] = P(t | t')
# Smoothed: (C(t' → t) + 1) / (C(t') + |T|)
trans_p = {}
for prev_tag in TAGS:
    total = sum(bigram_counts[prev_tag].values())
    trans_p[prev_tag] = {
        curr_tag: (bigram_counts[prev_tag][curr_tag] + 1) / (total + num_tags)
        for curr_tag in TAGS
    }

# Emission probabilities B[t][w] = P(w | t)
# Computed on demand (vocabulary too large to store in full matrix)
def emit_p(tag, word):
    """
    P(word | tag) with Laplace smoothing.
    For OOV words: returns a small uniform probability.
    """
    word = word.lower()
    count = emission_counts[tag].get(word, 0)
    total = tag_counts[tag]
    vocab_size = len(train_vocab)
    if count == 0:
        return 1.0 / (total + vocab_size + 1)   # OOV floor
    return (count + 1) / (total + vocab_size)

# Sanity check: emission probabilities for 'the'
print("P(w='the' | tag):")
for tag in TAGS:
    print(f"  {tag:6s}  {emit_p(tag, 'the'):.6f}")
```

### 1e. Inspect the Transition Matrix

The transition matrix $A$ captures the grammar of the language implicitly.

```python
import pandas as pd

A_df = pd.DataFrame(
    {curr: {prev: trans_p[prev][curr] for prev in TAGS} for curr in TAGS},
    index=TAGS, columns=TAGS
)
print("Transition matrix A[row=prev, col=curr] (rounded to 3 decimal places):")
print(A_df.round(3).to_string())
```

**Question 1.1.** Examine the transition matrix. Which tag most commonly follows a DET? Which tag most commonly follows VERB? Do these match your grammatical intuition about English?

**Question 1.2.** The start probability $\pi$ tells you how often each tag begins a sentence. Which three tags have the highest start probability? Does the Brown Corpus genre mix (news, fiction, academic) affect which tags start sentences?

**Question 1.3.** Find the transition $A_{t' \to t}$ with the highest probability. Find the transition with the second highest. What grammatical constructions do these encode?

---

## Part 2 — Viterbi Decoder (Manning & Schütze Figure 10.2)

### 2a. Log-Space Viterbi

Multiplying many small probabilities causes arithmetic underflow to zero. Work entirely in log space: replace multiplication with addition, and $\arg\max$ of products with $\arg\max$ of sums.

```python
def viterbi(words, tags, start_p, trans_p, emit_fn):
    """
    Viterbi decoding in log space.

    words:   list of observed word strings
    tags:    list of possible POS tags
    start_p: dict {tag: P(tag at position 0)}
    trans_p: dict of dicts {prev_tag: {curr_tag: P(curr | prev)}}
    emit_fn: callable (tag, word) -> P(word | tag)

    Returns: list of predicted tags, same length as words
    """
    n = len(words)
    if n == 0:
        return []

    # viterbi[t][tag] = best log-prob of any path ending at 'tag' at step t
    # backptr[t][tag] = which previous tag produced that best path
    viterbi_scores = [{}]
    backptr        = [{}]

    # Initialise at position 0
    for tag in tags:
        viterbi_scores[0][tag] = (math.log(start_p[tag]) +
                                  math.log(emit_fn(tag, words[0])))
        backptr[0][tag] = None

    # Recurrence
    for t in range(1, n):
        viterbi_scores.append({})
        backptr.append({})
        word = words[t]

        for curr_tag in tags:
            log_emit = math.log(emit_fn(curr_tag, word))
            best_prev, best_score = None, float('-inf')

            for prev_tag in tags:
                score = (viterbi_scores[t-1][prev_tag] +
                         math.log(trans_p[prev_tag][curr_tag]) +
                         log_emit)
                if score > best_score:
                    best_score = score
                    best_prev  = prev_tag

            viterbi_scores[t][curr_tag] = best_score
            backptr[t][curr_tag]        = best_prev

    # Termination: find best final tag
    best_final = max(viterbi_scores[n-1], key=viterbi_scores[n-1].get)

    # Backtrace
    path = [best_final]
    for t in range(n-1, 0, -1):
        path.append(backptr[t][path[-1]])
    path.reverse()
    return path
```

### 2b. Smoke Test

```python
test_sentences = [
    "The dog runs quickly in the park .".split(),
    "Stolen painting found by tree .".split(),
    "Juvenile court to try shooting defendant .".split(),
]

for words in test_sentences:
    predicted = viterbi(words, TAGS, start_p, trans_p, emit_p)
    print("Words: ", " ".join(f"{w:12s}" for w in words))
    print("Tags:  ", " ".join(f"{t:12s}" for t in predicted))
    print()
```

**Question 2.1.** "Stolen painting found by tree" and "Juvenile court to try shooting defendant" are two classic syntactic ambiguities. What does your tagger produce for each? What is the alternative parse? Why does the language model prior favor one reading over the other?

**Question 2.2.** In the Viterbi recurrence, each step considers all |T|² tag-pair combinations. What is the time complexity for a sentence of length n with |T| tags? What is the space complexity for storing the backpointer table?

---

## Part 3 — Evaluation on Brown Held-Out Set

### 3a. Per-Sentence Accuracy

```python
def tag_sentence(words, tags=TAGS):
    return viterbi(words, tags, start_p, trans_p, emit_p)

total_tokens   = 0
correct_tokens = 0
gold_tags_all  = []
pred_tags_all  = []

for sent in test_sents:
    if not sent:
        continue
    words = [w.lower() for w, t in sent]
    gold  = [t         for w, t in sent]

    pred = tag_sentence(words)

    gold_tags_all.extend(gold)
    pred_tags_all.extend(pred)
    correct_tokens += sum(g == p for g, p in zip(gold, pred))
    total_tokens   += len(gold)

accuracy = correct_tokens / total_tokens
print(f"Token accuracy on held-out Brown: {correct_tokens:,}/{total_tokens:,} = {accuracy:.4f}")
```

### 3b. Per-Tag Precision, Recall, F1

```python
from collections import defaultdict

per_tag = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

for g, p in zip(gold_tags_all, pred_tags_all):
    if g == p:
        per_tag[g]['tp'] += 1
    else:
        per_tag[g]['fn'] += 1
        per_tag[p]['fp'] += 1

print(f"\n{'Tag':6s}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Support':>8}")
print("-" * 45)
for tag in TAGS:
    tp = per_tag[tag]['tp']
    fp = per_tag[tag]['fp']
    fn = per_tag[tag]['fn']
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    supp = tp + fn
    print(f"{tag:6s}  {prec:6.3f}  {rec:6.3f}  {f1:6.3f}  {supp:8,}")
```

### 3c. Confusion Matrix

```python
import pandas as pd

conf = pd.DataFrame(0, index=TAGS, columns=TAGS)
for g, p in zip(gold_tags_all, pred_tags_all):
    conf.loc[g, p] += 1

print("\nConfusion matrix (rows=gold, cols=predicted):")
print(conf.to_string())

# Top confusions (off-diagonal)
errors = []
for g in TAGS:
    for p in TAGS:
        if g != p and conf.loc[g, p] > 0:
            errors.append((conf.loc[g, p], g, p))
errors.sort(reverse=True)

print("\nTop 10 confusions (gold → predicted, count):")
for count, g, p in errors[:10]:
    print(f"  {g} → {p}: {count:,}")
```

### 3d. Questions

**Question 3.1.** What is your overall token accuracy on the held-out Brown sentences? The rule-based tagger in NLTK (`nltk.tag.DefaultTagger` always predicting NOUN) achieves roughly 13% accuracy; a state-of-the-art modern tagger exceeds 97%. Where does your HMM fall, and why?

**Question 3.2.** Which tag has the lowest F1 score? Examine 10 examples of misclassified tokens for that tag. Is the error systematic (always confused with the same other tag) or scattered?

**Question 3.3.** Identify the single most common confusion (largest off-diagonal cell in the confusion matrix). Give three English words that frequently trigger this confusion and explain why the HMM gets them wrong.

---

## Part 4 — OOV Handling with Suffix Heuristics

Words not seen during training receive only the uniform OOV floor probability from `emit_p`. For Bluesky, where neologisms and informal spellings are common, this floor leaves a lot of information on the table. Suffix patterns are a strong signal.

### 4a. OOV Rate on Test Data

```python
oov_tokens = [(w.lower(), t)
              for sent in test_sents
              for w, t in sent
              if w.lower() not in train_vocab]

oov_total = len([t for sent in test_sents for w, t in sent])
print(f"OOV tokens in Brown test set: {len(oov_tokens):,} / {oov_total:,} "
      f"= {len(oov_tokens)/oov_total:.3f}")

# Tag distribution for OOV words
oov_tag_dist = Counter(t for _, t in oov_tokens)
print("\nGold tag distribution for OOV words:")
for tag, count in oov_tag_dist.most_common():
    print(f"  {tag:6s}  {count:5,}  ({count/len(oov_tokens):.3f})")
```

### 4b. Suffix Rule Table

| Suffix | Most likely tag | Example |
|---|---|---|
| `-ing` | VERB | *running, increasing* |
| `-ed` | VERB | *walked, improved* |
| `-ly` | ADV | *quickly, strongly* |
| `-tion`, `-sion` | NOUN | *nation, decision* |
| `-ity`, `-ty` | NOUN | *quality, beauty* |
| `-ous`, `-ious` | ADJ | *famous, previous* |
| `-able`, `-ible` | ADJ | *available, possible* |
| `-er`, `-or` | NOUN | *teacher, actor* |
| Capitalized (non-sentence-initial) | NOUN | *London, NASA* |
| All digits | NUM | *42, 1984* |

### 4c. Suffix-Aware Emission Function

```python
def suffix_tag_probs(word):
    """
    Returns a dict {tag: weight} reflecting suffix/shape priors.
    Weights are relative — they will be combined with corpus counts.
    Returns None if no suffix rule fires (fall through to default OOV floor).
    """
    w = word.lower()
    if w.isdigit():
        return {'NUM': 0.95, 'NOUN': 0.04, 'X': 0.01}
    if w.endswith(('tion', 'sion', 'ity', 'ty', 'er', 'or', 'ment', 'ness')):
        return {'NOUN': 0.75, 'VERB': 0.10, 'ADJ': 0.10, 'X': 0.05}
    if w.endswith(('ing',)):
        return {'VERB': 0.60, 'NOUN': 0.25, 'ADJ': 0.10, 'X': 0.05}
    if w.endswith(('ed',)):
        return {'VERB': 0.65, 'ADJ': 0.25, 'NOUN': 0.05, 'X': 0.05}
    if w.endswith(('ly',)):
        return {'ADV': 0.80, 'ADJ': 0.10, 'NOUN': 0.05, 'X': 0.05}
    if w.endswith(('ous', 'ious', 'able', 'ible', 'al', 'ful')):
        return {'ADJ': 0.80, 'NOUN': 0.10, 'ADV': 0.05, 'X': 0.05}
    return None

def emit_p_oov(tag, word, oov_floor=1e-6):
    """
    Emission probability that uses suffix heuristics for OOV words.
    For in-vocabulary words, falls back to corpus-estimated emit_p.
    """
    w = word.lower()
    if w in train_vocab:
        return emit_p(tag, word)

    priors = suffix_tag_probs(word)
    if priors is None:
        return oov_floor                  # no suffix fired: uniform floor
    return priors.get(tag, oov_floor)
```

### 4d. Re-Evaluate with OOV Heuristics

```python
correct_oov_before = sum(
    1 for sent in test_sents
    for w, g in sent
    if w.lower() not in train_vocab
    for p in [viterbi([tok.lower() for tok, _ in sent], TAGS,
                       start_p, trans_p, emit_p)[
                  [tok.lower() for tok, _ in sent].index(w.lower())]]
    if g == p
)

# Rebuild with improved emission function
def tag_sentence_oov(words):
    return viterbi(words, TAGS, start_p, trans_p, emit_p_oov)

correct_oov_after = 0
total_oov = 0
for sent in test_sents:
    words = [w.lower() for w, t in sent]
    gold  = [t         for w, t in sent]
    pred  = tag_sentence_oov(words)
    for w, g, p in zip(words, gold, pred):
        if w not in train_vocab:
            total_oov += 1
            if g == p:
                correct_oov_after += 1

print(f"OOV accuracy before suffix heuristics: see Part 3 confusion matrix")
print(f"OOV accuracy after  suffix heuristics: "
      f"{correct_oov_after:,}/{total_oov:,} = {correct_oov_after/total_oov:.4f}")
```

### 4e. Questions

**Question 4.1.** What fraction of Brown test-set OOV words are NOUNs? What fraction are VERBs? Does the suffix heuristic table above reflect this distribution, or does it over-weight any tag?

**Question 4.2.** How much does adding suffix heuristics improve OOV accuracy? Identify two suffixes that help the most and one that is misleading (fires on words whose gold tag is different from the heuristic's prediction).

**Question 4.3.** The capitalization heuristic (capitalized non-sentence-initial word → NOUN) works well for newswire but perhaps breaks on Twitter/Bluesky. Give two categories of capitalized words common in social media that are NOT nouns. Can you produce evidence from something in the data or your analysis that supports the idea that the heuristic "breaks"; or can you say that it still has value, based on your analysis or your own study of these data?

---

## Part 5 — Tagging Bluesky Skeets

### 5a. Load and Tokenize Skeets

```python
from google.cloud import bigquery
import re

client = bigquery.Client(project="stanford-f24-datasci-194d")

QUERY = """
SELECT text
FROM `stanford-f24-datasci-194d.EMS.bsky-firehose`
WHERE JSON_VALUE(post_json, '$.record.langs[0]') = 'en'
  AND DATE(timestamp) BETWEEN '2024-11-18' AND '2024-11-24'
  AND MOD(ABS(FARM_FINGERPRINT(CAST(sequence AS STRING))), 100) < 1
LIMIT 5000
"""

df_skeet = client.query(QUERY).to_dataframe()
print(f"Loaded {len(df_skeet):,} skeets")

def tokenize_skeet(text):
    """
    Light tokenization for Bluesky text.
    Strips URLs, keeps hashtags and @-mentions as single tokens.
    Splits on whitespace and punctuation, lowercases.
    """
    text = re.sub(r'https?://\S+', ' ', text)       # remove URLs
    tokens = re.findall(r'#\w+|@\w+|[a-zA-Z]+|\d+|[^\s\w]', text)
    return tokens

# Preview
for text in df_skeet['text'].dropna().head(5):
    print(f"Text:   {text[:100]}")
    print(f"Tokens: {tokenize_skeet(text)}")
    print()
```

### 5b. OOV Rate on Bluesky

```python
skeet_tokens = []
for text in df_skeet['text'].dropna():
    skeet_tokens.extend(tokenize_skeet(text))

skeet_vocab = set(t.lower() for t in skeet_tokens if t.isalpha())
oov_in_skeet = skeet_vocab - train_vocab

print(f"Unique word types in skeet sample:    {len(skeet_vocab):,}")
print(f"Types not in Brown training vocab:    {len(oov_in_skeet):,} "
      f"({len(oov_in_skeet)/len(skeet_vocab):.3f})")
print(f"\nSample OOV words:")
for w in sorted(oov_in_skeet)[:30]:
    print(f"  {w}")
```

### 5c. Tag a Sample of Skeets

```python
import random
random.seed(42)

sample_texts = df_skeet['text'].dropna().sample(200, random_state=42).tolist()
tagged_skeets = []

for text in sample_texts:
    tokens = tokenize_skeet(text)
    if not tokens:
        continue
    words = [t.lower() for t in tokens]
    try:
        tags = tag_sentence_oov(words)
    except Exception:
        tags = ['X'] * len(words)
    tagged_skeets.append(list(zip(tokens, tags)))

# Display first 10 tagged skeets
for tagged in tagged_skeets[:10]:
    print(" ".join(f"{w}/{t}" for w, t in tagged))
    print()
```

### 5d. Failure Mode Analysis

```python
# Collect tokens that were tagged X or whose tag seems suspicious
suspicious = []
hashtags    = []
mentions    = []

for tagged in tagged_skeets:
    for word, tag in tagged:
        if word.startswith('#'):
            hashtags.append((word, tag))
        elif word.startswith('@'):
            mentions.append((word, tag))
        elif tag == 'X':
            suspicious.append(word)

print(f"Hashtags found: {len(hashtags):,}")
print(f"  Tag distribution: {Counter(t for _, t in hashtags).most_common()}")
print(f"\n@-Mentions found: {len(mentions):,}")
print(f"  Tag distribution: {Counter(t for _, t in mentions).most_common()}")
print(f"\nTokens tagged X: {len(suspicious):,}")
print(f"  Sample: {suspicious[:20]}")
```

### 5e. Noun Phrase Density Comparison

A simple downstream application: how does POS tag distribution differ between formal text (Brown) and social media (Bluesky)?

```python
# Tag distribution in Brown test set (gold)
brown_dist = Counter(gold_tags_all)
brown_total = sum(brown_dist.values())

# Tag distribution in tagged skeets (predicted)
skeet_all_tags = [tag for sent in tagged_skeets for _, tag in sent]
skeet_dist = Counter(skeet_all_tags)
skeet_total = sum(skeet_dist.values())

print(f"{'Tag':6s}  {'Brown %':>8}  {'Bluesky %':>10}  {'Ratio':>6}")
print("-" * 40)
for tag in TAGS:
    b = brown_dist[tag] / brown_total
    s = skeet_dist[tag] / skeet_total if skeet_total > 0 else 0
    ratio = s / b if b > 0 else float('inf')
    print(f"{tag:6s}  {b:8.3f}  {s:10.3f}  {ratio:6.2f}")
```

### 5f. Questions

**Question 5.1.** What is the OOV rate (fraction of unique word types not in the Brown vocabulary) for your Bluesky sample? How does it compare to the OOV rate on Brown test sentences?

**Question 5.2.** What tag does your tagger most often assign to hashtags (e.g., `#blessed`, `#auspol`)? Is this linguistically defensible? What tag do @-mentions receive?

**Question 5.3.** Compare the NOUN and VERB proportions in Brown (gold) vs. Bluesky (predicted). Bluesky text tends to be more conversational. Does the tag distribution reflect this, and in which direction?

**Question 5.4.** Find 5 skeet tokens that your tagger clearly mislabels. For each, state the predicted tag, the likely correct tag, and the source of the error (OOV, ambiguity, domain shift, or tokenization).

---

## Part 6 — Error Analysis and Discussion

**Question 6.1.** The Brown Corpus was collected in 1964 from American English print media. Your tagger is applied to social media text from 2024. Name three specific word types or constructions that appear in Bluesky that did not exist in 1964, and explain how each stresses the HMM.

**Question 6.2.** The Viterbi algorithm finds the globally optimal tag sequence for the entire sentence. The greedy approach (always pick the locally best tag at each position) is simpler and faster. Under what circumstances would greedy tagging produce the same result as Viterbi? Under what circumstances would they diverge most dramatically?

**Question 6.3.** The HMM transition matrix encodes only first-order dependencies (each tag depends only on the previous tag). Second-order HMMs condition on the previous two tags. What is the number of parameters in the transition matrix for a 12-tag first-order model? For a second-order model? What are the statistical consequences of moving to second order on a corpus the size of Brown?

**Question 6.4.** Modern neural POS taggers (e.g., based on BERT or spaCy's transformer pipeline) exceed 98% accuracy on English newswire. Your HMM probably achieves 85–93%. Identify two specific structural limitations of the HMM that prevent it from reaching 98%, and describe how a contextual embedding model addresses each.

---

## Part 7 — Extension: Real-Word Error Detection (Optional)

The standard Viterbi tagger assumes every token is correctly spelled. A real-word error is a token that is spelled correctly but is the wrong word (e.g., writing *their* when *there* was intended). These are invisible to a spell checker.

POS context can catch some real-word errors: if the surrounding tags strongly imply that a NOUN is needed but the most probable tag for the current word is DET, that is a signal.

```python
def flag_real_word_errors(words, tags, threshold=0.1):
    """
    Flag positions where the winning tag probability is unexpectedly low
    given context — a simple proxy for real-word errors.

    Returns list of (position, word, tag, confidence) for suspect tokens.
    """
    # Re-run Viterbi, collecting marginal scores at each position
    # (simplified: compare top-1 and top-2 tag scores at each position)
    n = len(words)
    viterbi_scores = [{}]
    backptr        = [{}]

    for tag in TAGS:
        viterbi_scores[0][tag] = (math.log(start_p[tag]) +
                                  math.log(emit_p_oov(tag, words[0])))

    for t in range(1, n):
        viterbi_scores.append({})
        backptr.append({})
        word = words[t]
        for curr_tag in TAGS:
            log_emit = math.log(emit_p_oov(curr_tag, word))
            best_prev, best_score = None, float('-inf')
            for prev_tag in TAGS:
                score = (viterbi_scores[t-1][prev_tag] +
                         math.log(trans_p[prev_tag][curr_tag]) +
                         log_emit)
                if score > best_score:
                    best_score = score
                    best_prev  = prev_tag
            viterbi_scores[t][curr_tag] = best_score
            backptr[t][curr_tag]        = best_prev

    # At each position, compute the margin between top-1 and top-2 log-prob
    suspects = []
    for t in range(n):
        sorted_scores = sorted(viterbi_scores[t].values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 999
        if margin < threshold and words[t] in train_vocab:
            best_tag = max(viterbi_scores[t], key=viterbi_scores[t].get)
            suspects.append((t, words[t], best_tag, margin))
    return suspects

# Example
example = "there going to the store to buy food".split()
predicted = tag_sentence_oov(example)
suspects  = flag_real_word_errors(example, predicted)

print("Sentence:", " ".join(example))
print("Tags:    ", " ".join(predicted))
print("Suspects:", suspects)
```

**Question 7.1.** Does your real-word error detector flag `there` in the example above? What margin score does it receive? What would need to be true of the transition probabilities for the detector to reliably catch this specific error?

**Question 7.2.** How do your results on 7.1 change if you change the test phrase to `"they're going 2 tha store ta buy food"`?

---

## Deliverables

Submit this completed notebook with all cells executed:

1. The 12×12 transition matrix $A$ printed as a table.
2. Overall token accuracy on held-out Brown, plus per-tag precision/recall/F1.
3. The confusion matrix (12×12), with the top 5 off-diagonal entries identified and explained.
4. OOV accuracy before and after suffix heuristics.
5. Viterbi output on at least 5 Bluesky skeets, with the OOV rate and tag distribution comparison table.
6. Written answers to all numbered questions (2–4 sentences each).

---

## Ethics & Bias Reflection

**Question E1.** The Brown Corpus is American English, edited, compiled from the 1960's. The POS tagger you have constructed is trained using this corpus. Presumably this tagger is biased in the following manner. If used to tag Bluesky skeets, it may produce tag frequencies that resemble the frequencies the same tagger would produce on the Brown Corpus. Presumably any non-resemblance is ‘real’ and not bias, while any resemblance might be bias. How might you check whether some specific resemblance — say the tagging frequency of NOUN — is not bias?

**Question E2.** Modern neural POS taggers use drastically more compute and are trained on drastically more data than what we just did. What can you say (for example by doing outside reading) about the resource cost (compute, storage, human labelling) involved in constructing one of those taggers? What can you say about the value to society of spending such levels of resources? If the level of resources required to go from 80% accuracy to 98% is 10x, 100x, 1000x, 10000x, etc. At what point does it become ‘not ethical’ to pursue the project?
