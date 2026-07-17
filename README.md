# BERTopic Sample Run 

**Sample Run Data:** Reddit wallstreetbets comments, January 27–28 2021 (Reddit outage days)  
**Sample size:** 30,000 comments  
**Pipeline:** UMAP (768 → 5 dims) → HDBSCAN → c-TF-IDF → SafeTensors  

## Results at a Glance

| Metric | Value |
|---|---|
| Topics discovered | 223 |
| Outlier comments (topic −1) | 4,661 (15.5 %) |
| A — Uncertainty Expressions | 6 candidate topics |
| B — Identity Framing | 19 candidate topics |
| C — Normative Pressure | 32 candidate topics |

## Output Files

| File | What it stores |
|---|---|
| `topic_embeddings.safetensors` | Topic centroids in 5D UMAP space (224 × 5) |
| `ctfidf.safetensors` | The c-TF-IDF matrix (word weights per topic) |
| `topics.json` | Topic keywords, labels, and document counts |
| `umap_reduced_sample.npy` | (30,000 × 5) reduced embeddings checkpoint |
| `topic_info_sample.csv` | All 223 topics with keywords, doc counts, and A/B/C flags |
| `wsb_topics_sample.csv` | Every comment with its topic ID and flag columns |
| `flagged_topic_samples_sample.csv` | Representative & top-upvoted comments per flagged topic |

## First-Pass Topic Flags

Flags are assigned at **topic level**, not comment level. A topic is flagged when ≥ 2 seed words from the category lexicon appear among its top-20 c-TF-IDF keywords. The category assignments should be confirmed by reading the representative comments in `flagged_topic_samples_sample.csv`.

### A — Uncertainty Expressions
*Expressions of uncertainty about what users were seeing.*  
6 candidate topics flagged. Examples: confusion around trading halts, questions about price behaviour, disbelief at platform outages.

### B — Identity Framing
*Identity signals: retail traders vs Wall Street insiders.*  
19 candidate topics flagged. Examples: ape vs suits framing, Citadel/Melvin references, class warfare language, retail vs institutional distinctions.

### C — Normative Pressure
*Calls to hold or buy, not sell, buy/hold encouragement and anti-sell pressure.*  
32 candidate topics flagged. Examples: diamond hands, HODL, anti-paper-hands sentiment, moon/rocket imagery, explicit "don't sell" calls.


