"""
File Name: bertopic_pipeline.py
Date: 07/13/2026

Three topics under consideration
----------------------------------------
  A  UNCERTAINTY EXPRESSIONS  — expressions of uncertainty about what
                                 users were seeing
  B  IDENTITY FRAMING         — identity signals: retail traders vs Wall
                                 Street insiders, David vs Goliath framing
  C  NORMATIVE PRESSURE       — calls to hold or buy, not sell and this includes
                                 both buy/hold encouragement and explicit
                                 anti-sell pressure
-----------------------------------------

"""

import pathlib
import numpy as np
import pandas as pd

from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic

#Paths
ROOT          = pathlib.Path(__file__).resolve().parent.parent
EMBEDDINGS    = ROOT / "wsbEmbeddingsExpanded_float32.npy"
PARQUET       = ROOT / "redditWsbCommentsDf.parquet"
OUT           = pathlib.Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)

# Hyperparameters 

# UMAP
UMAP_FIT_SAMPLE   = 500_000   
UMAP_N_COMPONENTS = 5
UMAP_N_NEIGHBORS  = 15        # default
UMAP_MIN_DIST     = 0.0       
UMAP_METRIC       = "cosine"
SEED              = 42

# HDBSCAN 
HDBSCAN_MIN_CLUSTER = 150
HDBSCAN_MIN_SAMPLES = 10      

# c-TF-IDF vectorizer
NGRAM_RANGE = (1, 2)
MIN_DF      = 50
MAX_DF      = 0.95

TOP_WORDS       = 20    
MATCH_THRESHOLD = 2     

SEEDS = {
    # A — UNCERTAINTY EXPRESSIONS
    "uncertainty_expressions": {
        "confused", "confusion", "confusing", "understand", "understanding",
        "explain", "explanation", "wtf", "happened", "happening", "why",
        "what", "how", "know", "sure", "feel", "feels", "seems", "guess",
        "wonder", "wondering", "help", "lost", "question", "questions",
        "unclear", "anyone", "someone", "legitimate", "real", "fake",
        "crash", "glitch", "bug", "error", "down", "outage",
    },

    # B — IDENTITY FRAMING
    "identity_framing": {
        "retail", "retailer", "retailers",          
        "wall street", "wallstreet", "wall",        
        "hedge", "hedgefund", "hedgefunds", "hedgie", "hedgies",
        "citadel", "melvin", "kenny", "griffin",
        "institutional", "institution",
        "ape", "apes", "apeing", "suit", "suits",
        "us", "them", "they", "little", "guy", "guys",
        "david", "goliath", "rich", "billionaire", "billionaires",
        "rigged", "corrupt", "fair", "unfair",
        "manipulation", "manipulate", "class", "war", "fight",
        "power", "together", "community", "movement",
    },

    # C — NORMATIVE PRESSURE: added some emojis as well
    "normative_pressure": {
        "buy", "buying", "bought",                  
        "hold", "holding", "hodl", "hodling",
        "diamond", "hands", "💎", "🙌",
        "dont", "never", "not", "paper",            
        "sell", "selling", "sold",                  
        "moon", "rocket", "🚀", "tendies",
        "yolo", "squeeze",
        "strong", "belief", "believe", "trust", "faith",
        "line", "pressure", "push", "fight",
    },
}

# Load data 
print(f"  Loading embeddings: {EMBEDDINGS.name}")
emb = np.load(EMBEDDINGS, mmap_mode="r")
print(f"  Shape : {emb.shape}   dtype: {emb.dtype}   disk: {emb.nbytes/1e9:.1f} GB")

print(f"\n  Loading parquet: {PARQUET.name}")
df = pd.read_parquet(
    PARQUET,
    columns=["id", "body", "author", "score", "created_utc",
             "parent_id", "link_id"],
)
print(f"  Raw rows: {len(df):,}")

df = df[~df["body"].isin(["[deleted]", "[removed]"])].reset_index(drop=True)
print(f"  After filter: {len(df):,}")

assert len(df) == emb.shape[0], (
    f"Row mismatch: parquet={len(df):,}, embeddings={emb.shape[0]:,}. "
    "Re-check the filter condition."
)

df["datetime"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
docs = df["body"].tolist()
print(f"\n  {len(docs):,} documents ready.")

# UMAP 

umap_model = UMAP(
    n_components=UMAP_N_COMPONENTS,
    n_neighbors=UMAP_N_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    metric=UMAP_METRIC,
    low_memory=True,
    random_state=SEED,
    verbose=True,
)

rng      = np.random.default_rng(SEED)
fit_idx  = np.sort(rng.choice(len(docs), size=UMAP_FIT_SAMPLE, replace=False))
print(f"\n  Fitting UMAP on {UMAP_FIT_SAMPLE:,} sampled docs ...")
umap_model.fit(np.array(emb[fit_idx], dtype=np.float32))

print(f"\n  Transforming all {len(docs):,} docs in 100 K chunks ...")
CHUNK   = 100_000
reduced = []
for start in range(0, len(docs), CHUNK):
    end = min(start + CHUNK, len(docs))
    reduced.append(umap_model.transform(np.array(emb[start:end], dtype=np.float32)))
    if (start // CHUNK) % 10 == 0:
        print(f"    {end:>9,} / {len(docs):,}")

reduced = np.vstack(reduced).astype(np.float32)
print(f"\n  UMAP output shape: {reduced.shape}")

umap_ckpt = OUT / "umap_reduced.npy"
np.save(umap_ckpt, reduced)
print(f"  Saved checkpoint: {umap_ckpt}")

# HDBSCAN 
print(f"  min_cluster_size={HDBSCAN_MIN_CLUSTER}, min_samples={HDBSCAN_MIN_SAMPLES}")
print("  Cluster count and size inferred from data density.")

hdbscan_model = HDBSCAN(
    min_cluster_size=HDBSCAN_MIN_CLUSTER,
    min_samples=HDBSCAN_MIN_SAMPLES,
    metric="euclidean",            
    cluster_selection_method="eom",
    prediction_data=True,
    core_dist_n_jobs=-1,
)
hdbscan_model.fit(reduced)
labels = hdbscan_model.labels_

n_topics   = int(labels.max()) + 1
n_outliers = int((labels == -1).sum())
print(f"\n  Topics found     : {n_topics}")
print(f"  Outlier docs (-1): {n_outliers:,}  ({n_outliers/len(labels)*100:.1f} %)")

# c-TF-IDF 

vectorizer = CountVectorizer(
    stop_words="english",
    ngram_range=NGRAM_RANGE,
    min_df=MIN_DF,
    max_df=MAX_DF,
)

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    top_n_words=TOP_WORDS,
    calculate_probabilities=False,
    verbose=True,
)

topics, _ = topic_model.fit_transform(docs, embeddings=reduced)
# Here, intentionally passing the UMAP embeddings so that the results can be reused without rerunning the UMAP.

topic_info = topic_model.get_topic_info()
print(f"\n  Total topics: {len(topic_info) - 1}  (excl. outlier topic -1)")

print("FIRST-PASS FLAGGING")
print("  A: Uncertainty Expressions")
print("  B: Identity Framing")
print("  C: Normative Pressure")

CATEGORY_META = {
    "uncertainty_expressions": ("flag_uncertainty", "score_uncertainty",
                                "A — Uncertainty Expressions"),
    "identity_framing":        ("flag_identity_framing", "score_identity_framing",
                                "B — Identity Framing"),
    "normative_pressure":      ("flag_normative_pressure", "score_normative_pressure",
                                "C — Normative Pressure"),
}

def flag_topic(topic_id: int) -> dict:
    """
    Screen a topic against the three seed lexicons.
    Returns flag booleans and raw match counts.
    topic_id == -1 (outlier cluster) always returns all False.
    """
    result = {}
    for cat, seed_set in SEEDS.items():
        flag_col, score_col, _ = CATEGORY_META[cat]
        if topic_id == -1:
            result[flag_col]  = False
            result[score_col] = 0
            continue
        kw_tuples  = topic_model.get_topic(topic_id) or []
        kw_set     = {w.lower() for w, _ in kw_tuples}
        matches    = len(kw_set & seed_set)
        result[flag_col]  = matches >= MATCH_THRESHOLD
        result[score_col] = matches
    return result

flag_rows = []
for tid in topic_info["Topic"]:
    row = {"Topic": int(tid)}
    row.update(flag_topic(int(tid)))
    flag_rows.append(row)

flags_df   = pd.DataFrame(flag_rows)
topic_info = topic_info.merge(flags_df, on="Topic", how="left")

# Print candidate topics per category
for cat, (flag_col, score_col, label) in CATEGORY_META.items():
    flagged = topic_info[topic_info[flag_col] == True]
    print(f"\n  [{label}]  {len(flagged)} candidate topic(s):")
    for _, row in flagged.iterrows():
        kws = ", ".join([w for w, _ in (topic_model.get_topic(int(row["Topic"])) or [])[:8]])
        print(f"    Topic {int(row['Topic']):>4}  "
              f"({int(row['Count']):>6,} docs, {score_col}={int(row[score_col])})  →  {kws}")

# Review the representative comments
print("  Review these to confirm category assignment.")

repr_docs  = topic_model.representative_docs_   
N_REPR     = 5    
N_TOP      = 3    

all_flagged_cols = [meta[0] for meta in CATEGORY_META.values()]
flagged_topics   = topic_info[topic_info[all_flagged_cols].any(axis=1)]

sample_rows = []

for _, trow in flagged_topics.iterrows():
    tid        = int(trow["Topic"])
    which_cats = [label for cat, (flag_col, _, label) in CATEGORY_META.items()
                  if trow[flag_col]]
    kws        = ", ".join([w for w, _ in (topic_model.get_topic(tid) or [])[:10]])

    print(f"\n  ── Topic {tid}  |  {int(trow['Count']):,} docs")
    print(f"     Categories : {' | '.join(which_cats)}")
    print(f"     Keywords   : {kws}")

    # Representative documents closest to centroid
    centroid_docs = repr_docs.get(tid, [])[:N_REPR]
    print(f"     Centroid-closest comments ({len(centroid_docs)}):")
    for i, doc in enumerate(centroid_docs, 1):
        snippet = doc[:200].replace("\n", " ")
        print(f"       [{i}] {snippet}")
        sample_rows.append({"topic": tid, "category": " | ".join(which_cats),
                             "source": "centroid", "comment": doc})

    # Highest-upvoted comments in this topic
    topic_mask   = df["topic"] == tid
    top_comments = (df[topic_mask]
                    .nlargest(N_TOP, "score")[["body", "score", "datetime"]])
    print(f"     Highest-upvoted comments ({len(top_comments)}):")
    for _, cr in top_comments.iterrows():
        snippet = str(cr["body"])[:200].replace("\n", " ")
        print(f"       [score={cr['score']:>5}] {snippet}")
        sample_rows.append({"topic": tid, "category": " | ".join(which_cats),
                             "source": f"top_score_{cr['score']}",
                             "comment": str(cr["body"])})

# Save review table
samples_csv = OUT / "flagged_topic_samples.csv"
pd.DataFrame(sample_rows).to_csv(samples_csv, index=False)
print(f"\n  Review table saved : {samples_csv}")

# Save with SafeTensors

model_path = str(OUT / "bertopic_model")
topic_model.save(
    model_path,
    serialization="safetensors",
    save_ctfidf=True,
    save_embedding_model="sentence-transformers/all-mpnet-base-v2",
)
print(f"  BERTopic model : {model_path}/")

# Topic info table with flags
topic_csv = OUT / "topic_info.csv"
topic_info.to_csv(topic_csv, index=False)
print(f"  Topic info + flags           : {topic_csv}")

# Comment-level topic assignments with category flags
flag_lookup = flags_df.set_index("Topic")
for cat, (flag_col, _, _label) in CATEGORY_META.items():
    df[flag_col] = df["topic"].map(flag_lookup[flag_col]).fillna(False)

flag_cols    = [meta[0] for meta in CATEGORY_META.values()]
comments_csv = OUT / "wsb_topics.csv"
df[["id", "datetime", "author", "score", "topic"] + flag_cols].to_csv(
    comments_csv, index=False
)
print(f"  Comment-topic assignments    : {comments_csv}")

# Summary
print(f"  Total documents  : {len(docs):,}")
print(f"  Topics found     : {n_topics}")
print(f"  Outlier docs     : {n_outliers:,}  ({n_outliers/len(docs)*100:.1f} %)")
for cat, (flag_col, _, label) in CATEGORY_META.items():
    print(f"  {label:<40}: {int(topic_info[flag_col].sum())} candidate topics")

# Need to confirm assignments by reviewing flagged_topic_samples.csv

# print(f"    model = BERTopic.load('{model_path}')")
print("\n  Pipeline complete.")
