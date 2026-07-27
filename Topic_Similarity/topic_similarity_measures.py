"""
File Name: topic_similarity_measures.py

Builds two full-corpus measures per flagged topic:

  1. Continuous   — cosine similarity of every comment in the full corpus to
                    the topic's centroid (mean of original 768-dim SBERT
                    embeddings of the topic's window-assigned members).
  2. Binary       — whether that similarity falls in the top 90th/95th/99th
                    percentile, where the thresholds are calibrated on the
                    two-week window Jan 21 - Jan 31 2021 and
                    then applied as fixed cutoffs across the full corpus.

Two passes over the full corpus:
  Pass 1 — restricted to the calibration window, computes per-topic
           90/95/99th percentile similarity thresholds.
  Pass 2 — full corpus, computes continuous similarity + binary flags,
           written incrementally to Parquet.
"""

import pathlib
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul.*")

# Paths
ROOT       = pathlib.Path(__file__).resolve().parent.parent
EMBEDDINGS = ROOT / "wsbEmbeddingsExpanded_float32.npy"
PARQUET    = ROOT / "redditWsbCommentsDf.parquet"
WINDOW_DIR = pathlib.Path(__file__).resolve().parent / "output" / "outage1_2week_window"
OUT        = pathlib.Path(__file__).resolve().parent / "output" / "topic_similarity_measures_2week_window"
OUT.mkdir(parents=True, exist_ok=True)

TOPIC_INFO_CSV = WINDOW_DIR / "topic_info.csv"
WSB_TOPICS_CSV = WINDOW_DIR / "wsb_topics.csv"

WINDOW_DATES = [
    "2021-01-21", "2021-01-22", "2021-01-23", "2021-01-24", "2021-01-25",
    "2021-01-26", "2021-01-27", "2021-01-28", "2021-01-29", "2021-01-30",
    "2021-01-31", "2021-02-01", "2021-02-02", "2021-02-03",
]

# 11-day percentile-calibration window
CALIBRATION_START = "2021-01-21"
CALIBRATION_END   = "2021-02-03"

CATEGORY_META = {
    "uncertainty_expressions": "flag_uncertainty",
    "identity_framing":        "flag_identity_framing",
    "normative_pressure":      "flag_normative_pressure",
}
PERCENTILES = [90, 95, 99]

CHUNK_SIZE = 200_000


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def main():
    # Load full-corpus embeddings 
    print(f"Loading embeddings: {EMBEDDINGS.name}")
    emb = np.load(EMBEDDINGS, mmap_mode="r")
    print(f"  Shape: {emb.shape}  dtype: {emb.dtype}")

    print(f"\nLoading parquet: {PARQUET.name}")
    df = pd.read_parquet(PARQUET, columns=["id", "body", "created_utc"])
    print(f"  Raw rows: {len(df):,}")

    df = df[~df["body"].isin(["[deleted]", "[removed]"])].reset_index(drop=True)
    print(f"  After filter: {len(df):,}")

    assert len(df) == emb.shape[0], (
        f"Row mismatch: parquet={len(df):,}, embeddings={emb.shape[0]:,}."
    )

    df["datetime"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
    df["date_str"] = df["datetime"].dt.date.astype(str)

    window_mask  = df["date_str"].isin(WINDOW_DATES)
    orig_indices = np.where(window_mask)[0]
    print(f"\nWindow docs (Jan 21 - Feb 3, 2-week window): {len(orig_indices):,}")

    wsb_topics = pd.read_csv(WSB_TOPICS_CSV, usecols=["id", "topic"])
    assert len(wsb_topics) == len(orig_indices), (
        f"wsb_topics.csv rows ({len(wsb_topics):,}) != recomputed window rows "
        f"({len(orig_indices):,}) — date filter or parquet filter drifted."
    )
    # Spot-check alignment: ids should match at a handful of positions.
    check_pos = np.linspace(0, len(orig_indices) - 1, num=5, dtype=int)
    for p in check_pos:
        assert df.loc[orig_indices[p], "id"] == wsb_topics.loc[p, "id"], (
            f"Alignment check failed at position {p}."
        )
    print("  Alignment check passed (id spot-check).")

    window_topic = wsb_topics["topic"].to_numpy()

    # Build flagged-topic list 
    topic_info = pd.read_csv(TOPIC_INFO_CSV)
    flagged_pairs = []  # (topic_id, category)
    for category, flag_col in CATEGORY_META.items():
        flagged_ids = topic_info.loc[topic_info[flag_col] == True, "Topic"].astype(int).tolist()
        for tid in flagged_ids:
            flagged_pairs.append((tid, category))
    print(f"\nFlagged topic x category pairs: {len(flagged_pairs)}")

    unique_topic_ids = sorted({tid for tid, _ in flagged_pairs})
    print(f"Unique flagged topics: {len(unique_topic_ids)}")

    # Compute centroids 
    print("\nComputing topic centroids ...")
    centroid_rows = []
    member_counts = {}
    for tid in unique_topic_ids:
        member_mask = window_topic == tid
        n_members = int(member_mask.sum())
        member_counts[tid] = n_members
        member_emb_rows = orig_indices[member_mask]
        member_emb = np.array(emb[member_emb_rows], dtype=np.float32)
        centroid = member_emb.mean(axis=0)
        centroid_rows.append(centroid)
    C = l2_normalize(np.vstack(centroid_rows).astype(np.float32))  # (n_topics, 768)
    topic_id_to_row = {tid: i for i, tid in enumerate(unique_topic_ids)}
    print(f"  Centroid matrix: {C.shape}")

    columns = []  
    for tid, category in flagged_pairs:
        trow = topic_id_to_row[tid]
        base = f"{category}_topic_{tid}"
        columns.append((base, trow, "cont"))
        for p in PERCENTILES:
            columns.append((f"{base}_p{p}", trow, f"p{p}"))

    sidecar_rows = []
    for tid, category in flagged_pairs:
        sidecar_rows.append({
            "topic_id": tid,
            "category": category,
            "column_continuous": f"{category}_topic_{tid}",
            "n_window_members": member_counts[tid],
        })
    pd.DataFrame(sidecar_rows).to_csv(OUT / "topic_columns.csv", index=False)
    print(f"  Wrote {OUT / 'topic_columns.csv'}")

    # Pass 1: calibrate percentile thresholds on the 11-day window
    print(f"\nPass 1: calibrating percentile thresholds on "
          f"[{CALIBRATION_START}, {CALIBRATION_END}] ...")
    calib_mask = (df["date_str"] >= CALIBRATION_START) & (df["date_str"] <= CALIBRATION_END)
    calib_rows = np.where(calib_mask.to_numpy())[0]
    print(f"  Calibration-window rows: {len(calib_rows):,}")

    sim_accum = []
    for start in range(0, len(calib_rows), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(calib_rows))
        rows = calib_rows[start:end]
        chunk = np.array(emb[rows], dtype=np.float32)
        chunk = l2_normalize(chunk)
        sim_accum.append(chunk @ C.T)
        if (start // CHUNK_SIZE) % 10 == 0:
            print(f"    {end:>10,} / {len(calib_rows):,}")
    calib_sims = np.vstack(sim_accum)  # (n_calib_rows, n_topics)
    print(f"  Calibration similarity matrix: {calib_sims.shape}")

    thresholds = np.percentile(calib_sims, PERCENTILES, axis=0)  # (3, n_topics)
    threshold_table = {p: thresholds[i] for i, p in enumerate(PERCENTILES)}
    del sim_accum, calib_sims

    thr_records = []
    for tid in unique_topic_ids:
        trow = topic_id_to_row[tid]
        rec = {"topic_id": tid}
        for p in PERCENTILES:
            rec[f"p{p}_threshold"] = float(threshold_table[p][trow])
        thr_records.append(rec)
    pd.DataFrame(thr_records).to_csv(OUT / "topic_thresholds.csv", index=False)
    print(f"  Wrote {OUT / 'topic_thresholds.csv'}")

    # Pass 2: score full corpus
    print(f"\nPass 2: scoring full corpus ({len(df):,} rows) ...")
    parquet_path = OUT / "topic_similarity.parquet"

    ref_fields = [
        pa.field("id", pa.string()),
        pa.field("created_utc", pa.int64()),
        pa.field("date_str", pa.string()),
    ]
    col_fields = [
        pa.field(name, pa.bool_() if kind != "cont" else pa.float32())
        for name, _, kind in columns
    ]
    schema = pa.schema(ref_fields + col_fields)

    writer = pq.ParquetWriter(parquet_path, schema)
    n_total = len(df)
    try:
        for start in range(0, n_total, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n_total)
            chunk = np.array(emb[start:end], dtype=np.float32)
            chunk = l2_normalize(chunk)
            sims = chunk @ C.T  # (chunk_size, n_topics)

            arrays = [
                pa.array(df["id"].iloc[start:end]),
                pa.array(df["created_utc"].iloc[start:end]),
                pa.array(df["date_str"].iloc[start:end]),
            ]
            for name, trow, kind in columns:
                if kind == "cont":
                    arrays.append(pa.array(sims[:, trow], type=pa.float32()))
                else:
                    p = int(kind[1:])
                    thresh = threshold_table[p][trow]
                    arrays.append(pa.array(sims[:, trow] >= thresh, type=pa.bool_()))

            batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
            writer.write_batch(batch)

            if (start // CHUNK_SIZE) % 5 == 0:
                print(f"    {end:>10,} / {n_total:,}")
    finally:
        writer.close()

    print(f"\nWrote {parquet_path}")
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
