# -*- coding: utf-8 -*-
"""
Gower-based clustering pipeline for mixed (numeric + categorical) vehicle dataset.

See top-of-file docstring for details.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

EXCEL_PATH = Path("/Users/yilin/Desktop/Python/Cluster_Analysis/vehicle_sales_2025_5_with_type.xlsx")
OUT_DIR = Path("./out_gower")
SAMPLE_MAX = 10000
K_RANGE = list(range(2, 10))
LINKAGE = "average"
RANDOM_SEED = 42
MDS_SUBSET = 2000


def extract_price_mean(s):
    if pd.isna(s):
        return np.nan
    s = str(s).replace("–", "-").replace("—", "-").replace("－", "-")
    nums = re.findall(r"\d+\.?\d*", s)
    if len(nums) >= 2:
        vals = list(map(float, nums[:2]))
        return sum(vals) / 2.0
    elif len(nums) == 1:
        return float(nums[0])
    else:
        return np.nan


def looks_like_city_or_level(name):
    name_l = str(name).lower()
    keys = ["城市",  "级别", "车型", "类别"]
    return any(k in name_l for k in keys)


def gower_distance(X_num: np.ndarray, X_cat: np.ndarray, num_ranges: np.ndarray) -> np.ndarray:
    n = X_num.shape[0] if X_num is not None else X_cat.shape[0]
    D = np.zeros((n, n), dtype=np.float32)

    # Numeric
    if X_num is not None and X_num.shape[1] > 0:
        safe_ranges = num_ranges.copy()
        safe_ranges[safe_ranges == 0] = 1.0
        for j in range(X_num.shape[1]):
            col = X_num[:, j]
            valid = ~np.isnan(col)
            rng_j = safe_ranges[j]
            for start in range(0, n, 512):
                end = min(n, start + 512)
                block = col[start:end]
                valid_block = valid[start:end][:, None] & valid[None, :]
                diff = np.abs(block[:, None] - col[None, :]) / rng_j
                diff[~valid_block] = 0.0
                D[start:end, :] += diff.astype(np.float32)
        W_num = np.zeros((n, n), dtype=np.float32)
        for j in range(X_num.shape[1]):
            col = X_num[:, j]
            valid = ~np.isnan(col)
            for start in range(0, n, 512):
                end = min(n, start + 512)
                valid_block = valid[start:end][:, None] & valid[None, :]
                W_num[start:end, :] += valid_block.astype(np.float32)
    else:
        W_num = np.zeros((n, n), dtype=np.float32)

    # Categorical
    if X_cat is not None and X_cat.shape[1] > 0:
        W_cat = np.zeros((n, n), dtype=np.float32)
        for j in range(X_cat.shape[1]):
            col = np.array(X_cat[:, j], dtype=object)
            notna = pd.Series(col).notna().values
            for start in range(0, n, 512):
                end = min(n, start + 512)
                block = col[start:end]
                notna_block = notna[start:end][:, None] & notna[None, :]
                eq = (block[:, None] == col[None, :])
                eq = np.where(notna_block, eq, False)
                dist_cat = (~eq).astype(np.float32)
                dist_cat[~notna_block] = 0.0
                D[start:end, :] += dist_cat
                W_cat[start:end, :] += notna_block.astype(np.float32)
    else:
        W_cat = np.zeros((n, n), dtype=np.float32)

    W = W_num + W_cat
    W[W == 0] = 1.0
    D = D / W

    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    return D


def classical_mds_from_distance(D: np.ndarray, max_points: int, seed=42):
    n = D.shape[0]
    rng = np.random.RandomState(seed)
    idx = np.arange(n) if n <= max_points else rng.choice(n, size=max_points, replace=False)
    D_sub = D[np.ix_(idx, idx)].astype(float)

    D2 = D_sub ** 2
    m = D_sub.shape[0]
    J = np.eye(m) - np.ones((m, m)) / m
    B = -0.5 * J.dot(D2).dot(J)

    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    pos = eigvals > 1e-10
    eigvals_pos = eigvals[pos][:2]
    eigvecs_pos = eigvecs[:, pos][:, :2]
    X2d = eigvecs_pos * np.sqrt(eigvals_pos)

    return idx, X2d


def main():
    assert EXCEL_PATH.exists(), f"Excel not found at {EXCEL_PATH}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(EXCEL_PATH)

    # price mean
    price_candidates = [c for c in df.columns if any(k in str(c) for k in ["价", "price", "Price", "售价", "指导价"])]
    created_price_mean = False
    for c in price_candidates:
        if df[c].dtype == object and df[c].astype(str).str.contains("-").any():
            df["price_mean_auto"] = df[c].apply(extract_price_mean)
            created_price_mean = True
            break

    # datetime -> ordinal
    datetime_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
    for c in datetime_cols:
        df[c + "_ordinal"] = pd.to_datetime(df[c], errors="coerce").map(lambda x: x.toordinal() if pd.notna(x) else np.nan)

    # features
    drop_like = {"id", "ID", "编号", "index", "索引"}
    cols = [c for c in df.columns if str(c) not in drop_like]

    numeric_cols = list(df[cols].select_dtypes(include=[np.number]).columns)
    if created_price_mean and "price_mean_auto" not in numeric_cols:
        numeric_cols.append("price_mean_auto")

    categorical_cols = [c for c in cols if (df[c].dtype == object or str(df[c].dtype).startswith("category")) and df[c].notna().any()]
    categorical_cols = [c for c in categorical_cols if (df[c].nunique(dropna=True) <= 200) or looks_like_city_or_level(c)]

    if len(numeric_cols) + len(categorical_cols) == 0:
        raise RuntimeError("No usable features detected. Please specify columns manually.")

    X_num_full = df[numeric_cols].astype(float).values if len(numeric_cols) > 0 else None
    X_cat_full = df[categorical_cols].values.astype(object) if len(categorical_cols) > 0 else None

    if X_num_full is not None and X_num_full.shape[1] > 0:
        num_min = np.nanmin(X_num_full, axis=0)
        num_max = np.nanmax(X_num_full, axis=0)
        num_ranges = num_max - num_min
        X_num_full = X_num_full - num_min
    else:
        num_ranges = np.array([])

    # sample
    N = len(df)
    rng = np.random.RandomState(RANDOM_SEED)
    sample_idx = np.arange(N) if N <= SAMPLE_MAX else rng.choice(N, size=SAMPLE_MAX, replace=False)
    df_sample = df.iloc[sample_idx].reset_index(drop=True)
    X_num = X_num_full[sample_idx] if X_num_full is not None else None
    X_cat = X_cat_full[sample_idx] if X_cat_full is not None else None

    # gower
    D = gower_distance(X_num, X_cat, num_ranges)
    np.save(OUT_DIR / "gower_distance_sample.npy", D)

    # choose K
    silhouette_records = []
    best_k, best_score, best_labels = None, -1.0, None
    for k in K_RANGE:
        try:
            model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        except TypeError:
            model = AgglomerativeClustering(n_clusters=k, affinity="precomputed", linkage="average")
        labels = model.fit_predict(D)
        score = silhouette_score(D, labels, metric="precomputed")
        silhouette_records.append({"k": k, "silhouette": score})
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    pd.DataFrame(silhouette_records).to_csv(OUT_DIR / "silhouette_scores.csv", index=False)

    # final labels & save
    df_sample["cluster"] = best_labels
    df_sample.to_csv(OUT_DIR / "cluster_assignments_sample.csv", index=False)

    # profiling
    if len(numeric_cols) > 0:
        num_summary = df_sample.groupby("cluster")[numeric_cols].mean().reset_index()
        num_summary.to_csv(OUT_DIR / "cluster_numeric_profile.csv", index=False)

    for c in categorical_cols:
        vc = (
            df_sample.groupby("cluster")[c]
            .value_counts(normalize=True)
            .rename("proportion")
            .reset_index()
        )
        vc["rank"] = vc.groupby("cluster")["proportion"].rank("dense", ascending=False)
        top3 = vc[vc["rank"] <= 3].drop(columns=["rank"])
        top3.to_csv(OUT_DIR / f"cluster_top_categories__{c}.csv", index=False)

    # classical MDS
    idx, X2d = classical_mds_from_distance(D, max_points=MDS_SUBSET, seed=RANDOM_SEED)
    labels_sub = df_sample.iloc[idx]["cluster"].values

    plt.figure()
    for cl in sorted(np.unique(labels_sub)):
        pts = X2d[labels_sub == cl]
        plt.scatter(pts[:, 0], pts[:, 1], label=f"Cluster {cl}")
    plt.title("Classical MDS (from Gower distance) — subset")
    plt.xlabel("MDS-1")
    plt.ylabel("MDS-2")
    plt.legend()
    plt.savefig(OUT_DIR / "gower_clusters_mds_classical_subset.png", dpi=150, bbox_inches="tight")
    plt.close()

    pd.DataFrame({"x": X2d[:, 0], "y": X2d[:, 1], "cluster": labels_sub}).to_csv(
        OUT_DIR / "mds_2d_coords_classical_subset.csv", index=False
    )

    print("=== Done ===")
    print(f"Rows total: {N}, sampled: {len(df_sample)}")
    print(f"Numeric features: {numeric_cols}")
    print(f"Categorical features: {categorical_cols}")
    print(f"Best K: {best_k}, silhouette: {best_score:.4f}")
    print("Artifacts saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
