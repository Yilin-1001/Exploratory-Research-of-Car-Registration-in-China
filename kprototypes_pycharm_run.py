"""
k-Prototypes 聚类分析 (适合 PyCharm 直接运行)
支持混合数据类型 + 自动新能源/燃油列
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from kmodes.kprototypes import KPrototypes


# ===================== CONFIG（参数配置） =====================
# Excel 数据文件路径（建议填绝对路径，避免工作目录变化导致找不到文件）
INPUT_EXCEL = "/Users/yilin/Desktop/Python/Cluster_Analysis/vehicle_sales_2025_5_with_type.xlsx"

# 结果输出目录（相对当前脚本运行目录）
OUTPUT_DIR = "./out_kproto"

# 尝试的簇数列表（会逐个训练并记录 cost，选 cost 最小的 K）
K_LIST = [ 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# ===================== FUNCTIONS（功能函数） =====================

def parse_price(df, col):
    """
    解析价格区间文本为均值：如 "9.98-12.98(万)" -> (9.98+12.98)/2
    如果是单值如 "10.58(万)"，则直接转 float；
    无法解析则返回 NaN。
    """
    def get_mean(price_str):
        if pd.isna(price_str):
            return np.nan
        s = str(price_str).replace("万", "")
        if "-" in s:
            try:
                low, high = s.split("-")
                return (float(low) + float(high)) / 2
            except:
                return np.nan
        else:
            try:
                return float(s)
            except:
                return np.nan
    df["price_mean_auto"] = df[col].apply(get_mean)
    return df


def preprocess(df):
    """
    预处理总流程：
    1) 解析价格区间 -> 数值列 price_mean_auto（如果存在“指导价区间”）
    2) 缺失值填充：数值列用中位数，类别列用众数
    3) 区分数值列 / 类别列
    4) 数值列用 MinMaxScaler 归一化到 [0,1]（包括“上牌量”）
    返回：df_prep(缩放后), numeric_cols, categorical_cols, df_raw(原始副本)
    """
    df = df.copy()
    df_raw = df.copy()  # ← 保留原始值（含“上牌量”）

    # 1) 若存在价格区间列，则转为均值（两份都一致）
    if "指导价区间" in df.columns:
        df = parse_price(df, "指导价区间")
        df_raw = parse_price(df_raw, "指导价区间")

    # 2) 缺失值（两份一致）
    for col in df.columns:
        if df[col].dtype == "object":
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            df_raw[col] = df_raw[col].fillna(mode_val)
        else:
            med_val = df[col].median()
            df[col] = df[col].fillna(med_val)
            df_raw[col] = df_raw[col].fillna(med_val)

    # 3) 类型划分
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # 4) 对所有数值列做归一化（包括“上牌量”）
    scaler = MinMaxScaler()
    if numeric_cols:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    df_prep = df  # 缩放后用于聚类
    return df_prep, numeric_cols, categorical_cols, df_raw

def run_kprototypes(df_prep, numeric_cols, categorical_cols):
    """
    训练 K-Prototypes：
    1) 拼接混合矩阵 X = [数值块 | 类别块]（顺序非常关键）
    2) 计算 categorical 索引 = 紧跟在数值块之后的列索引
    3) 遍历 K_LIST：fit_predict -> 记录 cost -> 选 cost 最小的 K
    返回：best_k, labels, model, cost_records
    """
    # 1) 固定拼接顺序，避免列位次错乱引发字符串转 float 错误
    X_num = df_prep[numeric_cols].to_numpy(dtype=float) if numeric_cols else np.empty((len(df_prep), 0))
    X_cat = df_prep[categorical_cols].astype(str).to_numpy() if categorical_cols else np.empty((len(df_prep), 0))
    X = np.concatenate([X_num, X_cat], axis=1)

    # 2) 类别列在 X 中的索引（从数值块长度开始，连续若干）
    cat_idx = list(range(X_num.shape[1], X_num.shape[1] + X_cat.shape[1]))

    # 健壮性检查（排查常见错误）
    assert np.issubdtype(X_num.dtype, np.floating), "X_num 不是浮点类型"
    assert X_cat.dtype.kind in ("U", "S", "O"), "X_cat 不是字符串/对象类型"
    assert all(0 <= i < X.shape[1] for i in cat_idx), "cat_idx 越界"

    cost_records = {}
    best_k, best_cost, best_labels, best_model = None, np.inf, None, None

    # 3) 多个 K 值尝试，选择 cost 最小的
    for k in K_LIST:
        print(f"尝试 K={k} ...")
        model = KPrototypes(
            n_clusters=k,      # 聚成 k 簇
            init="Huang",      # Huang 初始化，处理类别特征更稳定
            n_init=5,          # 多次随机初始化，取最好
            verbose=1,
            random_state=42
        )
        labels = model.fit_predict(X, categorical=cat_idx)  # 同时传入类别列索引
        cost_records[k] = float(model.cost_)
        print(f"K={k}, cost={cost_records[k]}")
        if cost_records[k] < best_cost:
            best_k, best_cost, best_labels, best_model = k, cost_records[k], labels, model

    return best_k, best_labels, best_model, cost_records


def save_outputs(df, labels, numeric_cols, categorical_cols, cost_records, best_k, model, df_raw):
    """
    导出：
    - 全量样本 + 聚类标签（仍输出当前 df，这里保持原样）
    - 数值特征簇均值（两份：scaled版 & mixed版(仅“上牌量”为原始单位)）
    - 类别Top-10（频数+占比）
    - K vs Cost & 元信息
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 追加簇标签（两份都加，便于使用）
    df = df.copy(); df["cluster"] = labels          # 缩放后的表（聚类用）
    df_raw = df_raw.copy(); df_raw["cluster"] = labels  # 原始表（导出用“上牌量”）

    # 全量分配结果（建议保存原始尺度那份，阅读更直观）
    df_raw.to_csv(os.path.join(OUTPUT_DIR, "kprototypes_cluster_assignments_full.csv"), index=False)

    # 1) 数值特征簇均值（全部都是缩放后的 0~1）
    numeric_profile_scaled = df.groupby("cluster")[numeric_cols].mean()
    numeric_profile_scaled.to_csv(os.path.join(OUTPUT_DIR, "kprototypes_cluster_numeric_profile_scaled.csv"))

    # 2) 数值特征簇均值（仅“上牌量”替换为原始单位）
    numeric_profile_mixed = numeric_profile_scaled.copy()
    if "上牌量" in numeric_cols:
        # 用原始df_raw计算“上牌量”的簇均值（原始单位）
        plate_raw_mean = df_raw.groupby("cluster")["上牌量"].mean()
        numeric_profile_mixed.loc[:, "上牌量"] = plate_raw_mean
    numeric_profile_mixed.to_csv(os.path.join(OUTPUT_DIR, "kprototypes_cluster_numeric_profile_mixed.csv"))

    # 3) 类别特征 Top-10（频数 + 占比）— 用原始表做计数更贴近业务
    for col in categorical_cols:
        topcats = (
            df_raw.groupby("cluster")[col]
                  .apply(lambda x: {
                      k: f"{v} ({v / len(x):.1%})"
                      for k, v in x.value_counts().head(10).items()
                  })
        )
        topcats.to_csv(os.path.join(OUTPUT_DIR, f"kprototypes_cluster_top_categories__{col}.csv"))

    # 4) K vs Cost
    pd.DataFrame({"K": list(cost_records.keys()), "Cost": list(cost_records.values())}).to_csv(
        os.path.join(OUTPUT_DIR, "kprototypes_cost_by_k.csv"), index=False
    )

    # 5) 曲线
    plt.plot(list(cost_records.keys()), list(cost_records.values()), marker="o")
    plt.xlabel("K"); plt.ylabel("Cost"); plt.title("K vs Cost (K-Prototypes)")
    plt.savefig(os.path.join(OUTPUT_DIR, "kprototypes_cost_curve.png")); plt.close()

    # 6) meta
    meta = {
        "best_k": best_k,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "cost_by_k": cost_records,
        "note": "聚类使用归一化数据；numeric_profile_mixed 中仅“上牌量”为原始单位"
    }
    with open(os.path.join(OUTPUT_DIR, "kprototypes_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ 最优 K={best_k}，已输出 scaled 与 mixed（上牌量原始单位）两份数值画像到 {OUTPUT_DIR}")


# ===================== MAIN（程序入口） =====================
if __name__ == "__main__":
    # 1) 读取 Excel 数据
    df = pd.read_excel(INPUT_EXCEL)

    # 2) 预处理（解析价格、时间转 ordinal、填补缺失、数值归一化、区分数值/类别列）
    df_prep, numeric_cols, categorical_cols, df_raw = preprocess(df)

    # 3) 训练 K-Prototypes（多 K 试探，选 cost 最小）
    best_k, labels, model, cost_records = run_kprototypes(df_prep, numeric_cols, categorical_cols)

    # 4) 导出所有结果文件
    save_outputs(df_prep, labels, numeric_cols, categorical_cols, cost_records, best_k, model, df_raw)