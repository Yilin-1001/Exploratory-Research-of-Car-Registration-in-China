# -*- coding: utf-8 -*-
"""
探索型分析：关联规则分析 + 异常检测（适合 PyCharm 直接运行）

功能:
1) 关联规则（Apriori + association_rules）
   - 把类别特征做 one-hot
   - 以“上牌量是否高于中位数(或分位数)”作为目标标签之一
   - 导出强规则（支持度/置信度/提升度）

2) 异常检测（Isolation Forest + Local Outlier Factor）
   - 在数值特征空间中识别异常样本（可视为爆款/冷门/录入异常）
   - 导出按异常程度排序的清单

"""

import os
import re
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from mlxtend.frequent_patterns import apriori, association_rules


# ===================== 参数区（请按你的数据情况调整） =====================
# 数据文件路径（建议填绝对路径）
INPUT_EXCEL = "/Users/yilin/Desktop/Python/Cluster_Analysis/vehicle_sales_2025_5_with_type.xlsx"

# 输出目录
OUTPUT_DIR = "./out_explore"

# 关键列名（按你数据表实际列名来改）
COL_SALES = "上牌量"               # 数值，单位按你原表（不影响分析）
COL_PRICE_RANGE = "指导价区间"      # 如果存在，将解析为 price_mean_auto
COL_PRICE_MEAN = "price_mean_auto" # 解析后生成的价格均值（单位“万”，或按你的表）
COL_CITY = "城市"                  # 类别
COL_MODEL = "车型"                 # 类别
COL_SEG = "级别"                   # 类别（SUV/轿车/MPV...）
COL_ENERGY = "新能源标记"           # 类别（新能源/燃油），如没有可不填

# 参与关联规则的类别列（可按需增减）
CATEGORY_COLS_FOR_ARM = [COL_CITY, COL_MODEL, COL_SEG, COL_ENERGY]

# 参与异常检测的数值列（可按需增减；会自动从原始表取值）
NUMERIC_COLS_FOR_OD = [COL_SALES, COL_PRICE_MEAN]

# 关联规则参数
MIN_SUPPORT = 0.03      # 最小支持度
MIN_CONFIDENCE = 0.4    # 最小置信度
MIN_LIFT = 1.2          # 最小提升度
TOP_RULES = 200         # 最多导出多少条规则

# 高销量标签的阈值方式（"median" 用中位数；或使用分位数如 0.7）
HIGH_SALES_THRESHOLD_MODE = "median"  # "median" 或 "quantile"
HIGH_SALES_QUANTILE = 0.7             # 当 mode 为 "quantile" 时使用

# 异常检测参数
IF_CONTAMINATION = 0.05  # 预期异常比例（Isolation Forest）
# ========================================================================


def ensure_outdir(path: str):
    """确保输出目录存在"""
    os.makedirs(path, exist_ok=True)


def parse_price_mean(df: pd.DataFrame, col_price_range: str, out_col: str) -> pd.DataFrame:
    """
    将价格区间列解析为均值。例如：
    "9.98-12.98(万)" → (9.98+12.98)/2
    "10.58(万)" → 10.58
    """
    if col_price_range not in df.columns:
        return df

    def _extract_mean(s):
        if pd.isna(s):
            return np.nan
        txt = str(s)
        txt = txt.replace("万", "")
        # 统一短横线
        txt = txt.replace("－", "-").replace("—", "-").replace("–", "-")
        # 找到数字
        nums = re.findall(r"\d+\.?\d*", txt)
        nums = [float(x) for x in nums]
        if len(nums) >= 2:
            return (nums[0] + nums[1]) / 2.0
        elif len(nums) == 1:
            return nums[0]
        else:
            return np.nan

    df[out_col] = df[col_price_range].apply(_extract_mean)
    return df


def prepare_dataframe(path: str) -> pd.DataFrame:
    """读取并做基础清洗/派生"""
    df = pd.read_excel(path)

    # 解析价格均值
    if COL_PRICE_RANGE in df.columns:
        df = parse_price_mean(df, COL_PRICE_RANGE, COL_PRICE_MEAN)

    # 缺失值处理：类别→众数，数值→中位数
    for c in df.columns:
        if df[c].dtype == "object":
            if df[c].isna().any():
                mode_val = df[c].mode(dropna=True)
                if len(mode_val) > 0:
                    df[c] = df[c].fillna(mode_val[0])
                else:
                    df[c] = df[c].fillna("未知")
        else:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())

    return df


# ===================== 关联规则分析 =====================
def build_high_sales_label(df: pd.DataFrame, col_sales: str, mode="median", q=0.7) -> pd.Series:
    """构造高销量标签（布尔）：上牌量>=阈值"""
    if mode == "median":
        thr = df[col_sales].median()
    else:
        thr = df[col_sales].quantile(q)
    return (df[col_sales] >= thr).rename("高销量")


def association_rule_mining(df: pd.DataFrame) -> pd.DataFrame:
    """
    关联规则分析：
    - 类别列做 one-hot（包含高销量标签）
    - Apriori → 频繁项集
    - association_rules → 规则
    - 过滤支持度/置信度/提升度
    """
    # 构造高销量标签
    high_sales = build_high_sales_label(df, COL_SALES,
                                       mode=HIGH_SALES_THRESHOLD_MODE,
                                       q=HIGH_SALES_QUANTILE).astype(int)

    # 选择类别列（只取存在的）
    use_cats = [c for c in CATEGORY_COLS_FOR_ARM if c in df.columns]

    # one-hot 编码（含“高销量”标签）
    # 注意：为了让“高销量”为规则右部的结果，后续会筛选 consequents 包含该标签
    df_cat = pd.get_dummies(df[use_cats].astype(str), prefix=use_cats)
    df_cat["高销量=True"] = high_sales.values

    # 转换为布尔类型（避免 warning，也确保高销量列被纳入）
    df_cat = df_cat.astype(bool)

    # Apriori
    freq = apriori(df_cat, min_support=0.01, use_colnames=True)  # support 降到 1%
    rules = association_rules(freq, metric="lift", min_threshold=1.0)

    # 筛选
    rules = rules[
        (rules["support"] >= 0.01) &  # 降低阈值
        (rules["confidence"] >= 0.3) &  # 降低阈值
        (rules["lift"] >= 1.05)  # 放宽提升度要求
        ]
    rules = rules[rules["consequents"].astype(str).str.contains("高销量=True")]

    # 排序并裁剪
    rules = rules.sort_values(["lift", "confidence", "support"], ascending=False).head(TOP_RULES)

    # 美化展示：把 frozenset 转文本
    def set_to_text(s):
        if isinstance(s, (set, frozenset)):
            return " & ".join(sorted(list(s)))
        return str(s)

    rules["antecedents_txt"] = rules["antecedents"].apply(set_to_text)
    rules["consequents_txt"] = rules["consequents"].apply(set_to_text)

    # 调整列顺序
    show_cols = [
        "antecedents_txt", "consequents_txt",
        "support", "confidence", "lift",
        "leverage", "conviction"
    ]
    for c in show_cols:
        if c not in rules.columns:
            rules[c] = np.nan
    rules = rules[show_cols]

    return rules


# ===================== 异常检测 =====================
def anomaly_detection(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    异常检测：
    - 选择数值特征（NUMERIC_COLS_FOR_OD）
    - RobustScaler 标准化（对极值更稳健）
    - Isolation Forest 得分（score越小越异常，sklearn 的 decision_function 越低越异常）
    - LOF 得分（负值越小越异常）
    - 导出异常 TopN 清单
    """
    use_nums = [c for c in NUMERIC_COLS_FOR_OD if c in df.columns]
    if not use_nums:
        raise ValueError("NUMERIC_COLS_FOR_OD 中没有可用的数值列，请检查列名。")

    X = df[use_nums].copy()
    X = X.fillna(X.median())

    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)

    # Isolation Forest
    iso = IsolationForest(
        n_estimators=300,
        contamination=IF_CONTAMINATION,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(Xs)
    # decision_function 越低越异常；score_samples 越低越异常
    df_iso = df.copy()
    df_iso["IF_score"] = iso.decision_function(Xs)      # 正常点更高，异常更低
    df_iso["IF_pred"] = iso.predict(Xs)                 # 1 正常，-1 异常

    # LOF（注意：fit_predict 会返回 -1 异常，1 正常；negative_outlier_factor_ 越小越异常）
    lof = LocalOutlierFactor(
        n_neighbors=LOF_NEIGHBORS,
        contamination=IF_CONTAMINATION,
        novelty=False
    )
    lof_pred = lof.fit_predict(Xs)
    df_lof = df.copy()
    df_lof["LOF_pred"] = lof_pred
    df_lof["LOF_score"] = lof.negative_outlier_factor_  # 越小越异常

    # 合并两种方法结果（方便对照）
    merged = df.copy()
    merged["IF_score"] = df_iso["IF_score"]
    merged["IF_pred"] = df_iso["IF_pred"]
    merged["LOF_score"] = df_lof["LOF_score"]
    merged["LOF_pred"] = df_lof["LOF_pred"]

    # 取异常样本清单：IF 或 LOF 判为异常即可（可按需改成 AND 逻辑）
    anomalies = merged[(merged["IF_pred"] == -1) | (merged["LOF_pred"] == -1)].copy()

    # 排序：先按 IF_score 升序（越低越异常），再按 LOF_score 升序（越小越异常）
    anomalies = anomalies.sort_values(["IF_score", "LOF_score"], ascending=[True, True])

    return merged, anomalies


# ===================== 主流程 =====================
def main():
    ensure_outdir(OUTPUT_DIR)

    # 1) 读取数据 & 基础派生
    df = prepare_dataframe(INPUT_EXCEL)
    # 根据样本规模动态调整 LOF_NEIGHBORS
    global LOF_NEIGHBORS
    LOF_NEIGHBORS = max(35, int(len(df) * 0.02))

    # 2) 关联规则分析
    try:
        rules = association_rule_mining(df)
        rules_path = os.path.join(OUTPUT_DIR, "association_rules.xlsx")
        with pd.ExcelWriter(rules_path, engine="openpyxl") as w:
            rules.to_excel(w, index=False, sheet_name="rules")
        print(f"✅ 关联规则已导出：{rules_path}")
    except Exception as e:
        print("❌ 关联规则分析失败：", e)

    # 3) 异常检测
    try:
        merged_scores, anomalies = anomaly_detection(df)
        # 导出
        od_path = os.path.join(OUTPUT_DIR, "anomaly_detection.xlsx")
        with pd.ExcelWriter(od_path, engine="openpyxl") as w:
            merged_scores.to_excel(w, index=False, sheet_name="scores_all")
            anomalies.to_excel(w, index=False, sheet_name="anomalies_top")
        print(f"✅ 异常检测已导出：{od_path}")
    except Exception as e:
        print("❌ 异常检测失败：", e)

    # 4) 元信息记录
    meta = {
        "input_excel": INPUT_EXCEL,
        "category_cols_for_arm": CATEGORY_COLS_FOR_ARM,
        "numeric_cols_for_od": NUMERIC_COLS_FOR_OD,
        "arm_params": {
            "min_support": MIN_SUPPORT,
            "min_confidence": MIN_CONFIDENCE,
            "min_lift": MIN_LIFT,
            "top_rules": TOP_RULES,
            "high_sales_threshold_mode": HIGH_SALES_THRESHOLD_MODE,
            "high_sales_quantile": HIGH_SALES_QUANTILE
        },
        "od_params": {
            "if_contamination": IF_CONTAMINATION,
            "lof_neighbors": LOF_NEIGHBORS
        }
    }
    with open(os.path.join(OUTPUT_DIR, "explore_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=== Done ===")


if __name__ == "__main__":
    main()