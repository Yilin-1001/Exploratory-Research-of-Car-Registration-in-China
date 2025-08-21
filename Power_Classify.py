import pandas as pd

# 读取数据
file_path = "/Users/yilin/Desktop/Python/Cluster_Analysis/vehicle_sales_2025_5.xlsx"
df = pd.read_excel(file_path)

# 1) 几乎全新能源的品牌
NEV_BRANDS = {
    "蔚来", "理想", "小鹏", "广汽埃安", "极氪", "哪吒", "零跑", "腾势", "岚图",
    "上汽智己", "阿维塔", "合创", "深蓝", "极越","AION","小米"
}

# 2) 车型关键词（新能源相关）
NEV_KEYWORDS = [
    "新能源", "纯电", "电动", "增程", "插电", "插混", "混动", "混合动力",
    "phev", "hev", "bev", "erev", "rev", "ev", "dm", "dm-i", "dm-p"
    "秦","元",'汉',"唐","宋","问界","智界","享界","尊界",
    # Common brand/model tokens (extend as needed)
    "比亚迪", "腾势", "岚图", "蔚来", "小鹏", "理想", "广汽埃安", "极氪", "哪吒", "零跑",
    "宏光mini ev", "mini ev", "海豚", "海鸥", "海豹", "秦", "汉", "元", "宋", "唐",
    "model 3", "model y", "AION","Aion"
]

# 3) 车型名→能源类型 的人工纠错映射
MODEL_FIXMAP = {
    "秦l": "新能源", "宋l": "新能源", "海豹06新能源": "新能源", "宏光miniev": "新能源",
    "su7": "新能源"
}


# 定义分类函数
def classify_car(model: str) -> str:
    name = str(model).lower()

    # 人工纠错优先
    for fix, label in MODEL_FIXMAP.items():
        if fix in name:
            return label

    # 品牌判断
    for brand in NEV_BRANDS:
        if brand.lower() in name:
            return "新能源"

    # 关键词判断
    for kw in NEV_KEYWORDS:
        if kw in name:
            return "新能源"

    return "燃油车"

# 新增一列分类结果
df["能源类型"] = df["车型"].apply(classify_car)

# 保存到新的Excel文件
output_path = "vehicle_sales_2025_5_with_type.xlsx"
df.to_excel(output_path, index=False)

print(f"已保存带分类的文件: {output_path}")