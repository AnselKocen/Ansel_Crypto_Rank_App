import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
history_path = Path(BASE_DIR) / "df_merged_history.csv"  # 或者直接写完整路径
df = pd.read_csv(history_path)

# 确保日期列是 datetime 类型
df["date"] = pd.to_datetime(df["date"]).dt.date

# 过滤掉 2024-08-20
df = df[df["date"] != pd.to_datetime("2025-08-20").date()]

# 保存覆盖原文件
df.to_csv(history_path, index=False)

print("已删除 2024-08-20 的数据，其他不受影响。")
