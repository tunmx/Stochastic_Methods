import pandas as pd
import numpy as np

# 替换为您的 Excel 文件路径
file_path = 'TaskWorksheets1.xlsx'
# 要搜索的字符串
search_string = 'Yan Jingyu'

# 读取 Excel 文件
df = pd.read_excel(file_path)

# 搜索第一列中的字符串
row_index = df[df[df.columns[0]] == search_string].index

# 检查是否找到字符串，并返回行号
if not row_index.empty:
    print(f"String found at row: {row_index[0] + 1}")  # +1 因为 DataFrame 是从0开始索引，而 Excel 是从1开始
else:
    print("String not found")
