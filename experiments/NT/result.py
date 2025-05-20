import json
import os
from collections import defaultdict

import pandas as pd

# 定义要遍历的目录
base_dir = "./results"  # 当前目录

# 定义输出文件
output_file = "downstream_results.xlsx"
output_path = os.path.join(base_dir, output_file)

# 初始化数据结构
data = defaultdict(dict)

# 遍历所有目录
for dir_name in os.listdir(base_dir):
    # if dir_name.startswith("outputs_"):
    if True:
        results_path = os.path.join(base_dir, dir_name, "results.json")
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                results = json.load(f)
                for entry in results:
                    task = entry["task"]
                    # data[dir_name][f'{task}_loss'] = entry['test_loss']
                    data[dir_name][f"{task}_mcc"] = entry["test_mcc"]
                    # data[dir_name][f"{task}_accuracy"] = entry["test_accuracy"]

# 将数据转换为DataFrame
df = pd.DataFrame.from_dict(data, orient="index")

# 对列进行排序，确保每个task的loss、mcc、accuracy在一起
df = df.reindex(sorted(df.columns, reverse=True), axis=1)

# 创建MultiIndex列
columns = []
for col in df.columns:
    # 使用rsplit从右边开始分割，确保metric是最后一个下划线之后的部分
    task, metric = col.rsplit("_", 1)
    columns.append((task, metric))
df.columns = pd.MultiIndex.from_tuples(columns)

df = df.sort_index(axis=0)
# 保存为Excel文件
df.to_excel(output_path, index_label="output_dir")

print(f"Results have been combined and saved to {output_path}")
