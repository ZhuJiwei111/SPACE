import json

import pandas as pd

# 读取 JSON 文件
with open(
    "/home/jiwei_zhu/disk/Enformer/enformer_MoE/experiments/variant/outputs_multi_embedding_28000_new2.json",
    "r",
) as f:
    data = json.load(f)

# 将 JSON 数据转换为 DataFrame
df = pd.DataFrame(data).T

# 计算每个指标的均值
mean_values = df.mean()

print(mean_values)
