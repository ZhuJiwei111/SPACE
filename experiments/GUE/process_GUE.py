import os

import pandas as pd

path = "/home/jiwei_zhu/disk/Enformer/enformer_MoE/experiments/benchmark/GUE"
dataset = {
    "EMP": 500,
    "mouse": 100, 
    "prom": 300, 
    "prom_core": 70,
    "splice": 400, 
    "tf": 100,
    "virus": 1000
    }
for name, num in dataset.items():
    for dir in os.listdir(f"{path}/{name}"):
        dir = os.path.join(f"{path}/{name}", dir)
        for split in ["dev", "test", "train"]:
            df = pd.read_csv(f"{dir}/{split}.csv")
            length = len(df.loc[0,"sequence"])
            num = len(df)
            df = df[df["sequence"].apply(lambda x: len(x) == length)]
            df.to_csv(f"{dir}/{split}_new.csv", index=False)
            print(f"{length}: {num}->{len(df)}")