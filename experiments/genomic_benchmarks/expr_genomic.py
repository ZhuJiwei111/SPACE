import json
import os
import random
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from datasets import load_from_disk

# from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch import nn
from torch.utils.data import Dataset, random_split
from transformers import HfArgumentParser, Trainer, TrainingArguments

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.insert(0, project_root)


import wandb

from model.modeling_enformer import Enformer
from model.modeling_space import SpaceConfig, TrainingSpace


class GenomicDataset(Dataset):
    def __init__(self, file_path, split="train"):
        self.file_path = file_path
        self.data = load_from_disk(f"file://{file_path}")[split]
        self.data = self.data.with_format("torch")
        #! 对不等长的序列进行padding
        self.max_length = max(len(seq) for seq in self.data['seq'])

    def sequence_to_onehot(self, sequence):
        mapping = {
            "A": [1, 0, 0, 0],
            "C": [0, 1, 0, 0],
            "G": [0, 0, 1, 0],
            "T": [0, 0, 0, 1],
            "N": [0, 0, 0, 0],  # 默认编码
        }
        # 使用 mapping.get(base, mapping["N"])，如果 base 不在 mapping 中，则返回 N 的编码
        onehot = np.array(
            [mapping.get(base.upper(), mapping["N"]) for base in sequence], dtype=np.float32
        )
        return onehot

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        sequence = self.sequence_to_onehot(d["seq"])
        current_length = sequence.shape[0]
        pad_len = self.max_length - current_length
        
        # 在两侧填充N的编码（全零向量）
        if pad_len > 0:
            pad_left = pad_len // 2
            pad_right = pad_len - pad_left
            # 创建填充矩阵
            pad_left_arr = np.zeros((pad_left, 4), dtype=np.float32)
            pad_right_arr = np.zeros((pad_right, 4), dtype=np.float32)
            # 拼接填充和原始序列
            sequence = np.concatenate([pad_left_arr, sequence, pad_right_arr], axis=0)
        
        # 转换为PyTorch张量
        sequence = torch.from_numpy(sequence)  # 形状变为 (max_length, 4)
        labels = d["label"]
        return {"sequence": sequence, "labels": labels}


class Fintune(nn.Module):
    def __init__(self, model_path, num_labels=2, target_length=3, dataset="splice/reconstructed"):
        super().__init__()
        config = SpaceConfig.from_pretrained(os.path.join(model_path, "config.json"))
        self.target_length = target_length
        if "enformer" in  model_path:
            self.model = Enformer.from_pretrained(model_path, use_tf_gamma=False)
        else:
            model = TrainingSpace(config)
            state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
            model.load_state_dict(state_dict)
            self.model = model.model
            # ? 如何处理demo_human_or_worm人类和蠕虫分类任务
            # 果蝇需要重新train一个species embedding和各层的gate network
            if "drosophila" in dataset or "worm" in dataset:
                if "drosophila" in dataset:
                    self.species = "drosophila"
                elif "worm" in dataset:
                    self.species = "worm"
                self.model.transformer.species_embedding[self.species] = nn.Parameter(torch.randn(1, 1, config.dim))
                for block in self.model.transformer.transformer:
                    block.feed_forward.gates[self.species] = nn.Sequential(
                        nn.Linear(config.dim, config.species_num_experts), nn.LeakyReLU()
                    )

            elif "mouse" in dataset:
                self.species = "mouse"
            else:
                self.species = "human"
            
        self.fc = nn.Linear(config.dim * 2, num_labels)
        self.loss = nn.CrossEntropyLoss()

    def comupute_loss(self, outputs, labels):
        return self.loss(outputs, labels)

    def forward(self, sequence, labels):
        outputs = self.model(
            sequence, return_only_embeddings=True, target_length=sequence.shape[1]// 128 + 1, species=self.species
        )  # (bs, seq_len, dim)
        outputs = torch.sum(outputs, dim=1)  # (bs, dim)
        logits = self.fc(outputs)
        loss = self.comupute_loss(logits, labels)
        return {"loss": loss, "predictions": logits, "label_ids": labels}
    
    
@dataclass
class ModelArguments:
    model_name_or_path: str = "/home/jiwei_zhu/SPACE/results/Space_species_tracks/checkpoint"

@dataclass
class DataTrainingArguments:
    parent_dir: str = "../../datasets/genomic_benchmarks"
    dataset_name: str = "demo_coding_vs_intergenomic_seqs"
    
    
class TrainingArgs(TrainingArguments):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_strategy="steps"
        self.eval_strategy="steps"
        self.logging_steps=100
        self.save_steps=500 
        self.eval_steps=500
        self.save_total_limit=2
        self.max_steps=10000
        self.per_device_train_batch_size = 32
        self._n_gpu = 1
        self.save_safetensors = False
    

def accuracy_score(y_true, y_pred):
    """
    计算准确率
    :param y_true: 真实标签 (torch.Tensor)
    :param y_pred: 预测标签 (torch.Tensor)
    :return: 准确率 (float)
    """
    assert y_true.shape == y_pred.shape, "y_true和y_pred的形状必须一致"
    correct = (y_true == y_pred).sum().item()
    total = y_true.shape[0]
    return correct / total
   
def f1_score(y_true, y_pred, average='binary'):
    """
    计算F1分数
    :param y_true: 真实标签 (torch.Tensor)
    :param y_pred: 预测标签 (torch.Tensor)
    :param average: 计算方式 ('binary', 'micro', 'macro', 'weighted')
    :return: F1分数 (float)
    """
    assert y_true.shape == y_pred.shape, "y_true和y_pred的形状必须一致"
    
    # 计算混淆矩阵
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()
    
    # 计算precision和recall
    precision = tp / (tp + fp + 1e-10)  # 防止除零
    recall = tp / (tp + fn + 1e-10)     # 防止除零
    
    # 计算F1分数
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    return f1

     
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = preds.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1":f1}

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def main():
    # args
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # model_args = ModelArguments()
    # data_args = DataTrainingArguments()
    # training_args = TrainingArgs(output_dir="output_temp")

    print(f"Model arguments: {model_args}")
    print(f"Data arguments: {data_args}")
    print(f"Trainging arguments: {training_args}")
    # set seed
    set_seed(training_args.seed)

    # load dataset
    path = os.path.join(data_args.parent_dir, data_args.dataset_name)
    train_dataset = GenomicDataset(path, "train")
    test_dataset = GenomicDataset(path, "test")
    
    train_size = int(0.9 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])
    
    num_labels = 2 if data_args.dataset_name != "human_ensembl_regulatory" else 3
    # load model
    model = Fintune(
        model_args.model_name_or_path,
        num_labels=num_labels,
        dataset=data_args.dataset_name
    )
    with open(f"{training_args.output_dir}/model.txt", "w") as f:
        f.write(str(model))

    # set trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # predict
    model.eval()
    with torch.no_grad():
        results = trainer.predict(test_dataset)
    print(results.metrics)
    # if we have wandb and the current rank is 0, log the results
    wandb.log(results.metrics)
    results.metrics["task"] = data_args.dataset_name
    with open(f"{training_args.output_dir}/{data_args.dataset_name}.json", "w") as file:
        json.dump(results.metrics, file, indent=4)


if __name__ == "__main__":
    main()

