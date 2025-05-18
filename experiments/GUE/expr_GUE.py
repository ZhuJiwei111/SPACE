import numpy as np
import torch
from dataclasses import dataclass
import pandas as pd
import torch.distributed as dist
import random
import os
import json
from torch.utils.data import Dataset
import wandb
from torch import nn
from torch.utils.data import random_split
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertConfig,
)
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.insert(0, project_root)

from model.modeling_enformer import Enformer
from model.modeling_space import SpaceConfig, TrainingSpace


class GUEDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path, sep=",")
        self.data.columns = ["sequence", "label"]
        self.num_labels = len(self.data["label"].unique())
        self.sequences = []
        self.labels = []
        for i in range(len(self.data)):
            self.sequences.append(self.sequence_to_onehot(self.data.iloc[i]["sequence"]))
            self.labels.append(self.data.iloc[i]["label"])

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
        # sequence, labels = (
        #     self.data.iloc[idx]["sequence"],
        #     self.data.iloc[idx]["label"],
        # )
        # sequence = self.sequence_to_onehot(sequence)
        sequence, labels = self.sequences[idx], self.labels[idx]
        return {"sequence": sequence, "labels": labels}


class Fintune(nn.Module):
    def __init__(self, model_path, num_labels=2, target_length=3, dataset="splice/reconstructed"):
        super().__init__()
        self.target_length = target_length
        config = SpaceConfig.from_pretrained(os.path.join(model_path, "config.json"))
        if "enformer" in model_path:
            self.model = Enformer.from_pretrained(model_path, use_tf_gamma=False)
        else:
            self.species = "human"
            model = TrainingSpace(config)
            state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        
            new_state_dict = {}
            for key, value in state_dict.items():
                key = key.replace("enformer", "model")
                if key == "model.tracks.output.human.weight":
                    key = "model.heads.human.0.weight"
                if key == "model.tracks.output.human.bias":
                    key = "model.heads.human.0.bias"
                if key == "model.tracks.output.mouse.weight":
                    key = "model.heads.mouse.0.weight"
                if key == "model.tracks.output.mouse.bias":
                    key = "model.heads.mouse.0.bias"
                new_state_dict[key] = value
            state_dict = new_state_dict

            model.load_state_dict(state_dict)
            self.model = model.model
            # ? 是否要为新物种训练不同的gate和species embedding
            # 酵母菌和病毒需要重新train一个species embedding和各层的gate network
            # if "EMP" in dataset or "virus" in dataset:
            #     if "EMP" in dataset:
            #         self.species = "yeast"
            #     elif "virus" in dataset:
            #         self.species = "virus"
            #     # self.model.transformer.species_embedding[self.species] = nn.Parameter(torch.randn(1, 1, config.dim))
            #     for block in self.model.transformer.transformer:
            #         block.species_embedding[self.species] = nn.Parameter(torch.randn(1, 1, config.dim))
            #         block.feed_forward.gates[self.species] = nn.Sequential(
            #             nn.Linear(config.dim, config.species_num_experts), nn.LeakyReLU()
            #         )

            # elif "mouse" in dataset:
            #     self.species = "mouse"
            # else:
            #     self.species = "human"
        

        self.fc = nn.Linear(config.dim * 2, num_labels)
        self.loss = nn.CrossEntropyLoss()

    def comupute_loss(self, outputs, labels):
        return self.loss(outputs, labels)

    def forward(self, sequence, labels):
        outputs = self.model(
            sequence, return_only_embeddings=True, target_length=self.target_length, species=self.species
        )  # (bs, seq_len, dim)
        outputs = torch.sum(outputs, dim=1)  # (bs, dim)
        logits = self.fc(outputs)
        loss = self.comupute_loss(logits, labels)
        return {"loss": loss, "predictions": logits, "label_ids": labels}


@dataclass
class ModelArguments:
    # model_name_or_path: str = "/home/jiwei_zhu/disk/Enformer/enformer_ckpt/"
    model_name_or_path: str = "/home/jiwei_zhu/SPACE/results/Space_species_tracks/checkpoint-28000"


@dataclass
class DataTrainingArguments:
    parent_dir: str = "./experiments/benchmark/GUE"
    dataset_name: str = "EMP/H3"


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


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = preds.argmax(-1)
    mcc = matthews_corrcoef(labels, preds)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"mcc": mcc, "accuracy": acc, "f1":f1}


def get_data_path(parent_dir, dataset_name):
    train_path = os.path.join(parent_dir, dataset_name, "train_new.csv")
    valid_path = os.path.join(parent_dir, dataset_name, "dev_new.csv")
    test_path = os.path.join(parent_dir, dataset_name, "test_new.csv")

    return train_path, valid_path, test_path


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
    train_path, valid_path, test_path = get_data_path(
        data_args.parent_dir, data_args.dataset_name
    )
    train_dataset = GUEDataset(train_path)
    valid_dataset = GUEDataset(valid_path)
    test_dataset = GUEDataset(test_path)
    num_labels = test_dataset.num_labels
    target_length = test_dataset[0]["sequence"].shape[0] // 128 + 1
    # load model
    model = Fintune(
        model_args.model_name_or_path,
        num_labels=num_labels,
        target_length=target_length,
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

    # train
    # excute "pip uninstall triton" before running the following line
    # os.system("pip uninstall triton")
    trainer.train()

    # predict
    model.eval()
    with torch.no_grad():
        results = trainer.predict(test_dataset)
    print(results.metrics)
    # if we have wandb and the current rank is 0, log the results
    wandb.log(results.metrics)
    results.metrics["task"] = data_args.dataset_name
    with open(f"{training_args.output_dir}.json", "w") as file:
        json.dump(results.metrics, file, indent=4)


if __name__ == "__main__":
    main()
