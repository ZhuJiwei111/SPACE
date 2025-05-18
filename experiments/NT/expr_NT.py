import numpy as np
import torch
from dataclasses import dataclass
import pandas as pd
import torch.distributed as dist
import random
import json
from peft import IA3Config, get_peft_model
from torch.utils.data import Dataset
import wandb
from torch import nn
from torch.utils.data import random_split
from sklearn.metrics import matthews_corrcoef, accuracy_score
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


class NTDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path, sep=",")
        self.data.columns = ["chromosome", "start", "end", "category", "sequence"]
        self.num_labels = len(self.data["category"].unique())

    def sequence_to_onehot(self, sequence):
        mapping = {
            "A": [1, 0, 0, 0],
            "C": [0, 1, 0, 0],
            "G": [0, 0, 1, 0],
            "T": [0, 0, 0, 1],
            "N": [0, 0, 0, 0],
        }
        onehot = np.array([mapping[base] for base in sequence], dtype=np.float32)
        return onehot

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, labels = (
            self.data.iloc[idx]["sequence"],
            self.data.iloc[idx]["category"],
        )
        sequence = self.sequence_to_onehot(sequence)
        return {"sequence": sequence, "labels": labels}


class FintuneModel(nn.Module):
    def __init__(self, model_path, num_labels=2, target_length=3):
        super().__init__()
        self.target_length = target_length
        config = SpaceConfig.from_pretrained(os.path.join(model_path, "config.json"))
        if "enformer" in model_path:
            self.model = Enformer.from_pretrained(model_path, use_tf_gamma=False)
        else:
            config.target_length = target_length
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
            #! 使用随机species embedding和gate
            # self.model.transformer.species_embedding["human"] = nn.Parameter(torch.randn(1, 1, config.dim))
            # for block in self.model.transformer.transformer:
            #     block.feed_forward.gates["human"] = nn.Sequential(
            #         nn.Linear(config.dim, config.species_num_experts), nn.LeakyReLU()
            #     )
        
        self.fc = nn.Linear(config.dim * 2, num_labels)
        self.loss = nn.CrossEntropyLoss()

    def comupute_loss(self, outputs, labels):
        return self.loss(outputs, labels)

    def forward(self, sequence, labels):
        outputs = self.model(
            sequence, return_only_embeddings=True, target_length=self.target_length
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
    parent_dir: str = "./dataset"
    dataset_name: str = "enhancers"


class TrainingArgs(TrainingArguments):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
    return {"mcc": mcc, "accuracy": acc}


def get_data_path(parent_dir, dataset_name):

    train_path = os.path.join(parent_dir, dataset_name, "train.csv")
    valid_path = os.path.join(parent_dir, dataset_name, "dev.csv")
    test_path = os.path.join(parent_dir, dataset_name, "test.csv")

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
    # training_args = TrainingArgs(output_dir="output")

    print(f"Model arguments: {model_args}")
    print(f"Data arguments: {data_args}")
    print(f"Trainging arguments: {training_args}")
    # set seed
    set_seed(training_args.seed)
    # load dataset
    train_path, valid_path, test_path = get_data_path(
        data_args.parent_dir, data_args.dataset_name
    )
    train_dataset = NTDataset(train_path)
    train_size = int(0.9 * len(train_dataset))
    valid_size = len(train_dataset) - train_size

    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])
    test_dataset = NTDataset(test_path)
    num_labels = test_dataset.num_labels
    target_length = test_dataset[0]["sequence"].shape[0] // 128 + 1
    # load model
    model = FintuneModel(
        model_args.model_name_or_path,
        num_labels=num_labels,
        target_length=target_length,
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
