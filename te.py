import os
import time
from dataclasses import dataclass

import numpy as np
import torch
from Bio import SeqIO
from torchviz import make_dot
from transformers import HfArgumentParser, Trainer, TrainerCallback, TrainingArguments

from dataloaders.h5dataset import GEPBedDataset, MultiSpeciesDataset
from model import Space, SpaceConfig
from model.modeling_space import TrainingSpace
from MyTrainer import MultiLossTrainer
from utils.misc import set_seed


@dataclass
class ModelArguments:
    model_name_or_path: str = "/home/jiwei_zhu/disk/Enformer/enformer_ckpt"
    moe: str = "species_tracks"
    MIloss_lambda: float = 0.01
    zloss_lambda: float = 0.001
    cvloss_lambda: float = 0.001
    species_num_experts: int = 4
    tracks_num_experts: int = 8
    top_k: int = 3


@dataclass
class DataTrainingArguments:
    human_train_data_path: str = "/home/jiwei_zhu/disk/Enformer/Data/human_train.h5"
    human_valid_data_path: str = "/home/jiwei_zhu/disk/Enformer/Data/human_valid.h5"
    human_test_data_path: str = "/home/jiwei_zhu/disk/Enformer/Data/human_test.h5"
    human_train_bed_path: str = "/home/jiwei_zhu/disk/Enformer/Data/human_train.bed"
    human_valid_bed_path: str = "/home/jiwei_zhu/disk/Enformer/Data/human_valid.bed"
    human_test_bed_path: str = "/home/jiwei_zhu/disk/Enformer/Data/human_test.bed"
    human_genome_path: str = "/home/jiwei_zhu/disk/Enformer/Data/hg38.ml.fa"

    mouse_train_data_path: str = "/home/jiwei_zhu/disk/Enformer/Data/mouse_train.h5"
    mouse_valid_data_path: str = "/home/jiwei_zhu/disk/Enformer/Data/mouse_valid.h5"
    mouse_test_data_path: str = "/home/jiwei_zhu/disk/Enformer/Data/mouse_test.h5"
    mouse_train_bed_path: str = "/home/jiwei_zhu/disk/Enformer/Data/mouse_train.bed"
    mouse_valid_bed_path: str = "/home/jiwei_zhu/disk/Enformer/Data/mouse_valid.bed"
    mouse_test_bed_path: str = "/home/jiwei_zhu/disk/Enformer/Data/mouse_test.bed"
    mouse_genome_path: str = "/home/jiwei_zhu/disk/Enformer/Data/mm10.fa"

    dataset_path: str = "/home/jiwei_zhu/disk/Enformer/Data/data"
    seqlen: int = 131072
    shift_aug: bool = True
    rc_aug: bool = True


class TrainingArgs(TrainingArguments):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_steps = 1000
        self._n_gpu = 1
        self.per_device_train_batch_size = 1
        self.per_device_eval_batch_size = 4
        self.gradient_accumulation_steps = 4
        self.dataloader_num_workers = 8
        self.dataloader_prefetch_factor = 2
        self.learning_rate = 5e-4
        self.lr_scheduler_type = "cosine"
        self.weight_decay = 1e-2
        self.save_steps = 2
        self.eval_steps = 2
        self.logging_steps = 1
        self.warmup_steps = 5000
        self.optim = "adamw_torch"
        self.save_strategy = "steps"
        self.eval_strategy = "steps"
        self.remove_unused_columns = False
        self.save_safetensors = False
        self.max_grad_norm = 0.2


def main():
    # torch.autograd.set_detect_anomaly(True)
    # args
    model_args = ModelArguments()
    data_args = DataTrainingArguments()
    training_args = TrainingArgs(output_dir="outputs_temp")
    print(f"Model arguments: {model_args}")
    print(f"Data arguments: {data_args}")
    # set seed
    set_seed(training_args.seed)
    # load config
    config = SpaceConfig.from_pretrained(model_args.model_name_or_path)
    print(config)
    config.update(
        {
            "moe": model_args.moe.split("_"),
            "MIloss_lambda": model_args.MIloss_lambda,
            "zloss_lambda": model_args.zloss_lambda,
            "cvloss_lambda": model_args.cvloss_lambda,
            "species_num_experts": model_args.species_num_experts,
            "tracks_num_experts": model_args.tracks_num_experts,
            "topk": model_args.top_k,
            "dim": 768
        }
    )
    model = TrainingSpace(config)
    # model_path = "/home/jiwei_zhu/disk/Enformer/enformer_MoE/outputs/Enformer_version5_species_tracks_4/checkpoint-28000"
    # config = SpaceConfig.from_pretrained(os.path.join(model_path, "config.json"))
    # model = TrainingSpace(config)
    # model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin")))
    if not os.path.exists("./outputs_temp"):
        os.makedirs("./outputs_temp")
    output_file = "./outputs_temp/model.txt"
    with open(output_file, "w") as f:
        f.write(str(model))

    # load dataset
    train_dataset = MultiSpeciesDataset(
        file_paths = [data_args.human_train_data_path, data_args.mouse_train_data_path],
        bed_paths = [data_args.human_train_bed_path, data_args.mouse_train_bed_path],
        seqlen = data_args.seqlen,
        genome_paths = [data_args.human_genome_path, data_args.mouse_genome_path],
        shift_aug = data_args.shift_aug,
        rc_aug = data_args.rc_aug,
    )
    valid_dataset = MultiSpeciesDataset(
        file_paths = [data_args.human_valid_data_path, data_args.mouse_valid_data_path],
        bed_paths = [data_args.human_valid_bed_path, data_args.mouse_valid_bed_path],
        seqlen = data_args.seqlen,
        genome_paths = [data_args.human_genome_path, data_args.mouse_genome_path],
        shift_aug = False,
        rc_aug = False,
    )

    # load model
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Number of parameters: {num_params} M")
    # set trainer
    trainer = MultiLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=[valid_dataset[i] for i in range(30)],
    )
    trainer.train()


if __name__ == "__main__":
    main()
