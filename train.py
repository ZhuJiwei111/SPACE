from dataclasses import dataclass, field
from typing import Optional
import time
import numpy as np
import torch
from transformers import Trainer, TrainingArguments, HfArgumentParser, TrainerCallback
from Bio import SeqIO

from utils.misc import set_seed
from dataloaders.h5dataset import GEPBedDataset, MultiSpeciesDataset
from model import Space, SpaceConfig
from model.modeling_space import TrainingSpace
from MyTrainer import MultiLossTrainer


@dataclass
class ModelArguments:
    model_name_or_path: str = "/home/jiwei_zhu/disk/Enformer/enformer_ckpt"
    moe: str = "baseline"
    MIloss_lambda: float = 0.01
    zloss_lambda: float = 0.001
    cvloss_lambda: float = 0.001
    species_num_experts: int = 4
    tracks_num_experts: int = 15
    dim: int = 768
    tracks_topk: int = 3
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


def main():
    # args
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(f"Model arguments: {model_args}")
    print(f"Data arguments: {data_args}")
    # set seed
    set_seed(training_args.seed)
    # load config
    config = SpaceConfig.from_pretrained(model_args.model_name_or_path)
    config.update(
        {
            "moe": model_args.moe.split("_"),
            "MIloss_lambda": model_args.MIloss_lambda,
            "zloss_lambda": model_args.zloss_lambda,
            "cvloss_lambda": model_args.cvloss_lambda,
            "species_num_experts": model_args.species_num_experts,
            "tracks_num_experts": model_args.tracks_num_experts,
            "tracks_topk": model_args.tracks_topk,
            "topk": model_args.top_k,
            "dim": model_args.dim,
        }
    )
    model = TrainingSpace(config)
    output_file = training_args.output_dir + "/model.txt"
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
        eval_dataset=valid_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
