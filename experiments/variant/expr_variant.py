import random
from dataclasses import dataclass
import os
import sys
import numpy as np
import torch
from Bio import SeqIO
from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.insert(0, project_root)

from model.modeling_enformer import Enformer

from model.modeling_space import SpaceConfig, TrainingSpace

from dataloaders.h5dataset import VCFDataset


class SpaceForVEPModel(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        # self.model = Enformer.from_pretrained(model_path, use_tf_gamma=False)
        config = SpaceConfig.from_pretrained(os.path.join(model_path, "config.json"))
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

    def forward(
        self,
        ref_x=None,
        alt_x=None,
    ):
        ref_pred = self.model(ref_x, species="human")["out"]  # (bs, seq_len, 5313)
        ref_pred = torch.sum(ref_pred, dim=1)  # (bs, 5313)
        alt_pred = self.model(alt_x, species="human")["out"]  # (bs, seq_len, 5313)
        alt_pred = torch.sum(alt_pred, dim=1)  # (bs, 5313)
        pred = torch.cat([ref_pred, alt_pred], dim=1) # (bs, 5313 * 2)
        return pred

        # ref_pred_human = self.model(ref_x, head="human")["out"]  # (bs, seq_len, 5313)
        # ref_pred_mouse = self.model(ref_x, head="mouse")["out"]  # (bs, seq_len, 1643)
        # ref_pred = torch.cat([ref_pred_human, ref_pred_mouse], dim=-1)  # (bs, seq_len, 5313 + 1643)
        # ref_pred = torch.sum(ref_pred, dim=1)  # (bs, 5313 + 1643)

        # alt_pred_human = self.model(alt_x, head="human")["out"]  # (bs, seq_len, 5313)
        # alt_pred_mouse = self.model(alt_x, head="mouse")["out"]  # (bs, seq_len, 1643)
        # alt_pred = torch.cat([alt_pred_human, alt_pred_mouse], dim=-1)  # (bs, seq_len, 5313 + 1643)
        # alt_pred = torch.sum(alt_pred, dim=1)  # (bs, 5313 + 1643)

        # pred = torch.cat([ref_pred, alt_pred], dim=1)  # (bs, 5313 + 1643 + 5313 + 1643)
        # return pred

        # ref_pred_human = self.model(ref_x, head="human")["out"]  # (bs, seq_len, 5313)
        # ref_pred_mouse = self.model(ref_x, head="mouse")["out"]  # (bs, seq_len, 1643)
        # ref_pred = torch.cat([ref_pred_human, ref_pred_mouse], dim=-1)  # (bs, seq_len, 5313 + 1643)
        # ref_pred = torch.sum(ref_pred, dim=1)  # (bs, 5313 + 1643)

        # alt_pred_human = self.model(alt_x, head="human")["out"]  # (bs, seq_len, 5313)
        # alt_pred_mouse = self.model(alt_x, head="mouse")["out"]  # (bs, seq_len, 1643)
        # alt_pred = torch.cat([alt_pred_human, alt_pred_mouse], dim=-1)  # (bs, seq_len, 5313 + 1643)
        # alt_pred = torch.sum(alt_pred, dim=1)  # (bs, 5313 + 1643)

        # pred = torch.cat([ref_pred, alt_pred], dim=1)  # (bs, 5313 + 1643 + 5313 + 1643)
        # return pred


class EnformerForVEPModel(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = Enformer.from_pretrained(model_path, use_tf_gamma=False)

    def forward(
        self,
        ref_x=None,
        alt_x=None,
    ):
        ref_pred = self.model(ref_x, head="human")  # (bs, seq_len, 5313)
        ref_pred = torch.sum(ref_pred, dim=1)  # (bs, 5313)
        alt_pred = self.model(alt_x, head="human")  # (bs, seq_len, 5313)
        alt_pred = torch.sum(alt_pred, dim=1)  # (bs, 5313)
        pred = torch.cat([ref_pred, alt_pred], dim=1) # (bs, 5313 * 2)
        return pred



@dataclass
class ModelArguments:
    # model_name_or_path: str = "/home/jiwei_zhu/disk/Enformer/enformer_ckpt/"
    model_name_or_path: str = "/home/jiwei_zhu/SPACE/results/Space_species_tracks/checkpoint-28000"


@dataclass
class DataTrainingArguments:
    tissue_name: str = "Adipose_Subcutaneous"
    genome_path = "./genome/GRCh38.primary_assembly.genome.fa"
    # 1024 8192 32768 131072 196608
    window_size: int = 196608


@dataclass
class MyTrainingArguments(TrainingArguments):
    output_dir: str = "./outputs/"
    seed: int = 42
    per_device_train_batch_size: int = 12
    per_device_eval_batch_size: int = 12
    dataloader_num_workers: int = 16
    remove_unused_columns: bool = False
    report_to: str = "none"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, MyTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(f"Model arguments: {model_args}")
    print(f"Data arguments: {data_args}")
    set_seed(training_args.seed)
    pos_vcf_path = (
        f"./calibrated_vcf/{data_args.tissue_name}_pos.vcf"
    )
    neg_vcf_path = (
        f"./calibrated_vcf/{data_args.tissue_name}_neg.vcf"
    )
    genome_dict = SeqIO.to_dict(SeqIO.parse(data_args.genome_path, "fasta"))
    # import debugpy; debugpy.connect(5678); debugpy.wait_for_client(); breakpoint()
    pos_dataset = VCFDataset(pos_vcf_path, genome_dict, data_args.window_size)
    neg_dataset = VCFDataset(neg_vcf_path, genome_dict, data_args.window_size)
    if "enformer" in model_args.model_name_or_path:
        model = EnformerForVEPModel(model_args.model_name_or_path)
    else:
        model = SpaceForVEPModel(model_args.model_name_or_path)

    trainer = Trainer(
        model=model,
        args=training_args,
    )

    try:
        pos_pred = trainer.predict(pos_dataset).predictions  # (bs, 5313 * 2)
        print(f"shape of pos_pred: {pos_pred.shape}")
        print(f"len of ref dataset: {len(pos_dataset)}")
        neg_pred = trainer.predict(neg_dataset).predictions  # (bs, 5313 * 2)
        print(f"shape of neg_pred: {neg_pred.shape}")
        print(f"len of alt dataset: {len(neg_dataset)}")

        # 将预测结果转换为numpy数组并添加新轴
        pos_pred_expanded = np.expand_dims(pos_pred, axis=1)
        neg_pred_expanded = np.expand_dims(neg_pred, axis=1)

        # (bs, 2, 5313 * 2)
        pred = np.concatenate([pos_pred_expanded, neg_pred_expanded], axis=1)
        print(f"shape of pred: {pred.shape}")

        np.save(f"{training_args.output_dir}/{data_args.tissue_name}.npy", pred)

    except Exception as e:
        print(f"Error processing tissue {data_args.tissue_name}: {e}", file=sys.stderr)
        return 1  # 返回非零表示失败


if __name__ == "__main__":
    sys.exit(main())
