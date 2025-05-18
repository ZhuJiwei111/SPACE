import math
from pathlib import Path
import wandb
import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint_sequential
import pandas as pd
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from model.data import str_to_one_hot, seq_indices_to_one_hot

from model.config_space import SpaceConfig

from transformers import PreTrainedModel

from .modules import (
    exponential_linspace_int,
    pearson_corr_coef,
    poisson_loss,
    GELU,
    Residual,
    ConvBlock,
    TransformerModel,
    AttentionPool,
    TargetLengthCrop,
    SpeciesMoE,
    TracksMoE,
)


# constants
SEQUENCE_LENGTH = 196_608
TARGET_LENGTH = 896


# main class
class Space(PreTrainedModel):
    config_class = SpaceConfig
    base_model_prefix = "space"

    @staticmethod
    def from_hparams(**kwargs):
        return Space(SpaceConfig(**kwargs))

    def __init__(self, config):
        super().__init__(config)
        self.dim = config.dim
        half_dim = config.dim // 2
        twice_dim = config.dim * 2
        self.species = list(config.output_heads.keys())

        # create stem

        self.stem = nn.Sequential(
            nn.Conv1d(4, half_dim, 15, padding=7),
            Residual(ConvBlock(half_dim)),
            AttentionPool(half_dim, pool_size=2),
        )

        # create conv tower

        filter_list = exponential_linspace_int(
            half_dim,
            config.dim,
            num=(config.num_downsamples - 1),
            divisible_by=config.dim_divisible_by,
        )
        filter_list = [half_dim, *filter_list]

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(
                nn.Sequential(
                    ConvBlock(dim_in, dim_out, kernel_size=5),
                    Residual(ConvBlock(dim_out, dim_out, 1)),
                    AttentionPool(dim_out, pool_size=2),
                )
            )

        self.conv_tower = nn.Sequential(*conv_layers)

        # whether to use tensorflow gamma positions

        use_tf_gamma = config.use_tf_gamma
        self.use_tf_gamma = use_tf_gamma

        # transformer

        self.transformer = TransformerModel(config, self.species, use_tf_gamma)

        # target cropping

        self.target_length = config.target_length
        self.crop_final = TargetLengthCrop(config.target_length)

        # final pointwise

        self.final_pointwise = nn.Sequential(
            Rearrange("b n d -> b d n"),
            ConvBlock(filter_list[-1], twice_dim, 1),
            Rearrange("b d n -> b n d"),
            nn.Dropout(config.dropout_rate / 8),
            GELU(),
        )

        # create final heads for human and mouse
        self.heads = nn.ModuleDict({
            key: nn.Sequential(nn.Linear(self.dim * 2, features),) 
            for key, features in config.output_heads.items()
        })

        if "tracks" in config.moe:
            self.tracks = TracksMoE(config, self.species, topk=config.tracks_topk)
            
        self.softplus = nn.Softplus()

        # use checkpointing on transformer trunk

        self.use_checkpointing = config.use_checkpointing

    def set_target_length(self, target_length):
        self.crop_final.target_length = target_length

    @property
    def trunk(self):
        return self._trunk

    @property
    def heads(self):
        return self._heads

    def trunk_checkpointed(self, x):
        x = rearrange(x, "b n d -> b d n")
        x = self.stem(x)
        x = self.conv_tower(x)
        x = rearrange(x, "b d n -> b n d")
        x = checkpoint_sequential(self.transformer, len(self.transformer), x)
        x = self.crop_final(x)
        x = self.final_pointwise(x)
        return x

    def forward(
        self,
        x,
        labels=None,
        return_corr_coef=False,
        return_embeddings=False,
        return_only_embeddings=False,
        species="human",
        target_length=896,
    ):
        
        if isinstance(x, list):
            x = str_to_one_hot(x)

        elif type(x) == torch.Tensor and x.dtype == torch.long:
            x = seq_indices_to_one_hot(x)
        x.to(self.device)

        no_batch = x.ndim == 2

        if no_batch:
            x = rearrange(x, "... -> () ...")

        if target_length is not None:
            self.set_target_length(target_length)
        x = Rearrange("b n d -> b d n")(x)
        x = self.stem(x)
        x = self.conv_tower(x)
        x = Rearrange("b d n -> b n d")(x)

        #!------------------------------------------------------
        # x：(batch_size,1024,768)
        x, gates_list, species_zloss, species_cvloss = self.transformer(x, species)
        x, embedding = x[:, :-1, :], x[:, -1, :]
        species_statistics = {"gates": gates_list, "zloss": species_zloss, "cvloss": species_cvloss}
        #!------------------------------------------------------

        x = self.crop_final(x)
        x = self.final_pointwise(x)

        if return_only_embeddings:
            if no_batch:
                x = rearrange(x, "() ... -> ...")
            # return x
            return x, species_statistics
        
        out = self.heads[species](x)
        #!------------------------------------------------------
        #! tracks MoE
        tracks_statistics = {}
        if "tracks" in self.config.moe:
            #! 计划将species_embedding添加进门控选择，通过species_embedding门控和token门控相加的方式
            out_enhanced, all_gates, zloss, cvloss, weights = self.tracks(x, out, species, embedding)
            tracks_statistics = {"gates": all_gates, "zloss": zloss, "cvloss": cvloss, "weights": weights}
            out = (out + out_enhanced) / 2
        #!------------------------------------------------------

        out = self.softplus(out)
        
        if no_batch:
            x = rearrange(x, "() ... -> ...")
            out = rearrange(out, "() ... -> ...")

        if labels is not None:
            assert species is not None, "species must be passed in if one were to calculate loss directly with targets"

            if return_corr_coef:
                return pearson_corr_coef(out, labels)

            loss = poisson_loss(out, labels)
            #! 将原始out和增强后的out同时计算loss，然后取平均
            if self.training and "tracks" in self.config.moe:
                out_enhanced = self.softplus(out_enhanced)
                loss = (loss + poisson_loss(out_enhanced, labels)) / 2
            return {
                "loss": loss,
                "species": species_statistics,
                "tracks": tracks_statistics,
            }
        if return_embeddings:
            return out, x

        return {
            "out": out,
            "species": species_statistics,
            "tracks": tracks_statistics,
        }


class TrainingSpace(PreTrainedModel):
    """
    一共有3个辅助loss:
    1.zloss: 计算logits的二范数，保证logits平滑，避免极端决策
    2.cvloss: 计算gates的方差除以均值的平方，鼓励专家被均匀使用
    3.MIloss: 计算负互信息，鼓励任务选择特定的专家
    zloss是用于平滑logits的，所以在MoE中加以约束。
    MIloss是针对所有任务的，所以需要在主函数中，计算完所有任务再加入
    #? cvloss，我不确定对整体加还是对每个任务加更好，开始实现的时候我默认是对每个任务加的，但是想了一下感觉对整体加更好
    #? 但是这也不是什么重要的点，卡又比较紧张就一直没试
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = Space(config)
        self.species = ["human", "mouse"]
        self.species_num_experts = config.species_num_experts
        self.tracks_num_experts = config.tracks_num_experts
        self.MIloss_lambda = config.MIloss_lambda
        self.zloss_lambda = config.zloss_lambda
        self.cvloss_lambda = config.cvloss_lambda

    def forward(self, human_x, mouse_x, human_labels=None, mouse_labels=None):
        """
        config.moe是用于规定使用MoE的种类
        species表示对物种使用MoE，也就是替换transformer的ffn层的
        tracks表示对tracks使用MoE，也就是加在outputs后面的
        最后所有loss都会加在total上，计算梯度的时候也只计算total，return的其他loss只是为了在wandb上观察各自的结果
        而且eval的时候不会计算辅助loss
        """
        output_human = self.model(human_x, labels=human_labels, species="human")
        output_mouse = self.model(mouse_x, labels=mouse_labels, species="mouse")

        if human_labels is not None and mouse_labels is not None:
            loss = {
                "human": output_human["loss"],
                "mouse": output_mouse["loss"],
            }
            device = human_labels.device
            #! 计算辅助loss
            if "species" in self.config.moe and self.training:
                batch_size = human_labels.shape[0]
                species_loss = self.get_species_loss(output_human["species"], output_mouse["species"], batch_size, device)
                loss.update(species_loss)
            if "tracks" in self.config.moe and self.training:
                batch_size = human_labels.shape[0]
                tracks_loss = self.get_tracks_loss(output_human["tracks"], output_mouse["tracks"], batch_size, device)
                loss.update(tracks_loss)

            total_loss = torch.zeros_like(loss["human"])
            for k, v in loss.items():
                total_loss = total_loss + v
            loss["total"] = total_loss
            return {"loss": loss, "human": output_human, "mouse": output_mouse}

        return {"human": output_human, "mouse": output_mouse}

    def get_species_loss(self, statistics_human, statistics_mouse, batch_size, device):
        """
        species是替换transformer的ffn层，所以计算辅助loss时要把所有层的loss加起来
        tot是用于对gates归一化的
        """
        tot = batch_size * self.config.num * 2
        total_MIloss = torch.tensor(0.0, device=device)
        # total_cvloss = torch.tensor(0.0, device=device)
        for i, block in enumerate(self.model.transformer.transformer):
            if isinstance(block.feed_forward, SpeciesMoE):
                gates = torch.stack(
                    [statistics_human["gates"][i], statistics_mouse["gates"][i]], dim=0
                )
                gates = gates / tot
                MIloss, cvloss = self._compute_aux_loss(gates)
                total_MIloss = total_MIloss + self.MIloss_lambda * MIloss
                # total_cvloss = total_cvloss + self.cvloss_lambda * cvloss
        zloss = self.zloss_lambda * (statistics_human["zloss"] + statistics_mouse["zloss"])
        cvloss = self.cvloss_lambda * (statistics_human["cvloss"] + statistics_mouse["cvloss"])
        loss = {
            "species_auxloss": total_MIloss,
            "species_zloss": zloss,
            "species_cvloss": cvloss,
        }
        return loss

    def get_tracks_loss(self, statistics_human, statistics_mouse, batch_size, device):
        tot = batch_size * (5313 + 1643)
        gates = statistics_human["gates"]
        gates.update(statistics_mouse["gates"])
        gates = torch.stack([gates[key] for key in gates.keys()], dim=0)

        # gates = torch.stack([
        #     statistics_human["gates"][tracks] + statistics_mouse["gates"][tracks]
        #     for tracks in statistics_human["gates"].keys()
        # ], dim=0)

        gates = gates / tot
        MIloss, cvloss = self._compute_aux_loss(gates)
        cvloss = self.cvloss_lambda * cvloss          #? 对整体的gates计算cvloss 
        MIloss = self.MIloss_lambda * MIloss
        zloss = self.zloss_lambda * (statistics_human["zloss"] + statistics_mouse["zloss"])
        # cvloss = 0.001 * (statistics_human["cvloss"] + statistics_mouse["cvloss"])    #? 对每个task自己计算cvloss

        # weights = torch.stack([statistics_human["weights"].sum(dim=0), statistics_mouse["weights"].sum(dim=0)], dim=0)
        # weights = weights / batch_size / 8
        # MIloss += self.MIloss_lambda * self._compute_aux_loss(weights)[0]
        loss = {
            "tracks_auxloss": MIloss,
            "tracks_zloss": zloss,
            "tracks_cvloss": cvloss,
        }
        return loss

    def _compute_aux_loss(self, gates):
        #! 互信息损失
        eps = 1e-10
        # 计算MIloss
        P_TI = torch.sum(gates, dim=1, keepdim=True) + eps
        P_EI = torch.sum(gates, dim=0, keepdim=True) + eps
        MIloss = -(gates * torch.log(gates / P_TI / P_EI + eps)).sum()
        # ? 计算cvloss，对整体的gates计算
        experts_usage = gates.sum(dim=0)
        cvloss = experts_usage.var() / (experts_usage.mean() ** 2 + eps)

        return MIloss, cvloss
