import math
from pathlib import Path
import wandb
import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributed as dist
import pandas as pd
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch.cuda.amp import custom_fwd


# gamma positions from tensorflow
# addressing a difference between xlogy results from tensorflow and pytorch
# solution came from @johahi

DIR = Path(__file__).parents[0]
TF_GAMMAS = torch.load(str(DIR / "precomputed" / "tf_gammas.pt"))

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def always(val):
    def inner(*args, **kwargs):
        return val

    return inner


def exponential_linspace_int(start, end, num, divisible_by=1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


# maybe sync batchnorm, for distributed training


def MaybeSyncBatchnorm(is_distributed=True):
    is_distributed = default(
        is_distributed, dist.is_initialized() and dist.get_world_size() > 1
    )
    # print(f"sync batchnorm for distributed training: {is_distributed}")
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm1d


# losses and metrics


def poisson_loss(pred, target):
    return (pred - target * log(pred)).mean()


def pearson_corr_coef(x, y, dim=1, reduce_dims=(-1,)):
    x_centered = x - x.mean(dim=dim, keepdim=True)
    y_centered = y - y.mean(dim=dim, keepdim=True)
    return F.cosine_similarity(x_centered, y_centered, dim=dim).mean(dim=reduce_dims)


# relative positional encoding functions


def get_positional_features_exponential(
    positions, features, seq_len, min_half_life=3.0, dtype=torch.float
):
    max_range = math.log(seq_len) / math.log(2.0)
    half_life = 2 ** torch.linspace(
        min_half_life, max_range, features, device=positions.device
    )
    half_life = half_life[None, ...]
    positions = positions.abs()[..., None]
    return torch.exp(-math.log(2.0) / half_life * positions)


def get_positional_features_central_mask(
    positions, features, seq_len, dtype=torch.float
):
    center_widths = 2 ** torch.arange(1, features + 1, device=positions.device).to(
        dtype
    )
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).to(dtype)


def gamma_pdf(x, concentration, rate):
    log_unnormalized_prob = torch.xlogy(concentration - 1.0, x) - rate * x
    log_normalization = torch.lgamma(concentration) - concentration * torch.log(rate)
    return torch.exp(log_unnormalized_prob - log_normalization)


def get_positional_features_gamma(
    positions,
    features,
    seq_len,
    stddev=None,
    start_mean=None,
    eps=1e-8,
    dtype=torch.float,
):
    if not exists(stddev):
        stddev = seq_len / (2 * features)

    if not exists(start_mean):
        start_mean = seq_len / features

    mean = torch.linspace(start_mean, seq_len, features, device=positions.device)

    mean = mean[None, ...]
    concentration = (mean / stddev) ** 2
    rate = mean / stddev**2

    probabilities = gamma_pdf(positions.to(dtype).abs()[..., None], concentration, rate)
    probabilities = probabilities + eps
    outputs = probabilities / torch.amax(probabilities, dim=-1, keepdim=True)
    return outputs


def get_positional_embed(
    seq_len, feature_size, device, use_tf_gamma, dtype=torch.float
):
    distances = torch.arange(-seq_len + 1, seq_len, device=device)

    assert (
        not use_tf_gamma or seq_len == 1536
    ), "if using tf gamma, only sequence length of 1536 allowed for now"

    feature_functions = [
        get_positional_features_exponential,
        get_positional_features_central_mask,
        (
            get_positional_features_gamma
            if not use_tf_gamma
            else always(TF_GAMMAS.to(device))
        ),
    ]

    num_components = len(feature_functions) * 2

    if (feature_size % num_components) != 0:
        raise ValueError(
            f"feature size is not divisible by number of components ({num_components})"
        )

    num_basis_per_class = feature_size // num_components

    embeddings = []
    for fn in feature_functions:
        embeddings.append(fn(distances, num_basis_per_class, seq_len, dtype=dtype))

    embeddings = torch.cat(embeddings, dim=-1)
    embeddings = torch.cat(
        (embeddings, torch.sign(distances)[..., None] * embeddings), dim=-1
    )
    return embeddings.to(dtype)


def relative_shift(x):
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim=-1)
    _, h, t1, t2 = x.shape
    x = x.reshape(-1, h, t2, t1)
    x = x[:, :, 1:, :]
    x = x.reshape(-1, h, t1, t2 - 1)
    return x[..., : ((t2 + 1) // 2)]


# classes


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x


class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size=2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange("b d (n p) -> b d n p", p=pool_size)

        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias=False)

        nn.init.dirac_(self.to_attn_logits.weight)

        with torch.no_grad():
            self.to_attn_logits.weight.mul_(2)

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value=0)
            mask = torch.zeros((b, 1, n), dtype=torch.bool, device=x.device)
            mask = F.pad(mask, (0, remainder), value=True)

        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim=-1)

        return (x * attn).sum(dim=-1)


class TargetLengthCrop(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-2], self.target_length

        if target_len == -1:
            return x

        if seq_len < target_len:
            raise ValueError(
                f"sequence length {seq_len} is less than target length {target_len}"
            )

        trim = (target_len - seq_len) // 2

        if trim == 0:
            return x

        return x[:, -trim:trim]


def ConvBlock(dim, dim_out=None, kernel_size=1, is_distributed=True):
    batchnorm_klass = MaybeSyncBatchnorm(is_distributed=is_distributed)

    return nn.Sequential(
        batchnorm_klass(dim),
        GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding=kernel_size // 2),
    )


# attention classes


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_rel_pos_features,
        heads=8,
        dim_key=64,
        dim_value=64,
        dropout=0.0,
        pos_dropout=0.0,
        use_tf_gamma=False,
    ):
        super().__init__()
        self.scale = dim_key**-0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias=False)

        self.to_out = nn.Linear(dim_value * heads, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        # relative positional encoding

        self.num_rel_pos_features = num_rel_pos_features

        self.to_rel_k = nn.Linear(num_rel_pos_features, dim_key * heads, bias=False)
        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        # dropouts

        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # whether to use tf gamma

        self.use_tf_gamma = use_tf_gamma

    def forward(self, x):
        n, h, device = x.shape[-2], self.heads, x.device

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        q = q * self.scale

        content_logits = einsum(
            "b h i d, b h j d -> b h i j", q + self.rel_content_bias, k
        )

        positions = get_positional_embed(
            n,
            self.num_rel_pos_features,
            device,
            use_tf_gamma=self.use_tf_gamma,
            dtype=self.to_rel_k.weight.dtype,
        )
        positions = self.pos_dropout(positions)
        rel_k = self.to_rel_k(positions)

        rel_k = rearrange(rel_k, "n (h d) -> h n d", h=h)
        rel_logits = einsum("b h i d, h j d -> b h i j", q + self.rel_pos_bias, rel_k)
        rel_logits = relative_shift(rel_logits)

        logits = content_logits + rel_logits
        attn = logits.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


#!----------------------------------------------------------------------------------------
def compute_cvloss(gates):
    # 鼓励所有专家被均匀地使用，通过计算专家使用率的方差除以均值的平方来反映专家使用分布的均匀性。
    expert_usage = gates
    mean_usage = expert_usage.mean()
    var_usage = expert_usage.var()
    eps = 1e-10
    loss = var_usage / (mean_usage**2 + eps)
    return loss

def compute_zloss(logits):
    # 确保 logits 分布平滑，避免极端决策，提高训练稳定性。
    loss = torch.mean(logits**2)
    return loss

## MLP
class MLP(nn.Module):
    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.input = nn.Linear(dim, dim * 2)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
        self.output = nn.Linear(dim * 2, dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.input(x)
        x = self.dropout1(self.activation(x))
        x = self.dropout2(self.output(x))
        return x + residual


## Experts实际上是多个linear，只是优化了forward的计算
# 主要是把token选择了哪些专家，转换为专家被哪些token选择，需要计算哪些token的输出，方便矩阵运算
class Experts(nn.Module):
    def __init__(self, input_size, output_size, num_experts, bias=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_experts = num_experts

        # 初始化权重和偏置
        self.weight = nn.Parameter(torch.empty(num_experts, input_size, output_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_experts, output_size))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # 权重初始化
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
        # 偏置初始化
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @custom_fwd(cast_inputs=torch.float16)
    def forward(self, inputs, expert_size):
        """
        inputs: [batch_size * length * k, dim]
        """
        # 根据专家选择对输入进行划分
        input_list = torch.split(inputs, expert_size.tolist(), dim=0)
        # 存储每个专家的输出
        output_list = []

        for i in range(self.num_experts):
            # 计算每个专家的输出
            expert_output = torch.mm(input_list[i], self.weight[i])
            if self.bias is not None:
                expert_output += self.bias[i]
            output_list.append(expert_output)
            del expert_output
        # 拼接所有专家的输出
        output = torch.cat(output_list, dim=0)
        return output


@torch.jit.script
def compute_gating(k: int, logits: torch.Tensor, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor):
    """
    logits: [batch_size * length, num_experts]
    top_k_gates: [batch_size * length, k]
    top_k_indices: [batch_size * length, k]
    """
    # 门控矩阵，主要用于后续的辅助loss的计算，以及最终的可视化
    gates = torch.zeros_like(logits)
    gates.scatter_(1, top_k_indices, top_k_gates)

    # 计算每个专家被选中的次数，即每个专家要处理的tokens数量
    expert_size = (gates > 0).sum(dim=0)

    # 获取非零的 gates 和对应的专家索引
    nonzero_gates = top_k_gates[top_k_gates != 0]
    nonzero_experts = top_k_indices[top_k_gates != 0]

    # 合并专家索引和对应的 batch 索引
    sorted_experts, sorted_indices = nonzero_experts.sort()
    batch_index = sorted_indices.div(k, rounding_mode="trunc")
    batch_gates = nonzero_gates[sorted_indices]
    del sorted_experts, sorted_indices, nonzero_experts
    return gates, expert_size, batch_index, batch_gates


## SpeciesMoE
class SpeciesMoE(nn.Module):
    def __init__(self, species: list, dim: int, num_experts: int, topk: int, dropout_rate: float, noisy_gating=False):
        super().__init__()
        self.species = species
        self.num_experts = num_experts
        self.k = topk

        # 门控
        self.noisy_gating = noisy_gating
        output_dim = 2 * num_experts if noisy_gating else num_experts  # 如果使用噪声门控，就需要生成对应噪声，所以output_dim需要乘以2
        self.gates = nn.ModuleDict({
            k: nn.Sequential(nn.Linear(dim, output_dim), nn.LeakyReLU())
            for k in species
        })

        # 专家
        self.layer_norm = nn.LayerNorm(dim)
        self.input = Experts(dim, dim * 2, num_experts)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
        self.output = Experts(dim * 2, dim, num_experts)
        self.dropout2 = nn.Dropout(dropout_rate)

    def top_k_gating(self, x, species, noise_epsilon=1e-5):
        logits = self.gates[species](x)

        if self.noisy_gating:
            # 如果是噪声门控，通过gate得到logits和原始噪声标准差
            logits, raw_noise_std = logits.chunk(2, dim=-1)
            if self.training:
                noise_std = F.softplus(raw_noise_std) + noise_epsilon
                eps = torch.randn_like(logits)
                logits = logits + eps * noise_std

        top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=1)
        probs = F.softmax(top_k_logits, dim=1)

        gates, expert_size, batch_index, batch_gates = compute_gating(self.k, logits, probs, top_k_indices)
        zloss = compute_zloss(logits)
        cvloss = compute_cvloss(gates)
        return expert_size, batch_index, batch_gates, gates.sum(dim=0), zloss, cvloss

    def forward(self, x, species):
        """
        x: [batch_size, length, dim](bs, 1024, 768)
        species: str
        """
        residual = x
        batch_size, length, dim = x.shape
        x = x.view(batch_size * length, dim)
        expert_size, batch_index, batch_gates, gates, zloss, cvloss = self.top_k_gating(x, species)
        #! expert_size: 每个专家被选中的次数，即每个专家要处理的tokens数量，shape为[num_experts]，sum(expert_size) = batch_size * length * k
        #! batch_index: tokens按照专家选择的顺序排列后的索引，通过batch_index扩充x，shape为[batch_size * length * k]
            # 比如，4个专家3个token，token0选择了专家1,2,3，token1选择了专家2,3,4，token2选择了专家1,2,4，那么batch_index=[0,2,0,1,2,0,1,1,2]
            # 通过x[batch_index]将输入扩充，然后会在Experts类内部按expert_size划分，[0,2 || 0,1,2 || 0,1 || 1,2]，计算每个专家的输出
        #! batch_gates: 对应batch_index的gate值，shape为[batch_size * length * k]
        #! gates: 每个专家被选中的概率，shape为[num_experts]，用于后续的辅助loss的计算，以及最终的可视化
        #! zloss和cvloss都是辅助损失，用于优化门控网络的选择和专家的使用

        x = self.layer_norm(x)
        inputs = x[batch_index]                                         # inputs: [batch_size * length * k, dim]
        inputs = self.input(inputs, expert_size)
        inputs = self.dropout1(self.activation(inputs))
        outputs = self.dropout2(self.output(inputs, expert_size))       # outputs: [batch_size * length * k, dim]

        outputs = outputs * batch_gates.unsqueeze(1)  # 加权
        y = torch.zeros_like(x)
        y.index_add_(0, batch_index, outputs)           # y: [batch_size * length, dim]，通过batch_index将同一个token的k个输出加和
        y = y.view(batch_size, length, dim) + residual

        return y, gates, zloss, cvloss


## transformer block
class TransformerBlock(nn.Module):
    def __init__(self, config, species, use_tf_gamma=False, use_species_moe=False):
        super().__init__()
        self.attention = Residual(
            nn.Sequential(
                nn.LayerNorm(config.dim),
                Attention(
                    config.dim,
                    heads=config.heads,
                    dim_key=config.attn_dim_key,
                    dim_value=config.dim // config.heads,
                    dropout=config.attn_dropout,
                    pos_dropout=config.pos_dropout,
                    num_rel_pos_features=config.dim // config.heads,
                    use_tf_gamma=use_tf_gamma,
                ),
                nn.Dropout(config.dropout_rate),
            )
        )
        self.use_species_moe = use_species_moe
        if self.use_species_moe:
            # self.species_embedding = nn.ParameterDict(
            #     {key: nn.Parameter(torch.randn(1, 1, config.dim)) for key in species}
            # )
            self.feed_forward = SpeciesMoE(
                species,
                config.dim,
                config.species_num_experts,
                config.topk,
                config.dropout_rate,
            )
        else:
            self.feed_forward = MLP(config.dim, config.dropout_rate)

    def forward(self, x, species):
        x = self.attention(x)
        if self.use_species_moe:
            # x = x + self.species_embedding[species]
            x, gates, zloss, cvloss = self.feed_forward(x, species)
        else:
            x = self.feed_forward(x)
            gates = None
            zloss = torch.tensor(0.0, device=x.device)
            cvloss = torch.tensor(0.0, device=x.device)
        return x, gates, zloss, cvloss


## Transformer model
class TransformerModel(nn.Module):
    def __init__(self, config, species, use_tf_gamma=False, use_moe=False):
        super().__init__()
        self.config = config
        self.species = species
        #! species embedding
        self.species_embedding = nn.ParameterDict(
            {key: nn.Parameter(torch.randn(1, 1, config.dim)) for key in species}
        )
        transformer = []
        for i in range(config.depth):
            # use_moe = False
            # if config.moe == "moe":  # 如果使用MoE，交替
            #     use_moe = (i % 2 == 0)
            use_moe = "species" in config.moe
            transformer.append(TransformerBlock(config, species, use_tf_gamma, use_moe))
        self.transformer = nn.ModuleList(transformer)

    def forward(self, x, species):
        gates_list = []
        total_zloss = torch.tensor(0.0, device=x.device)
        total_cvloss = torch.tensor(0.0, device=x.device)
        #! 添加species embedding
        x = torch.cat([x, self.species_embedding[species].repeat(x.shape[0], 1, 1)], dim=1).contiguous()

        for transformer in self.transformer:
            x, gates, zloss, cvloss = transformer(x, species)
            gates_list.append(gates)
            total_zloss += zloss
            total_cvloss += cvloss
        return x, gates_list, total_zloss, total_cvloss


class TracksMoE(nn.Module):
    """
    对于Tracks目前是分为了4大类，我调用语言模型的api对target的说明文档标注了一下类别，然后根据类别把文档重排了一下
    其实只是TF ChIP-seq和Histone ChIP-seq说明文档中混在了一起。
    然后按照重排之后的index，把预测结果的5313或1643重排，用以分开tracks
    """
    TRACK_TYPES = ["DNASE/ATAC", "TF ChIP-seq", "Histone ChIP-seq", "CAGE"]

    def __init__(self, config, species, topk: int = 3, seqlen: int = 896):
        super().__init__()
        self.config = config
        self.species = species
        self.topk = topk
        self.seqlen = seqlen
        self.gates_per_type = 2  # 每个类型的门控网络数量
        self.num_experts = config.tracks_num_experts
        self.gates_num = self.gates_per_type * len(self.TRACK_TYPES)

        # 门控选择器
        self.gate_selector = nn.Linear(2 * config.dim, self.gates_num)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.embedding_gate_selector = nn.Linear(config.dim, self.gates_num)

        # 门控网络
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(seqlen, self.num_experts), nn.LeakyReLU()) 
            for _ in range(self.gates_num)
        ])

        # 专家网络
        self.layer_norm = nn.LayerNorm(seqlen)
        self.input_proj = Experts(seqlen, seqlen * 2, self.num_experts)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.activation = nn.ReLU()
        self.output_proj = Experts(seqlen * 2, seqlen, self.num_experts)

        # tracks embedding
        self.tracks_embedding = nn.ParameterDict(
            {key: nn.Parameter(torch.randn(1, 1, seqlen)) for key in self.TRACK_TYPES}
        )

        # 索引映射
        self.index = self._load_indices(species)
        self.types_index = self._init_types_index()

    def _load_indices(self, species):
        """
        加载物种对应的索引信息。
        """
        index = {}
        for sp in species:
            df = pd.read_csv(f"/home/jiwei_zhu/SPACE/dataset/targets_{sp}_sorted.txt", sep="\t")
            index[sp] = df["index"].tolist()
        return index

    def _init_types_index(self):
        """
        初始化任务类型的索引范围。
        """
        return {
            "human": {
                "human DNASE/ATAC": (0, 684),
                "human TF ChIP-seq": (684, 2573),
                "human Histone ChIP-seq": (2573, 4675),
                "human CAGE": (4675, 5313),
            },
            "mouse": {
                "mouse DNASE/ATAC": (0, 228),
                "mouse TF ChIP-seq": (228, 519),
                "mouse Histone ChIP-seq": (519, 1286),
                "mouse CAGE": (1286, 1643),
            },
        }

    def forward(self, x, out, species, embedding):
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, 2 * dim)。
            out (torch.Tensor): 初步预测，形状为 (batch_size, seq_len, tracks_num)。
            species (str): 当前物种。
            embedding (torch.Tensor): 当前物种的embedding，形状为(1, 1, dim)

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
                - 输出结果，形状为 (batch_size, seq_len, dim)。
                - 每种任务类型的专家选择情况。
                - 门控网络的正则化损失。
        """
        index = self.index[species]
        types_index = self.types_index[species]

        # 计算门控网络选择的logits
        x_pooled = self.pooling(x.transpose(1, 2)).squeeze(-1)  # x_pooled:(batch_size,2*dim)
        gate_logits = self.gate_selector(x_pooled)              # gate_logits:(batch_size, gates_num)
        embedding_logits = self.embedding_gate_selector(embedding.squeeze(1))    # embedding_logits:(batch_size, gates_num)

        residual = out        #? 如果使用原本的loss和增强后的loss相加，那这里是不是没必要加残差了？
        out = rearrange(out, "b n d -> b d n")  # 转置

        batch_size, tracks, seqlen = out.shape
        out = out[:, index, :]  # 按照tracks类型调整顺序

        gates_list = []
        weights_list = []
        y = torch.zeros_like(out)
        total_zloss = torch.tensor(0.0, device=out.device)
        total_cvloss = torch.tensor(0.0, device=out.device)

        for i, t in enumerate(self.TRACK_TYPES):
            task = f"{species} {t}"
            start, end = types_index[task]
            #? 感觉这里添加任务embedding是不是有必要？但是我还没做过这里的实验
            temp = out[:, start:end, :] + self.tracks_embedding[t]  # 添加任务embedding
            # 对于不同的track，计算 gates和辅助损失
            gates, zloss, cvloss, weights = self._compute_gates(temp, gate_logits, embedding_logits, i, batch_size)
            #! gates:(bs, temp_tracks, num_experts), temp_tracks是当前track类型的数量
            #! weights:(gates_per_type)，表示每个gate的权重，这里返回用于观察结果
            total_zloss = total_zloss + zloss
            total_cvloss = total_cvloss + cvloss
            gates_list.append(gates)
            weights_list.append(weights.detach())

        #! 这里将所有gate拼接一起处理，避免更多显式的循环
        #! 之后就类似SpeciesMoE的处理了
        gates = torch.cat(gates_list, dim=1).view(batch_size * tracks, self.num_experts)
        weights = torch.cat(weights_list, dim=1)

        # 计算每个专家被选中的次数，即每个专家要处理的tokens数量
        expert_size = (gates > 0).sum(dim=0)

        # 获取非零 gates 和对应的 batch 索引
        batch_index, batch_gates = self._get_batch_index(gates)

        gates = gates.view(batch_size, tracks, self.num_experts)
        all_gates = {
            t: gates[:, start:end, :].reshape(-1, self.num_experts).sum(dim=0)
            for t, (start, end) in types_index.items()
        }

        out = out.view(batch_size * tracks, seqlen)
        inputs = out[batch_index]
        inputs = self.input_proj(inputs, expert_size)
        inputs = self.dropout(self.activation(inputs))
        outputs = self.output_proj(inputs, expert_size)

        zeros = torch.zeros_like(out)
        temp_output = zeros.scatter_add_(0, batch_index.unsqueeze(1).expand(-1, seqlen), outputs * batch_gates.unsqueeze(1))
        temp_output = temp_output.view(batch_size, tracks, seqlen)

        y[:, index, :] = temp_output
        y = rearrange(y, "b d n -> b n d")
        return y + residual, all_gates, total_zloss, total_cvloss, weights

    def _compute_gates(self, temp, gate_logits, embedding_logits, task_idx, batch_size):
        """
        计算 gates 的辅助函数，为每种任务类型计算门控网络输出。
        """
        temp_tracks = temp.shape[1]
        temp = temp.view(batch_size * temp_tracks, self.seqlen)

        gates = torch.zeros((batch_size, temp_tracks, self.num_experts), device=temp.device)
        zloss = torch.tensor(0.0, device=temp.device)
        cvloss = torch.tensor(0.0, device=temp.device)
        temp_token = gate_logits[:, task_idx * self.gates_per_type:(task_idx + 1) * self.gates_per_type]
        temp_embedding = embedding_logits[:, task_idx * self.gates_per_type:(task_idx + 1) * self.gates_per_type]
        token_weights = F.softmax(temp_token, dim=1)
        embedding_weights = F.softmax(temp_embedding, dim=1)
        weights = (token_weights + embedding_weights) / 2.0
        zloss = zloss + compute_zloss(temp_token) + compute_zloss(temp_embedding)
        for j in range(self.gates_per_type):
            logits = self.gates[task_idx * self.gates_per_type + j](temp)
            top_k_logits, top_k_indices = torch.topk(logits, self.topk, dim=1)
            zloss += compute_zloss(logits)
            probs = F.softmax(top_k_logits, dim=1)
            temp_gates = torch.zeros((batch_size * temp_tracks, self.num_experts), device=temp.device)
            temp_gates.scatter_(1, top_k_indices, probs)
            # cvloss += compute_cvloss(temp_gates)
            temp_gates = temp_gates.view(batch_size, temp_tracks, self.num_experts)
            temp_gates = temp_gates * weights[:, j].view(batch_size, 1, 1)
            gates = gates + temp_gates

        return gates, zloss, cvloss, weights

    def _get_batch_index(self, gates):
        """
        获取非零 gates 的索引和对应的 batch 索引。
        """
        # 获取非零的 gates 和对应的专家索引
        batch_indices, expert_indices = torch.nonzero(gates, as_tuple=True)  # 获取非零 gates 的索引
        nonzero_gates = gates[batch_indices, expert_indices]  # 提取非零 gates 的值

        # 合并专家索引和对应的 batch 索引
        sorted_experts, sorted_indices = expert_indices.sort()  # 按专家索引排序，返回排序后的专家索引和排序顺序
        batch_index = batch_indices[sorted_indices]  # 按排序顺序重新排列对应的批次索引
        batch_gates = nonzero_gates[sorted_indices]  # 按排序顺序重新排列 gates 的值
        return batch_index, batch_gates

