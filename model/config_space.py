from transformers import PretrainedConfig

class SpaceConfig(PretrainedConfig):
    model_type = "enformer"

    def __init__(
        self,
        dim = 768,      # small
        depth = 11,
        heads = 8,
        output_heads = dict(human = 5313, mouse= 1643),
        target_length = 896,
        attn_dim_key = 64,
        dropout_rate = 0.4,
        attn_dropout = 0.05,
        pos_dropout = 0.01,
        use_checkpointing = False,
        use_convnext = False,
        num_downsamples = 7,    # genetic sequence is downsampled 2 ** 7 == 128x in default Enformer - can be changed for higher resolution
        dim_divisible_by = 128,
        use_tf_gamma = False,
        **kwargs,
    ):
        self.dim: int = dim
        self.depth: int = depth
        self.heads: int = heads
        self.output_heads: dict = output_heads
        self.target_length: int = target_length
        self.attn_dim_key: int = attn_dim_key
        self.dropout_rate: float = dropout_rate
        self.attn_dropout: float = attn_dropout
        self.pos_dropout: float = pos_dropout
        self.use_checkpointing: bool = use_checkpointing
        self.num_downsamples: int = num_downsamples
        self.dim_divisible_by: int = dim_divisible_by
        self.use_tf_gamma: bool = use_tf_gamma
        self.seq_length: int = 131_072
        self.num: int = self.seq_length // (2 ** self.num_downsamples)
        self.MIloss_lambda: float = 0.01
        self.zloss_lambda: float = 0.001
        self.cvloss_lambda: float = 0.001
        self.tracks_topk: int = 3
        self.topk: int = 3
        self.moe: str = "ffn"
        self.species_num_experts: int = 4
        self.tracks_num_experts: int = 15

        super().__init__(**kwargs)