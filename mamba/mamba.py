import torch
import torch.nn.functional as F
from mamba_ssm.models.mixer_seq_simple import MixerModel
from torch import nn
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.text import Perplexity
from tqdm import trange

from composer.models import ComposerModel

from .datatrove import MambaBatch

SEED = 42


class MambaModel(ComposerModel):

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_intermediate: int,
        n_layer: int,
        fsdp_layer_wrap: bool,
        activation_checkpointing: bool,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        torch.manual_seed(SEED)
        self.backbone = MixerModel(
            d_model=d_model,
            d_intermediate=d_intermediate,
            n_layer=n_layer,
            vocab_size=vocab_size,
            dtype=dtype,
            fused_add_norm=True,
            rms_norm=True,
        )

        # convert D and A_log in backbone to same dtype as the rest of model for fsdp compability
        self.backbone.to(dtype=dtype)
        self.projection = nn.Linear(d_model, vocab_size, bias=False, dtype=dtype)

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_intermediate = d_intermediate
        self.num_layers = n_layer
        self.d_state = self.backbone.layers[0].mixer.d_state
        self.dt_rank = self.backbone.layers[0].mixer.dt_rank
        self.d_inner = self.backbone.layers[0].mixer.d_inner

        self.fsdp_layer_wrap = fsdp_layer_wrap
        self.activation_checkpointing = activation_checkpointing

        # metrics
        self.top10_accuracy = MulticlassAccuracy(num_classes=vocab_size, top_k=10)
        self.top3_accuracy = MulticlassAccuracy(num_classes=vocab_size, top_k=3)
        self.perplexity = Perplexity()

        for i, block in enumerate(self.backbone.layers):
            if self.fsdp_layer_wrap:
                print(f"Layer {i} is being FSDP wrapped")
                block._fsdp_wrap = True  # marking the block for FSDP wrapping
            if self.activation_checkpointing:
                print(f"Layer {i} is being activation checkpointed")
                block._activation_checkpointing = True  # marking the block for AC

    def forward(self, batch: MambaBatch):
        x = self.backbone(batch.input_ids)
        x = self.projection(x)
        return x

    def complete(self, tokens, n_tokens=10):
        with torch.no_grad():
            for _ in trange(n_tokens):
                x = self.backbone(tokens)
                x = self.projection(x)
                new_token = x[:, -1, :].argmax()
                tokens = torch.cat([tokens, new_token.unsqueeze(0).unsqueeze(0)], dim=1)

        return tokens

    def loss(self, outputs, batch):
        targets = batch.target_ids

        if batch.loss_factors is None:
            return F.cross_entropy(
                outputs.transpose(-1, -2), targets, reduction="none"
            ).mean()
        else:
            return (
                F.cross_entropy(outputs.transpose(-1, -2), targets, reduction="none")
                * batch.loss_factors
            ).sum() / batch.loss_norm

    def eval_forward(self, batch, outputs=None):
        if outputs is not None:
            return outputs
        outputs = self(batch)
        return outputs

    def update_metric(self, batch, outputs, metric):
        targets = batch.target_ids
        if isinstance(metric, Perplexity):
            metric.update(outputs, targets)
        else:
            metric.update(outputs.transpose(-1, -2), targets)

    def get_metrics(self, is_train=False):
        # defines which metrics to use in each phase of train/eval
        return (
            {
                "TOP3_Accuracy": self.top3_accuracy,
                "TOP10_Accuracy": self.top10_accuracy,
                "Perplexity": self.perplexity,
            }
            if not is_train
            else {}
        )

    def flops_per_batch(self, batch):
        """
        Calculate total number of FLOPs of Mamba based on https://github.com/state-spaces/mamba/issues/110.
        Terms such as nonlinearities, biases, and layer normalization are omitted (https://arxiv.org/pdf/2001.08361.pdf).
        Very similar to Chinchilla in that it includes embeddings.
        """
        x = batch.input_ids
        batch_size, seq_len = x.shape

        # embeddings (uncomment if you want to include the embeddings in the total flops)
        # embeddings = 2 * seq_len * vocab_size * d_model

        # selective scan
        scan = 9 * seq_len * self.d_state * self.d_model

        # linear projections
        in_proj = 2 * seq_len * self.d_model * self.d_inner * 2
        x_proj = 2 * seq_len * self.d_inner * (self.dt_rank + self.d_state * 2)
        dt_proj = 2 * seq_len * self.dt_rank * self.d_inner
        out_proj = 2 * seq_len * self.d_inner * self.d_model

        # output projection
        projection = 2 * seq_len * self.vocab_size * self.d_model

        forward_flops = (
            self.num_layers * (in_proj + scan + x_proj + dt_proj + out_proj)
            + projection
        )
        backward_flops = 2 * forward_flops
        total_flops = forward_flops + backward_flops

        return total_flops * batch_size
