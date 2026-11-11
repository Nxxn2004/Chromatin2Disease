"""Graph Attention Network for trace classification (CS224W proposal, Sec. 2).

    h_v  <- act( MSG_self(h_v) + sum_u alpha_vu MSG(h_u) )       single-head GATConv + self-skip, n layers
    h_G   = ELU( sum_v softmax_v(Linear(h_v)) * MSG(h_v) )        virtual-node readout
    y_hat = sigmoid( w_t . h_G + b_t )                            one head per task

alpha_vu = softmax_u( LeakyReLU(a . [W h_v || W h_u]) ). With attn_leaky=False the LeakyReLU slope is set to 1,
which makes it the identity, so the score is linear in the node features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import scatter, softmax

ACTIVATIONS = {"elu": F.elu, "leaky_relu": F.leaky_relu, "relu": F.relu, "sigmoid": torch.sigmoid}


class ChromatinGAT(nn.Module):
    def __init__(self, n_bins, n_tasks=2, hidden=64, num_layers=3, dropout=0.2, act="elu", attn_leaky=True):
        super().__init__()
        self.dropout, self.act = dropout, ACTIVATIONS[act]
        self.encoder = nn.Linear(2 + 3, hidden)      # x (2 features) + centred xyz
        self.bin_emb = nn.Embedding(n_bins, hidden)  # which DNA segment
        self.convs = nn.ModuleList(GATConv(hidden, hidden, add_self_loops=False, negative_slope=0.2 if attn_leaky else 1.0)
                                   for _ in range(num_layers))
        self.skips = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(num_layers))
        self.gate = nn.Linear(hidden, 1)
        self.msg = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, n_tasks)

    def forward(self, data, return_attention=False):
        h = self.encoder(torch.cat([data.x, data.pos], -1)) + self.bin_emb(data.bin)
        for conv, skip in zip(self.convs, self.skips):
            h = F.dropout(self.act(skip(h) + conv(h, data.edge_index)), self.dropout, self.training)
        alpha = softmax(self.gate(h).squeeze(-1), data.batch)
        hg = F.elu(scatter(alpha[:, None] * self.msg(h), data.batch, 0, dim_size=data.num_graphs, reduce="sum"))
        logits = self.head(F.dropout(hg, self.dropout, self.training)).gather(1, data.task[:, None]).squeeze(1)
        return (logits, alpha) if return_attention else logits


def random_rotation(pos, batch, num_graphs):
    """Rotate each graph's coordinates by its own random 3D rotation."""
    q = torch.linalg.qr(torch.randn(num_graphs, 3, 3, device=pos.device)).Q
    q[:, :, 0] *= torch.linalg.det(q).sign()[:, None]  # det = +1
    return torch.einsum("nij,nj->ni", q[batch], pos)
