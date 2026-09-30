import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv, GlobalAttentionPooling
import dgl
import random
import numpy as np
import MMD

class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_random_seed(42)

class AdversarialGAT(nn.Module):
    def __init__(self,gcn_hidden,  vocab_size, embed_dim = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

        self.gcn = GraphConv(embed_dim, 100, allow_zero_in_degree=True)

        self.fc = nn.Linear(100, gcn_hidden)

        self.gate_nn = nn.Sequential(
            nn.Linear(100, 32),
            # nn.LayerNorm(32),
            nn.Tanh(),
            # nn.Dropout(0.5),
            nn.Linear(32, 1)
        )
        self.pooling = GlobalAttentionPooling(self.gate_nn)

        self.domain_clf = nn.Sequential(
            nn.Linear( gcn_hidden, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )

        self.defect_clf = nn.Sequential(
            nn.Linear(gcn_hidden, 1),
            # nn.Linear(32, 16),
            # nn.Tanh(),
            # nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def process_graph(self, g, x):
        x = self.gcn(g, x)
        x = torch.relu(x)
        g.ndata['h'] = x
        return dgl.max_nodes(g, 'h')

    def forward(self, g, g_src, g_tar, alpha=1.0, mask=None, compute_node_mmd=True):

        x = self.embed(g.ndata['feat'].long())
        x_src = self.embed(g_src.ndata['feat'].long())
        x_tar = self.embed(g_tar.ndata['feat'].long())

        if mask is not None:
            x = x * mask.unsqueeze(-1)

        x = self.process_graph(g, x)
        x_src = self.process_graph(g_src, x_src)
        x_tar = self.process_graph(g_tar, x_tar)

        x_batch_mmd = self.fc(x)
        x_src_mmd = self.fc(x_src)
        x_tar_mmd = self.fc(x_tar)

        src_node_feat = self.fc(g_src.ndata['h'])  # shape: [N_src, 32]
        tgt_node_feat = self.fc(g_tar.ndata['h'])  # shape: [N_tgt, 32]

        if compute_node_mmd:
            x_loss_mmd_node = compute_node_pair_mmd(
            g_src.ndata['feat'], g_tar.ndata['feat'], src_node_feat, tgt_node_feat)
        else:
            x_loss_mmd_node = torch.tensor(0.0, device=g.device)


        defect_prob = self.defect_clf(x_batch_mmd)

        #reversed_feature = GradientReverse.apply(x_batch_mmd, alpha)

        #domain_logits = self.domain_clf(reversed_feature)

        return x_batch_mmd, defect_prob, x_src_mmd, x_tar_mmd, x_loss_mmd_node

def compute_node_pair_mmd(src_types, tgt_types, src_feats, tgt_feats, min_type_count = 5, max_nodes_per_type = 25):

    src_type_set = set(src_types.tolist())
    tgt_type_set = set(tgt_types.tolist())
    common_types = sorted(list(src_type_set & tgt_type_set))

    total_mmd = 0.0
    type_count = 0

    for t in common_types:
        if t == 0:
            continue

        src_mask = (src_types == t)
        tgt_mask = (tgt_types == t)

        if src_mask.sum() < min_type_count or tgt_mask.sum() < min_type_count:
            continue

        src_feat_t = src_feats[src_mask]
        tgt_feat_t = tgt_feats[tgt_mask]

        if src_feat_t.size(0) > max_nodes_per_type:
            idx = torch.randperm(src_feat_t.size(0))[:max_nodes_per_type]
            src_feat_t = src_feat_t[idx]
        if tgt_feat_t.size(0) > max_nodes_per_type:
            idx = torch.randperm(tgt_feat_t.size(0))[:max_nodes_per_type]
            tgt_feat_t = tgt_feat_t[idx]

        mmd_t = MMD.mmd_loss(src_feat_t, tgt_feat_t)
        if torch.isnan(mmd_t) or torch.isinf(mmd_t):
            continue

        total_mmd += mmd_t
        type_count += 1

    if type_count == 0:
        return torch.tensor(0.0, device=src_feats.device)
    else:
        return total_mmd / type_count
