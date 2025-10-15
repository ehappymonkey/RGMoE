import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, n_in, n_h):
        super().__init__()

        self.num_neg = 4 
        self.proj_prev = nn.Linear(n_in, n_h)
        # self.proj_cur = nn.Linear(n_h, proj_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_h, n_h),
            nn.ReLU(),
            nn.Linear(n_h, 1)
        )

    # ---------- full MI loss ---------- #
    def forward(self, h_prev, h_cur, edge_index, idx):
        # print(len(idx))
        src, dst = edge_index
        mask = torch.isin(src, idx) | torch.isin(dst, idx)  
        edge_index = edge_index[:, mask]  

        pos, neg = self.edge_scores(h_prev, h_cur, edge_index)
        mi_edge = self.jsd_per_edge(pos, neg) # [E]
        num_nodes = h_prev.size(0)
        # mi_per_node = edge_mi_to_node_mi(mi_edge, edge_index, num_nodes, use_src=True) 

        mi_per_node = edge_mi_to_node_mi_idx(mi_edge, edge_index,idx, use_src=True)
        
        # print(len(mi_per_node))
        
        return mi_edge, mi_per_node  

    def edge_scores(self, h_prev, h_cur, edge_index):
        src, dst = edge_index                    

        # p_prev = self.proj_prev(h_prev)  # (N, proj_dim)
        # p_cur  = self.proj_cur(h_cur)

        if h_prev.shape[1] == h_cur.shape[1]:
            p_prev = h_prev  
        else:
            p_prev = self.proj_prev(h_prev)
        p_cur = h_cur
        
        pos_samples = torch.cat([p_prev[src], p_cur[dst]], dim=-1)  
        pos_scores = self.mlp(pos_samples).squeeze(-1)  # (E,)
        del pos_samples

        num_nodes = h_prev.size(0)
        E = edge_index.size(1)
        neg_src = torch.randint(0, num_nodes, (self.num_neg, E), device=p_prev.device) 
        neg_samples = torch.cat([p_prev[neg_src], p_cur[dst].unsqueeze(0).expand(self.num_neg, -1, -1)], dim=2) 
        neg_scores = self.mlp(neg_samples).squeeze(-1)    # (num_neg, E)
        neg_scores = neg_scores.reshape(self.num_neg, -1).transpose(0, 1)  # (E, num_neg)
        del neg_samples
        return pos_scores, neg_scores
    
    # ---------- DV bound per‑edge ---------- #
    def dv_per_edge(self, p_samples, n_samples):
        # pos: (E,), neg: (E,K)
        E_p = p_samples
        E_n = torch.logsumexp(n_samples, dim=1) - torch.log(torch.tensor(self.num_neg, device=n_samples.device, dtype=n_samples.dtype))
        return E_p - E_n              

    def jsd_per_edge(self, p_samples, n_samples):
        # p_samples: (E,), n_samples: (E,K)
        log_2 = math.log(2.)
        Ep = log_2 - F.softplus(-p_samples) 
        En = F.softplus(-n_samples) + n_samples - log_2 
        return Ep - En.mean(dim=1)
    
    def infonce_per_edge(self, p_samples, n_samples):
        all_scores = torch.cat([p_samples.unsqueeze(1), n_samples], dim=1)  # (E, K+1)
        
        log_probs = F.log_softmax(all_scores, dim=1)  # (E, K+1)
        return log_probs[:, 0]  # shape: (E,)



from typing import List, Tuple


def edge_mi_to_node_mi_idx(
        dv: torch.Tensor,                 
        edge_index: torch.LongTensor,     
        idx: torch.LongTensor,            
        *, use_src: bool = True           
) -> Tuple[torch.Tensor, ...]:
    dv = dv.contiguous()
    src, dst = edge_index
    nodes = (src if use_src else dst).to(dv.device)      # (E,)

    M = idx.numel()
    # max_id = int(nodes.max().item()) + 1
    max_id = int(torch.max(nodes.max(), idx.max()).item()) + 1
    mapping = -torch.ones(max_id, dtype=torch.long, device=dv.device)
    mapping[idx] = torch.arange(M, device=dv.device)     # idx[k] ↦ k

    mapped = mapping[nodes]                            
    valid_mask = mapped >= 0                            
    mapped = mapped[valid_mask]
    dv      = dv[valid_mask]

    perm = torch.argsort(mapped)
    mapped = mapped[perm]
    dv     = dv[perm]

    deg = torch.bincount(mapped, minlength=M)            # (M,)
    node_mi = torch.split(dv, deg.tolist())             

    return node_mi          


