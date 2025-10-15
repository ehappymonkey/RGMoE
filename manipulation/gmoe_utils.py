import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def accuracy(output, labels):
    """Return accuracy of output compared to labels.
    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels
    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


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



from typing import Tuple


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


class NoisyTopKGate(nn.Module):
    def __init__(self, in_dim, num_experts, weight_importance=1, weight_load=1, noise_std=0):
        super(NoisyTopKGate, self).__init__()
        self.num_experts = num_experts
        self.noise_std = 0 # noise_std
        self.noise_gate = True

        self.W_g = nn.Parameter(torch.Tensor(in_dim, num_experts))
        self.W_n = nn.Parameter(torch.Tensor(in_dim, num_experts))
        nn.init.xavier_uniform_(self.W_g)
        nn.init.xavier_uniform_(self.W_n)

        self.weight_importance = weight_importance
        self.weight_load = weight_load
        self.normal = Normal(0, 1)

    def forward(self, h, top_k):

        # Q = torch.matmul(h, self.W_g) + self.noise_std * F.softplus(torch.matmul(h, self.W_n))
        clean_logits = torch.matmul(h, self.W_g)
        noise_std = F.softplus(torch.matmul(h, self.W_n))+1e-2
        if self.noise_gate:
            Q = clean_logits + (torch.randn_like(clean_logits) * noise_std)
        else:
            Q = clean_logits

        full_gates = F.softmax(Q, dim=1)
        topk_values, topk_indices = torch.topk(full_gates, top_k, dim=1)
        topk_gates = F.softmax(topk_values, dim=1)
        sparse_gates = torch.zeros_like(full_gates)
        sparse_gates.scatter_(1, topk_indices, topk_gates)
        return sparse_gates, topk_indices, full_gates

    def cal_importantloss(self, full_gates):
        importance = full_gates.sum(dim=0)  # [num_experts]
        mean = importance.mean()
        std = importance.std()
        loss = self.weight_importance * (std / (mean + 1e-8))**2
        return loss

    def cal_load_loss(self, h, top_k):
        """
        h: [N, in_dim]
        return: scalar load-loss
        """
        H     = h @ self.W_g                     # [N,E]
        sigma = F.softplus(h @ self.W_n).clamp_min(1e-6)   # [N,E]

        H_sorted, idx_sorted = H.sort(dim=1, descending=True)        # [N,E]
        rank = idx_sorted.argsort(dim=1)                            

        kth = H_sorted[:, top_k-1]          
        kth_plus1 = H_sorted[:, top_k] if top_k < H.shape[1] \
                    else kth                    
        T_val = torch.where(rank < top_k,   
                            kth_plus1.unsqueeze(1),
                            kth.unsqueeze(1))    # [N,E]

        z   = (H - T_val) / sigma                # [N,E]
        P   = 0.5 * (1 + torch.erf(z / 1.41421356237))  # Normal(0,1).cdf(z)
        load = P.sum(dim=0)                      # [E]
        cv2  = (load.std() / (load.mean() + 1e-6)).pow(2)
        return self.weight_load * cv2

    def calculate_balancing_loss(self, h, top_k):
        _, _, full_gates = self.forward(h, top_k)
        imp_loss = self.cal_importantloss(full_gates)
        l_loss = self.cal_load_loss(h, top_k)
        total_loss = imp_loss + l_loss
        return total_loss


def make_random_gates(num_nodes: int,
                      num_experts: int,
                      K: int,
                      device=None,
                      dtype=torch.float32,
                      generator: torch.Generator | None = None):
    device = torch.device(device) if device is not None else torch.device("cpu")

    gates_random = torch.zeros(num_nodes, num_experts, device=device, dtype=dtype)
    if K == 0:
        topk_indices = torch.empty(num_nodes, 0, dtype=torch.long, device=device)
        return gates_random, topk_indices

    scores = torch.rand((num_nodes, num_experts), device=device, generator=generator)
    _, topk_indices = scores.topk(K, dim=1)  # [num_nodes, K]

    gates_random.scatter_(1, topk_indices, 1.0 / K)
    return gates_random, topk_indices


from itertools import combinations

def compute_diversity_loss_fast(
        mi_per_node: list,          # len = num_experts; 
        topk_indices: torch.Tensor, # [N, K]
        margin: float = 0.5, 
        idx = None) -> torch.Tensor:

    if idx is not None:
        idx_set = set(idx.tolist())
        mi_per_node = [
            [mi for i, mi in enumerate(mi_expert) if i in idx_set]
            for mi_expert in mi_per_node]
        topk_indices = topk_indices[idx]
            
    if topk_indices.ndim != 2:
        raise ValueError("topk_indices must be 2-D [num_nodes, K]")
    K = topk_indices.size(1)


    device     = topk_indices.device
    num_nodes  = topk_indices.size(0)
    n_pairs    = K * (K - 1) // 2         


    deg = torch.tensor([mi_per_node[0][v].numel() for v in range(num_nodes)],
                       device=device)
    valid_mask = deg >= 2                 
    if not valid_mask.any():
        return torch.tensor(0., device=device, requires_grad=True)

    row_idx_list, x_list, y_list = [], [], []
    pair_combo = list(combinations(range(K), 2))        

    for v in range(num_nodes):
        d = deg[v].item()
        if d < 2:
            continue
        for pair_idx, (i, j) in enumerate(pair_combo):
            row_id = v * n_pairs + pair_idx           
            row_idx_list.append(torch.full((d,), row_id, device=device))

            e1, e2 = int(topk_indices[v, i]), int(topk_indices[v, j])
            x_list.append(mi_per_node[e1][v])           
            y_list.append(mi_per_node[e2][v])

    row_idx = torch.cat(row_idx_list)                     # [E_total]
    x       = torch.cat(x_list).to(device=device, dtype=torch.float32)
    y       = torch.cat(y_list).to(device=device, dtype=torch.float32)

    total_rows = num_nodes * n_pairs                      
    dot = torch.zeros(total_rows, device=device) \
          .scatter_reduce_(0, row_idx, x * y, reduce="sum")
    nx2 = torch.zeros_like(dot) \
          .scatter_reduce_(0, row_idx, x * x, reduce="sum")
    ny2 = torch.zeros_like(dot) \
          .scatter_reduce_(0, row_idx, y * y, reduce="sum")


    valid_rows_mask = torch.repeat_interleave(valid_mask, n_pairs)
    dot, nx2, ny2 = dot[valid_rows_mask], nx2[valid_rows_mask], ny2[valid_rows_mask]

    eps  = 1e-12
    sim  = dot / ((nx2 + eps).sqrt() * (ny2 + eps).sqrt())
    loss = torch.relu(sim - margin).mean()
    return loss

