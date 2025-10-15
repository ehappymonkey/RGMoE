import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


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

