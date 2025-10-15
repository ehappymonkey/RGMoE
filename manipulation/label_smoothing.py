import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math



class PurifiedGMoE(nn.Module):
    def __init__(self, gmoe, top_k, device):
        super(PurifiedGMoE, self).__init__()   
        self.gmoe = gmoe
        self.device = device 
        self.num_experts = self.gmoe.moe_layer.num_experts
        self.top_k = top_k 
        self.purified_router = copy.deepcopy(self.gmoe.moe_layer.router)
        self.purified_router.noise_gate = False

    def fit(self, x, edge_index, edge_weight, idx_train, idx_hat_uncertain, soft_labels, w_certain=0, epochs=50, lr=0.01):
        self.gmoe.eval()

        with torch.no_grad():     
            all_expert_h2, _, _ = self.gmoe.get_all_expert_outputs(x, edge_index, edge_weight)
            h2_stack = torch.stack(all_expert_h2, dim=1)

        idx_train = (
            torch.tensor(idx_train, device=self.device)
            if not torch.is_tensor(idx_train)
            else idx_train
        )
        idx_hat_uncertain = (
            torch.tensor(idx_hat_uncertain, device=self.device)
            if not torch.is_tensor(idx_hat_uncertain)
            else idx_hat_uncertain
        )
        idx_hat_certain = idx_train[~torch.isin(idx_train, idx_hat_uncertain)]

        weights=None
        best_loss_uncertain = math.inf
        optimizer = torch.optim.AdamW(self.purified_router.parameters(), lr=lr)
        for epoch in range(epochs):
            self.purified_router.train()
            optimizer.zero_grad()
            sparse_gates, _, full_gates, _ = self.purified_router(x, edge_index, edge_weight, self.top_k)  # [N, E]
            #h_moe_new = torch.sum(sparse_gates.unsqueeze(-1) * h2_stack, dim=1)  # [N, D]
            h_moe_new = torch.sum(full_gates.unsqueeze(-1) * h2_stack, dim=1) 
            logits = self.gmoe.classifier(h_moe_new)
            log_probs = F.log_softmax(logits, dim=1)


            loss_certain = F.cross_entropy(logits[idx_hat_certain], soft_labels[idx_hat_certain])
            loss_uncertain = (-(soft_labels[idx_hat_uncertain] * log_probs[idx_hat_uncertain]).sum(dim=1)).mean()

            loss = loss_uncertain + w_certain*loss_certain
            loss.backward()
            optimizer.step()
            self.purified_router.eval()
            if (epoch + 1) % 20 == 0:
                print(f"[Epoch {epoch+1}] KLDiv Loss: {loss_uncertain.item():.4f} | CE Loss: {loss_certain:.4f}")
            if loss_uncertain.item() < best_loss_uncertain:
                best_loss_uncertain = loss_uncertain.item()
                best_epoch = epoch + 1
                best_state_dict = copy.deepcopy(self.purified_router.state_dict())
        self.purified_router.load_state_dict(best_state_dict)
        print(f"Best model at epoch {best_epoch} with uncertain KLDiv loss: {best_loss_uncertain:.4f}")

    def forward(self, x, edge_index, edge_weight, args, rerouting=True):

        all_expert_h2, _, _ = self.gmoe.get_all_expert_outputs(x, edge_index, edge_weight)
        h2_stack = torch.stack(all_expert_h2, dim=1)
        routing_info = []
        if rerouting:
            sparse_gates_new, topk_indices_new, _, _ = self.purified_router(x, edge_index, edge_weight, self.top_k)
            h_moe_new = torch.sum(sparse_gates_new.unsqueeze(-1) * h2_stack, dim=1) 
            routing_info.append((sparse_gates_new, topk_indices_new))
            logits = self.gmoe.classifier(h_moe_new)
        else:
            sparse_gates_ori, topk_indices_ori, _,_ = self.gmoe.moe_layer.router(x, edge_index, edge_weight, topk=args.topk) 
            routing_info.append((sparse_gates_ori, topk_indices_ori))
            logits, _ = self.gmoe(x, edge_index, edge_weight, w_mi=0, w_div=0)
        log_probs = F.log_softmax(logits, dim=1) 
        return log_probs, routing_info




import torch, numpy as np
import matplotlib.pyplot as plt



import torch

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def get_smooth_soft_label_2(
    y,
    expert_probs,
    original_prob,
    uncertain_indices,  
    smoothing=0.1,
    eps=1e-12
):

    device = original_prob.device
    dtype = original_prob.dtype
    num_nodes, num_classes = original_prob.shape

    expert_probs = [torch.exp(ep.to(device=device, dtype=dtype)) for ep in expert_probs]
    original_prob = torch.exp(original_prob)  # [num_nodes, num_classes]

    soft_label = torch.stack(expert_probs, dim=0).mean(dim=0)

    def _to_index(idx_list):
        if isinstance(idx_list, torch.Tensor):
            idx = idx_list.to(device=device, dtype=torch.long).view(-1)
        elif isinstance(idx_list, (list, tuple)):
            parts = []
            for it in idx_list:
                if isinstance(it, torch.Tensor):
                    parts.append(it.to(device=device, dtype=torch.long).view(-1))
                else:
                    parts.append(torch.as_tensor(it, device=device, dtype=torch.long).view(-1))
            idx = torch.cat(parts, dim=0) if len(parts) > 0 else torch.empty(0, dtype=torch.long, device=device)
        else:
            raise ValueError("uncertain_indices need to be tensor list / Tensor。")

        if idx.numel() == 0:
            return idx
        idx = idx[(idx >= 0) & (idx < num_nodes)]
        return torch.unique(idx)

    idx_uncertain = _to_index(uncertain_indices)

    all_idx = torch.arange(num_nodes, device=device)
    if idx_uncertain.numel() == 0:
        idx_certain = all_idx
    else:
        mask_uncertain = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        mask_uncertain[idx_uncertain] = True
        idx_certain = all_idx[~mask_uncertain]

    soft_label_c = soft_label.clone()


    if idx_uncertain.numel() > 0:
        top_cls = original_prob.argmax(dim=1)  # [num_nodes]
        soft_label_c[idx_uncertain, top_cls[idx_uncertain]] = (
            soft_label_c[idx_uncertain, top_cls[idx_uncertain]] * smoothing
        )
        denom = soft_label_c[idx_uncertain].sum(dim=1, keepdim=True).clamp_min(eps)
        soft_label_c[idx_uncertain] = soft_label_c[idx_uncertain] / denom

    if y is not None and isinstance(y, torch.Tensor) and y.shape[0] == num_nodes:
        y_use = y.to(device=device, dtype=torch.long)
        one_hot_all = F.one_hot(y_use, num_classes=num_classes).to(dtype=dtype, device=device)
        soft_label_c[idx_certain] = one_hot_all[idx_certain]

    else:
        top_cls_all = original_prob.argmax(dim=1)  # [num_nodes]
        one_hot_all = F.one_hot(top_cls_all, num_classes=num_classes).to(dtype=dtype, device=device)
        soft_label_c[idx_certain] = one_hot_all[idx_certain]

    return soft_label, soft_label_c


def get_smooth_soft_label(y, uncertain_indices, num_classes, smoothing=0.9):

    soft_labels = torch.zeros((len(y), num_classes), 
                            dtype=torch.float32, 
                            device=y.device) 
    
    smooth_val = smoothing / (num_classes - 1)
    soft_labels.fill_(smooth_val)
    soft_labels[torch.arange(len(y)), y] = 1.0 - smoothing  
    

    uncertain_indices = torch.tensor([v.item() for v in uncertain_indices], device=y.device)  

    mask = torch.ones(len(y), dtype=torch.bool, device=y.device)
    mask[uncertain_indices] = False  
    print(soft_labels[mask].shape)
    # soft_labels[mask] = torch.zeros(num_classes, device=y.device)
    # soft_labels[mask, y[mask]] = 1.0  # one-hot

    soft_labels[mask] = torch.zeros_like(soft_labels[mask], device=y.device)  
    soft_labels[mask, y[mask]] = 1.0                         

    return soft_labels  


@torch.no_grad()
def analyze_expert_variance(idx, moe_log_probs, all_expert_log_probs, num_bins=5, chunk_size=None):
    device = moe_log_probs.device
    idx = torch.as_tensor(idx, device=device, dtype=torch.long)

    if isinstance(all_expert_log_probs, list):
        exp_log = torch.stack(all_expert_log_probs, dim=0).to(device)  # [E,N,C]
    else:
        exp_log = all_expert_log_probs.to(device)                      # [E,N,C]
    E, N, C = exp_log.shape
    M = idx.numel()


    if chunk_size is None:
        chunk_size = M  

    out = []
    for s in range(0, M, chunk_size):
        e = min(M, s + chunk_size)


        logQ = moe_log_probs.index_select(0, idx[s:e])          # [m,C]
        logP = exp_log.index_select(1, idx[s:e])                # [E,m,C]

        Q = logQ.exp().unsqueeze(0)                             # [1,m,C]
        kl_em = (Q * (logQ.unsqueeze(0) - logP)).sum(dim=-1)    # [E,m]
        node_kls_chunk = kl_em.sum(dim=0)                       # [m]

        out.append(node_kls_chunk)

    node_kls = torch.cat(out, dim=0)                            # [M]
    variances = node_kls.detach().cpu().numpy()
    return variances

