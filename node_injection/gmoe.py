

import torch, gc
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from copy import deepcopy
from gmoe_utils import Discriminator, compute_diversity_loss_fast, NoisyTopKGate, accuracy
import logging
import math
from tqdm import trange
from tdgia_utils import parse_args

args = parse_args()


class GCNExpert(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.5):
        super(GCNExpert, self).__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        h1 = self.conv1(x, edge_index, edge_weight)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, self.dropout, training=self.training) 
        h2 = self.conv2(h1, edge_index, edge_weight)
        return h2


class Router_h2(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, num_experts):
        super(Router_h2, self).__init__()
        self.gate = NoisyTopKGate(hidden_dim, num_experts)
        self.gcn_aggregator = GCNExpert(in_dim, hidden_dim, dropout)

    def forward(self, x, edge_index, edge_weight, topk):
        h_gates = self.gcn_aggregator(x, edge_index, edge_weight) # [N, hid_dim]
        gates, topk_indices, full_gates = self.gate(h_gates, top_k=topk) # [N, num_experts]
        L_balance = self.gate.calculate_balancing_loss(h_gates, top_k=topk)
        return gates, topk_indices, full_gates, L_balance


class MoELayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, num_experts, routing='h2'):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            GCNExpert(in_dim, hidden_dim, dropout)
            for _ in range(num_experts)
        ])
        if routing == 'h2':
            self.router = Router_h2(in_dim, hidden_dim, dropout, num_experts)
        self.discriminator = Discriminator(in_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight, w_mi, w_div, topk, idx):
        # h2_experts: [N, E, d]
        h2 = [expert(x, edge_index, edge_weight) for expert in self.experts] # num_experts个[num_nodes, degree]的list
        h2_experts = torch.stack(h2, dim=1)
        gates, topk_indices, full_gates, L_balance = self.router(x, edge_index, edge_weight, topk) 
        # gates = torch.ones(h2_experts.size(0), self.num_experts, device=x.device)/self.num_experts # [N, E]
        h2_moe = torch.sum(gates.unsqueeze(-1)*h2_experts, dim=1) # [N, d]

        if args.dataset == 'ogbn-arxiv' and idx is not None: 
            # print(idx)
            r = int(len(idx) * 0.0001)
            idx_sample = idx[torch.randperm(len(idx))[:r]]
            idx = idx_sample

        elif args.dataset == 'pubmed' and idx is not None:
            # print(idx)
            r = int(len(idx) * 0.01)
            idx_sample = idx[torch.randperm(len(idx))[:r]]
            idx = idx_sample

        elif args.dataset == 'flickr' and idx is not None: 
            # print(idx)
            r = int(len(idx) * 0.01)
            idx_sample = idx[torch.randperm(len(idx))[:r]]
            idx = idx_sample

        elif args.dataset == 'cora' and idx is not None: 
            # print(idx)
            r = int(len(idx) * 1)
            idx_sample = idx[torch.randperm(len(idx))[:r]]
            idx = idx_sample


        if w_mi>0:
            L_mi = self.cal_mi_loss_fast(x, h2, edge_index, idx)
        else:
            L_mi = torch.tensor(0.0, device=x.device, requires_grad=True)

        if w_div>0:
            mi = [self.discriminator(x, h2[i], edge_index, idx) for i in range(self.num_experts)]
            _, mi = zip(*mi) 
            topk_indices_idx = topk_indices[idx]  
            L_div = compute_diversity_loss_fast(mi, topk_indices_idx, margin=args.margin, idx=None) 
        else:
            L_div = torch.tensor(100.0, device=x.device, requires_grad=True)
            # L_div = math.inf

        return h2_moe, gates, topk_indices, L_balance, L_mi, L_div

    def cal_mi_loss_fast(self, h, h2, edge_index, idx):     
        mi_detached = [self.discriminator(h.detach(), h2[i].detach(), edge_index, idx) for i in range(self.num_experts)]
        _, mi_detached  = zip(*mi_detached)
        mi_experts = mi_detached  

        mi_loss_per_expert = 0
        for expert_idx, mi_expert_nodes in enumerate(mi_experts):  # expert_idx in 0 ~ num_experts-1
            mi_all = torch.cat(mi_expert_nodes)                   # [node*degree]
            lengths = torch.tensor([mi.numel() for mi in mi_expert_nodes], device=mi_all.device)  # [N]
            idx = torch.repeat_interleave(torch.arange(len(lengths), device=mi_all.device),lengths)  # node*degree
            out = torch.zeros(len(lengths), device=mi_all.device)
            out.scatter_reduce_(0, idx, mi_all, reduce="mean")    # [N]
            mi_expert = out
            loss = -mi_expert.mean()
            mi_loss_per_expert += loss
        L_mi = mi_loss_per_expert*1/self.num_experts
        return L_mi

        

class GraphMoE(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes,num_experts,dropout=0.5,w_bala=1,w_div=1, w_mi=1):
        super(GraphMoE, self).__init__()
        self.num_experts = num_experts
        self.moe_layer = MoELayer(in_dim, hidden_dim, dropout, num_experts)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.w_bala = w_bala
        self.w_mi = w_mi
        self.w_div = w_div
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight, w_mi=0, w_div=0, idx=None):
        h2_moe, gates, topk_indices, L_balance, L_mi, L_div = self.moe_layer(x, edge_index, edge_weight, w_mi, w_div, args.topk, idx)  
        logits = self.classifier(h2_moe)  # [N, d]->[N, num_classes]
        routing_info = []
        routing_info.append((gates, topk_indices, L_balance, L_mi, L_div))
        return logits, routing_info

    def fit(self, poison_x, poison_edge_index, poison_edge_weights, labels, idx_train, idx_val, idx_test, train_iters=200):
        no_disc_params, gnn_params, disc_params = [],[],[]
        for name, param in self.named_parameters():
            if "discriminator" not in name:
                no_disc_params.append(param)
            if "discriminator" in name: 
                disc_params.append(param)

        # no_disc_param_ids = set(id(p) for p in no_disc_params)
        # no_disc_param_names = [name for name, param in self.named_parameters() if id(param) in no_disc_param_ids]
        # disc_param_ids = set(id(p) for p in disc_params)
        # disc_param_names = [name for name, param in self.named_parameters() if id(param) in disc_param_ids]
        # print("Param names without discriminator:")
        # print(no_disc_param_names)
        # print("Disc params")
        # print(disc_param_names)

        optimizer_disc = torch.optim.AdamW(disc_params, lr=1e-2, weight_decay=5e-4) 
        optimizer_gmoe = torch.optim.AdamW(no_disc_params, lr=1e-2, weight_decay=5e-4) 

        best_epoch=-1
        weights = None
        best_acc = 0
        for epoch in trange(0, train_iters, desc="Training", unit="epoch"):
            self.train() 
            
            # with torch.cuda.amp.autocast(dtype=torch.float16):
            output, routing_info = self.forward(poison_x, poison_edge_index, poison_edge_weights, w_mi=0, w_div=self.w_div, idx=idx_train)
            log_probs = F.log_softmax(output, dim=-1)
            L_balance = routing_info[0][2]
            L_div = routing_info[0][4]
            L_pred = F.nll_loss(log_probs[idx_train], labels[idx_train])
            L_gmoe = L_pred + self.w_bala*L_balance +self.w_div*L_div
            optimizer_gmoe.zero_grad(set_to_none=True)
            L_gmoe.backward()
            optimizer_gmoe.step()

            L_mi=0
            if self.w_mi>0:
                output, routing_info = self.forward(poison_x, poison_edge_index, poison_edge_weights, w_mi=self.w_mi, w_div=0, idx=idx_train)
                L_mi = routing_info[0][3]
                L_disc = self.w_mi*L_mi
                optimizer_disc.zero_grad(set_to_none=True)
                L_disc.backward()
                optimizer_disc.step()
                # lr_scheduler_disc.step()

            self.eval()
            with torch.no_grad():
                output, _ = self.forward(poison_x, poison_edge_index, poison_edge_weights, w_mi=0, w_div=0)
                acc_val = accuracy(output[idx_val], labels[idx_val])
            if epoch %5 ==0:
                # print(f'Epoch: {epoch:03d}, Classify Loss: {L_pred:.4f}, Balance Loss: {L_balance:.4f}, Val Acc: {acc_val:.4f}')
                print(f'Epoch: {epoch:03d}, Classify Loss: {L_pred:.4f}, Balance Loss: {L_balance:.4f}, MI Loss: {self.w_mi*L_mi:.4f}, Diversity Loss: {self.w_div*L_div:.4f}, Val Acc: {acc_val:.4f}')
                asr_experts, ca_experts = self.check_every_expert(poison_x, poison_edge_index, poison_edge_weights, labels, idx_test)
                acc = calculate_acc(output, labels, idx_test)
                print(f"Test Acc: {acc*100:.2f}")

            weights = deepcopy(self.state_dict())

            gc.collect()
            torch.cuda.empty_cache()  # sample model

        if weights is not None:
            self.load_state_dict(weights)
            print(f"Best Val ACC: {best_acc:.4f} at epoch {best_epoch}.")
            # print(f"Lowest L_div (acc>{args.val_threhold}): {lowest_l_div:.4f} at epoch {best_epoch} with Val ACC: {acc_val:.4f}")
        else:
            print("No Valid Model Selected.")

    def check_every_expert(self, features, edge_index, edge_weights, labels, idx_test, log=False):
        self.eval()
        asr_experts, ca_experts = [],[]
        target_acc_experts = []
        with torch.no_grad():
            for i, expert in enumerate(self.moe_layer.experts):
                h2 = expert(features, edge_index, edge_weights)
                logits = self.classifier(h2) 
                acc_i= calculate_acc(logits, labels, idx_test)
                ca_experts.append(round(acc_i * 100, 2))
        if log:
            logging.info('Every CA: %s', ca_experts)
        else:
            print('Every CA: %s', ca_experts)
        return asr_experts, ca_experts

    
    def get_all_expert_outputs(self, h, edge_index, edge_weight):
        self.eval()
        all_expert_h2, all_expert_logits, all_expert_probs = [],[],[]
        with torch.no_grad():
            for i, expert in enumerate(self.moe_layer.experts):
                h2 = expert(h, edge_index, edge_weight)
                logits = self.classifier(h2) 
                log_probs = F.log_softmax(logits, dim=1)
                all_expert_h2.append(h2)
                all_expert_logits.append(logits)
                all_expert_probs.append(log_probs)
        return all_expert_h2, all_expert_logits, all_expert_probs


    def pred_with_random_experts(self, h, edge_index, edge_weight, K):
        self.eval()
        all_expert_h2, all_expert_logits, all_expert_probs = self.get_all_expert_outputs(h, edge_index, edge_weight)
        num_nodes = h.size(0)
        from gmoe_utils import make_random_gates
        gates_random, _ = make_random_gates(num_nodes, self.num_experts, K, device=h.device) # [N, E]
        all_expert_h2 = torch.stack(all_expert_h2, dim=1)
        h2_moe = torch.sum(gates_random.unsqueeze(-1)*all_expert_h2, dim=1)
        logits = self.classifier(h2_moe)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, gates_random


def calculate_acc(logits, labels, idx_test):
    # calculate acc
    acc = accuracy(logits[idx_test], labels[idx_test])
    return acc.item()

