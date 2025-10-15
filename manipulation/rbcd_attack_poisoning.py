import copy
import os.path as osp
import sys
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from rbcd_attack import metric, test, train
from torch import Tensor
from torch.optim import Adam
import torch_geometric.transforms as T
from torch_geometric.contrib.nn import PRBCDAttack
from torch_geometric.datasets import Planetoid, Flickr
from ogb.nodeproppred import PygNodePropPredDataset
import higher
import numpy as np
from torch_geometric.utils import to_undirected
from rbcd_utils import get_split, parse_args


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(args):
    transform = T.Compose([T.NormalizeFeatures()])
    if(args.dataset == 'cora' or args.dataset == 'citeseer' or args.dataset == 'pubmed'):
        dataset = Planetoid('./', args.dataset, transform=transform)
    elif(args.dataset == 'flickr'):
        dataset = Flickr(root='./data/flickr/', \
                        transform=transform)
    elif(args.dataset == 'ogbn-arxiv'):
        dataset = PygNodePropPredDataset(name='ogbn-arxiv')
        split_idx = dataset.get_idx_split() 
    data = dataset[0].to(device)

    if(args.dataset == 'ogbn-arxiv'):
        nNode = data.num_nodes
        setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
        data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
        data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
        data.y = data.y.squeeze(1)
    return data

n_epochs = 50
lr = 0.04
weight_decay = 5e-4

class PoisoningPRBCDAttack(PRBCDAttack):
    def _forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                 **kwargs) -> Tensor:
        """Forward model."""
        self.model.reset_parameters()

        with torch.enable_grad():
            ped = copy.copy(data)
            ped.x, ped.edge_index, ped.edge_weight = x, edge_index, edge_weight
            train(self.model, ped, n_epochs, lr, weight_decay)

        self.model.eval()
        return self.model(x, edge_index, edge_weight)

    def _forward_and_gradient(self, x: Tensor, labels: Tensor,
                              idx_attack: Optional[Tensor] = None,
                              **kwargs) -> Tuple[Tensor, Tensor]:
        """Forward and update edge weights."""
        self.block_edge_weight.requires_grad = True

        self.model.reset_parameters()

        self.model.train()
        opt = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        with higher.innerloop_ctx(self.model, opt, copy_initial_weights=False,track_higher_grads=False) as (fmodel, diffopt):
            edge_index, edge_weight = self._get_modified_adj(
                self.edge_index, self.edge_weight, self.block_edge_index,
                self.block_edge_weight)

            # Normalize only once (only relevant if model normalizes adj)
            if hasattr(fmodel, 'norm'):
                edge_index, edge_weight = fmodel.norm(
                    edge_index,
                    edge_weight,
                    num_nodes=x.size(0),
                    add_self_loops=True,
                )

            for _ in range(n_epochs):
                # pred = fmodel.forward(x, edge_index, edge_weight, skip_norm=True)
                pred = fmodel.forward(x, edge_index, edge_weight)
                loss = F.cross_entropy(pred[data.train_mask],
                                       data.y[data.train_mask])
                diffopt.step(loss)

            pred = fmodel(x, edge_index, edge_weight)
            loss = self.loss(pred, labels, idx_attack)

            gradient = torch.autograd.grad(loss, self.block_edge_weight)[0]

        # Clip gradient for stability:
        clip_norm = 0.5
        grad_len_sq = gradient.square().sum()
        if grad_len_sq > clip_norm:
            gradient *= clip_norm / grad_len_sq.sqrt()

        self.model.eval()

        return loss, gradient

args = parse_args()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))
    
data = load_data(args)
data.edge_index = to_undirected(data.edge_index)
data,idx_train, idx_val, idx_test = get_split(args, data, device, train_ratio=0.2)
print(len(idx_train))
print(len(idx_val))
print(len(idx_test))

rs = np.random.RandomState(args.seed) 
seeds = rs.randint(1000,size=3) 

acc_clean, acc_ptb = [],[]
for seed in seeds:
    print(f"Using seed: {seed}")
    set_seed(int(seed))

    from GCN import GCN, GCN_with_Linear
    gcn = GCN(nfeat=data.x.shape[1], nhid=32, nclass = max(data.y).item()+1, dropout=0.5, lr=0.01, weight_decay=5e-4, layer=2,device=device).to(device)
    # gcn = GCN_with_Linear(in_dim=data.x.shape[1], hidden_dim=32, num_classes=data.y.max().item() + 1).to(device)
    gcn.fit(data.x, data.edge_index, None, data.y, idx_train)
    clean_acc = gcn.test(data.x, data.edge_index, data.edge_weight, data.y, idx_test)
    print(f'Accuracy of Clean GCN: {clean_acc:.4f}')
    acc_clean.append(clean_acc)

acc_clean_mean = np.mean(acc_clean)
acc_clean_std = np.std(acc_clean)
print(f'Acc of Clean GCN: {acc_clean_mean*100:.2f} ± {acc_clean_std*100:.2f}')
print('------------- GCN: Global Poisoning -------------')

prbcd = PoisoningPRBCDAttack(gcn, block_size=2500_000, metric=metric, lr=100)
# prbcd = PoisoningPRBCDAttack(gcn, block_size=args.search_space_size, metric=metric, lr=100)

# PRBCD: Attack test set:
global_budget = int(args.ptb_rate * data.edge_index.size(1) / 2)  # Perturb 5% of edges
ptb_edge_index, perts = prbcd.attack(
    data.x,
    data.edge_index,
    data.y,
    budget=global_budget,
    idx_attack=data.test_mask,
)


for seed in seeds:
    print(f"Using seed: {seed}")
    set_seed(int(seed))
    if args.model_name == 'GCN':
        from GCN import GCN, GCN_with_Linear
        test_model = GCN(nfeat=data.x.shape[1], nhid=32, nclass = max(data.y).item()+1, dropout=0.5, lr=0.01, weight_decay=5e-4, layer=2,device=device).to(device)
        # test_model = GCN_with_Linear(in_dim=data.x.shape[1], hidden_dim=32, num_classes=data.y.max().item() + 1).to(device)
        test_model.fit(data.x, ptb_edge_index, data.edge_weight, data.y, idx_train)
        ptb_acc = test_model.test(data.x, ptb_edge_index, data.edge_weight, data.y, idx_test)
        print(f"Acc of GCN After Ptb: {ptb_acc:.4f}")
    # Note that the values here a bit more noisy than in the evasion case:
    print(f'PRBCD: Accuracy dropped from {clean_acc:.3f} to {ptb_acc:.3f}')
    acc_ptb.append(ptb_acc)


save_path = f'./rbcd_poisoning_{args.dataset}_{args.ptb_rate}.pt'
torch.save({
    'features': data.x,             
    'labels': data.y,                
    'edge_index': ptb_edge_index,
    'edge_weight':  data.edge_weight,
    'idx_train': torch.tensor(idx_train),   
    'idx_val': torch.tensor(idx_val),       
    'idx_test': torch.tensor(idx_test),      
}, save_path)
print(f"Saved to {save_path}")
