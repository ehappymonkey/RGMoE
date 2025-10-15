
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Flickr
from ogb.nodeproppred import PygNodePropPredDataset

import numpy as np
from torch_geometric.utils import to_undirected

from tdgia_utils import parse_args, get_split, build_adj_from_edge_index, to_edge_form_for_training

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

    from GCN import GCN
    gcn = GCN(nfeat=data.x.shape[1], nhid=32, nclass = max(data.y).item()+1, dropout=0.5, lr=0.01, weight_decay=5e-4, layer=2,device=device).to(device)
    gcn.fit(data.x, data.edge_index, None, data.y, idx_train)
    clean_acc = gcn.test(data.x, data.edge_index, data.edge_weight, data.y, idx_test)
    print(f'Accuracy of Clean GCN: {clean_acc:.4f}')
    acc_clean.append(clean_acc)

acc_clean_mean = np.mean(acc_clean)
acc_clean_std = np.std(acc_clean)
print(f'Acc of Clean GCN: {acc_clean_mean*100:.2f} ± {acc_clean_std*100:.2f}')


print('------------- GCN: Global Poisoning -------------')
from seqgia import SEQGIA
attacker = SEQGIA(epsilon=0.001, n_epoch=500, a_epoch=300, n_inject_max=args.n_inject_max, n_edge_max=args.n_edge_max,
                 feat_lim_min=-1, feat_lim_max=1, injection= 'tdgia', device=device) 
from GCN import GCN_Surrogate
model_surrogate = GCN_Surrogate(data.x.shape[1], 32, max(data.y).item()+1, num_layers=3, dropout=0.5, layer_norm_first=False, use_ln=False).to(device)
adj = build_adj_from_edge_index(data.edge_index, data.edge_weight, data.x.shape[0], 
                                make_undirected=True, device=device,
                                binarize=False, symmetric_reduce="sum")
adj_attack, features_attack = attacker.attack(model_surrogate, adj, data.x, idx_test, labels=None)

edge_index_new, edge_weight_new, x_new = to_edge_form_for_training(
    adj_attack=adj_attack,
    x_orig=data.x,
    features_attack=features_attack,      
    make_undirected=True,                 
    add_self_loops=False,                 
    reduce="sum",                        
    device=device
)
for seed in seeds:
    print(f"Using seed: {seed}")
    set_seed(int(seed))
    from GCN import GCN
    test_model = GCN(nfeat=data.x.shape[1], nhid=32, nclass = max(data.y).item()+1, dropout=0.5, lr=0.01, weight_decay=5e-4, layer=2,device=device).to(device)
    test_model.fit(x_new, edge_index_new, edge_weight_new, data.y, idx_train)
    ptb_acc = test_model.test(x_new, edge_index_new, edge_weight_new, data.y, idx_test)
    print(f"Acc of GCN After Ptb: {ptb_acc:.4f}")
    # Note that the values here a bit more noisy than in the evasion case:
    print(f'PRBCD: Accuracy dropped from {clean_acc:.3f} to {ptb_acc:.3f}')
    acc_ptb.append(ptb_acc)


save_path = f'./tdgia_poisoning_{args.dataset}_{args.n_inject_max}_{args.n_edge_max}.pt'
torch.save({
    'features': x_new,           
    'labels': data.y,                 
    'edge_index': edge_index_new,
    'edge_weight':  edge_weight_new,
    'idx_train': torch.tensor(idx_train),   
    'idx_val': torch.tensor(idx_val),       
    'idx_test': torch.tensor(idx_test),      
}, save_path)
print(f"Saved to {save_path}")