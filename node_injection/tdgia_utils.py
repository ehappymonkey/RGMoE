import numpy as np
import torch.nn.functional as F
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer','pubmed','flickr','ogbn-arxiv'], help='dataset')
    parser.add_argument('--ptb_rate', type=float, default=0.1, help='pertubation rate')
    parser.add_argument('--search_space_size', type=int, default=2_500_000)
    parser.add_argument('--model_name', type=str, default='RGCN', choices=['GCN', 'RGCN', 'GMoE', 'MedianGCN', 'SoftMedianGCN', 'SimPGCN', 'GNNGuard', 'DPMoE'], help='model name')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--n_inject_max', type=int, default=60, help='maximum number of nodes to inject')
    parser.add_argument('--n_edge_max', type=int, default=20, help='maximum number of edges per injected node')
    # GMoE
    parser.add_argument('--num_experts', type=int, default=12)
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--router',type=str, default='noisytopk',
                        choices=['noisytopk','cosine'])
    parser.add_argument('--w_div', type=float, default=1)
    parser.add_argument('--lr_pred', type=float, default=1e-3)
    parser.add_argument('--lr_div', type=float, default=1e-3)
    parser.add_argument('--w_mi', type=float, default=1) 
    parser.add_argument('--margin', type=float, default=0.0)
    parser.add_argument('--val_threhold', type=float, default=0.82) # Cora 0.8
    parser.add_argument('--noise_std', type=float, default=1) 
    parser.add_argument('--w_bala', type=float, default=1) 
    parser.add_argument('--topk_rerouting', type=int, default=32, help='Top-k experts to select')
    parser.add_argument('--w_certain', type=float, default=1)

    parser.add_argument('--lam_simpgcn', type=float, default=5, help='lambda for simpgcn')
    parser.add_argument('--gam_simpgcn', type=float, default=0.1, help='gamma for simpgcn')
    
    # GPU setting
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_known_args()[0]
    return args

args = parse_args()

# from UGBA, 0.2/0.1/0.7。
def get_split(args, data, device, train_ratio=0.2):
    rs = np.random.RandomState(args.seed)
    perm = rs.permutation(data.num_nodes)
    train_number = int(train_ratio*len(perm)) 
    idx_train = torch.tensor(sorted(perm[:train_number])).to(device)

    train_number_all = int(0.8*len(perm)) 
    idx_train_all = torch.tensor(sorted(perm[:train_number_all])).to(device)
    data.train_mask = torch.zeros_like(data.train_mask)
    data.train_mask[idx_train] = True

    val_number = int(0.1*len(perm))
    idx_val = torch.tensor(sorted(perm[train_number:train_number+val_number])).to(device)
    data.val_mask = torch.zeros_like(data.val_mask)
    data.val_mask[idx_val] = True

    test_number = int(0.2*len(perm))
    idx_test = torch.tensor(sorted(perm[train_number+val_number:train_number+val_number+test_number])).to(device)
    data.test_mask = torch.zeros_like(data.test_mask)
    data.test_mask[idx_test] = True
    return data, idx_train, idx_val, idx_test 


