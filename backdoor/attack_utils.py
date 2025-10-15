import time
import argparse
import numpy as np
import torch
import random

from torch_geometric.datasets import Planetoid,Reddit2,Flickr

# from torch_geometric.loader import DataLoader
from help_funcs import prune_unrelated_edge, prune_unrelated_edge_isolated
import scipy.sparse as sp
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Training settings
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true',
            default=True, help='debug mode')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=10, help='Random seed.')
    parser.add_argument('--model', type=str, default='GCN', help='model',
                        choices=['GCN','GAT','GraphSage','GIN'])
    parser.add_argument('--dataset', type=str, default='Cora', 
                        help='Dataset',
                        choices=['Cora','Citeseer','Pubmed','Flickr','ogbn-arxiv','ogbn-proteins','Physics'])
    parser.add_argument('--train_lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=32, 
                        help='Number of hidden units.')
    parser.add_argument('--thrd', type=float, default=0.5)
    parser.add_argument('--target_class', type=int, default=0) 
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train benign and backdoor model.')
    parser.add_argument('--trojan_epochs', type=int,  default=400, help='Number of epochs to train trigger generator.')
    parser.add_argument('--inner', type=int,  default=1, help='Number of inner')
    
    # backdoor setting
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--trigger_size', type=int, default=3,
                        help='tirgger_size')
    parser.add_argument('--use_vs_number', action='store_true', default=False,
                        help="if use detailed number to decide Vs")
    parser.add_argument('--vs_ratio', type=float, default=0.05,
                        help="ratio of poisoning nodes relative to the full graph")
    parser.add_argument('--vs_number', type=int, default=40,
                        help="number of poisoning nodes relative to the full graph")
    parser.add_argument('--train_ratio', type=float, default=0.2)
    # defense setting
    parser.add_argument('--defense_mode', type=str, default="none",
                        choices=['prune', 'isolate', 'none', 'reconstruct'],
                        help="Mode of defense")
    parser.add_argument('--prune_thr', type=float, default=0.1,
                        help="Threshold of prunning edges")
    parser.add_argument('--target_loss_weight', type=float, default=1,
                        help="Weight of optimize outter trigger generator")
    parser.add_argument('--homo_loss_weight', type=float, default=50,
                        help="Weight of optimize similarity loss")
    parser.add_argument('--homo_boost_thrd', type=float, default=0.5,
                        help="Threshold of increase similarity")
    parser.add_argument('--logging', action='store_true', default=False)
    parser.add_argument('--model_name', type=str, default='GMoE', choices=['GCN', 'GMoE', 'GNNGuard', 'RGCN', 'MedianGCN', 'SimPGCN', 'DPMoE'])

    # attack setting
    parser.add_argument('--dis_weight', type=float, default=1,
                        help="Weight of cluster distance")
    parser.add_argument('--selection_method', type=str, default='cluster_degree',
                        choices=['loss','conf','cluster','none','cluster_degree'],
                        help='Method to select idx_attach for training trojan model (none means randomly select)')
    parser.add_argument('--test_model', type=str, default='GCN',
                        choices=['GCN','GAT','GraphSage','GIN'],
                        help='Model used to attack')
    parser.add_argument('--evaluate_mode', type=str, default='overall',
                        choices=['overall','1by1'],
                        help='Model used to attack')
    parser.add_argument('--attack_method', type=str, default='ugba',
                        choices=['dpgba','ugba','gta'])
    
    # od
    parser.add_argument('--rec_epochs', type=int,  default=100, help='Number of epochs to train benign and backdoor model.')
    parser.add_argument('--threhold', type = float, default=97)

    # GMoE setting
    parser.add_argument('--conv_type', type=str, default='GCN')
    parser.add_argument('--num_experts', type=int, default=8, help='Number of experts')
    parser.add_argument('--topk', type=int, default=2, help='Top-k experts to select')
    parser.add_argument('--router',type=str, default='noisytopk',
                        choices=['noisytopk','cosine'])
    parser.add_argument('--w_div', type=float, default=1)
    parser.add_argument('--lr_pred', type=float, default=1e-3)
    parser.add_argument('--lr_div', type=float, default=1e-3)
    parser.add_argument('--w_mi', type=float, default=1) 
    parser.add_argument('--margin', type=float, default=0.0)
    parser.add_argument('--val_threhold', type=float, default=0.8) # Cora 0.8
    parser.add_argument('--s1_step', type=int, default=1)
    parser.add_argument('--s2_step', type=int, default=1)
    parser.add_argument('--ddp', action='store_true', default=False) 
    parser.add_argument('--cp_epochs', type=int, default=100) 
    parser.add_argument('--noise_std', type=float, default=1) 
    parser.add_argument('--w_bala', type=float, default=1) 
    parser.add_argument('--topk_rerouting', type=int, default=2, help='Top-k experts to select')
    parser.add_argument('--w_certain', type=float, default=1)
    parser.add_argument('--re_epochs', type=int, default=200) 
    parser.add_argument('--re_lr', type=float, default=1e-2)


    # GPU setting
    parser.add_argument('--device_id', type=int, default=0)
    # args = parser.parse_args()
    args = parser.parse_known_args()[0]
    return args

def load_poisoned_files(args, device):
    save_dir = "poisoned_files"
    os.makedirs(save_dir, exist_ok=True)  
    save_filename = f"{args.dataset}_{args.vs_ratio}_{args.selection_method}_" \
                    f"none_{args.attack_method}_backdoor_attack.pth"
    # save_filename = f"{args.dataset}_{args.vs_ratio}_{args.selection_method}_" \
    #                 f"{args.defense_mode}_backdoor_attack.pth"
    save_path = os.path.join(save_dir, save_filename)
    print(f"Loading from: {save_path}")
    loaded_attack_files = torch.load(save_path, map_location=device)

    data = loaded_attack_files["data"]
    poison_x = loaded_attack_files["poison_x"]
    poison_edge_index = loaded_attack_files["poison_edge_index"]
    poison_edge_weights = loaded_attack_files["poison_edge_weights"]
    poison_labels = loaded_attack_files["poison_labels"]
    idx_attach = loaded_attack_files["idx_attach"]
    bkd_tn_nodes = loaded_attack_files["bkd_tn_nodes"]
    idx_val = loaded_attack_files["idx_val"]
    idx_clean_test = loaded_attack_files["idx_clean_test"]
    idx_atk = loaded_attack_files["idx_atk"]
    mask_edge_index = loaded_attack_files["mask_edge_index"]
    model = loaded_attack_files["model"]
    print("Data & model loaded successfully, you can proceed with further testing or evaluation.")
    return data, poison_x, poison_edge_index, poison_edge_weights, poison_labels, idx_attach, bkd_tn_nodes, idx_val, idx_clean_test, idx_atk, mask_edge_index, model

def calculate_asr_fasr_acc(output, idx_atk, idx_clean_test, data, args):
    asr = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()
    flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
    flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()
    import utils
    ca = utils.accuracy(output[idx_clean_test], data.y[idx_clean_test])
    return asr, flip_asr, ca

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#         torch.use_deterministic_algorithms(True)
#         os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

