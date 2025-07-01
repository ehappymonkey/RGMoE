
#!/usr/bin/env python
# coding: utf-8

# In[1]: 

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

# Training settings
from attack_utils import parse_args, set_seed

args = parse_args()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))
set_seed(args.seed)
print(args)

def prepare_attack():
    from torch_geometric.utils import to_undirected
    import torch_geometric.transforms as T
    transform = T.Compose([T.NormalizeFeatures()])

    if(args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed'):
        dataset = Planetoid(root='./data/Cora', \
                            name=args.dataset,\
                            transform=transform)
    elif(args.dataset == 'Flickr'):
        dataset = Flickr(root='./data/Flickr/', \
                        transform=transform)
    # elif(args.dataset == 'ogbn-arxiv'):
    #     from ogb.nodeproppred import PygNodePropPredDataset
    #     # Download and process data at './dataset/ogbg_molhiv/'
    #     dataset = PygNodePropPredDataset(name = 'ogbn-arxiv', root='./data/')
    #     split_idx = dataset.get_idx_split() 

    data = dataset[0].to(device)

    if(args.dataset == 'ogbn-arxiv'):
        nNode = data.x.shape[0]
        setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
        # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
        data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
        data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
        data.y = data.y.squeeze(1)
    # we build our own train test split 

    from utils import get_split
    data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,data,device)

    from torch_geometric.utils import to_undirected
    from utils import subgraph
    data.edge_index = to_undirected(data.edge_index)
    train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
    mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]


    from backdoor import Backdoor
    import heuristic_selection as hs

    # from kmeans_pytorch import kmeans, kmeans_predict

    # filter out the unlabeled nodes except from training nodes and testing nodes, nonzero() is to get index, flatten is to get 1-d tensor
    unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()
    if(args.use_vs_number):
        size = args.vs_number
    else:
        size = int((len(data.test_mask)-data.test_mask.sum())*args.vs_ratio)
    print("#Attach Nodes:{}".format(size))
    assert size>0, 'The number of selected trigger nodes must be larger than 0!'
    # here is randomly select poison nodes from unlabeled nodes
    if(args.selection_method == 'none'):
        idx_attach = hs.obtain_attach_nodes(args,unlabeled_idx,size)
    elif(args.selection_method == 'cluster'):
        idx_attach = hs.cluster_distance_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
        idx_attach = torch.LongTensor(idx_attach).to(device)
    elif(args.selection_method == 'cluster_degree'):
        if(args.dataset == 'Pubmed'):
            idx_attach = hs.cluster_degree_selection_seperate_fixed(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
        else:
            idx_attach = hs.cluster_degree_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
        idx_attach = torch.LongTensor(idx_attach).to(device)
    print("idx_attach: {}".format(idx_attach)) #注入后门攻击的目标节点，idx_attach是被污染的节点
    unlabeled_idx = torch.tensor(list(set(unlabeled_idx.cpu().numpy()) - set(idx_attach.cpu().numpy()))).to(device)
    # print(unlabeled_idx) # 这个unlabeled节点干啥用？
    
    # train trigger generator，生成污染后的数据集。model是shadow GNN。
    model = Backdoor(args,device)
    model.fit(data.x, train_edge_index, None, data.y, idx_train, idx_attach, unlabeled_idx)
    poison_x, poison_edge_index, poison_edge_weights, poison_labels = model.get_poisoned() # 注入了触发器的图

    # 净化数据集的方式进行防御。模型在bkd_tn_nodes节点上进行训练
    if(args.defense_mode == 'prune'):
        poison_edge_index,poison_edge_weights = prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,device,large_graph=False)
        bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
    elif(args.defense_mode == 'isolate'):
        poison_edge_index,poison_edge_weights,rel_nodes = prune_unrelated_edge_isolated(args,poison_edge_index,poison_edge_weights,poison_x,device,large_graph=False)
        bkd_tn_nodes = torch.cat([idx_train,idx_attach]).tolist()
        bkd_tn_nodes = torch.LongTensor(list(set(bkd_tn_nodes) - set(rel_nodes))).to(device)
    else:
        bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
    print("precent of left attach nodes: {:.3f}"\
        .format(len(set(bkd_tn_nodes.tolist()) & set(idx_attach.tolist()))/len(idx_attach)))

    return data, poison_x, poison_edge_index, poison_edge_weights, poison_labels, idx_attach, bkd_tn_nodes, idx_val, idx_clean_test, idx_atk, mask_edge_index, model

data, poison_x, poison_edge_index, poison_edge_weights, poison_labels, idx_attach,  bkd_tn_nodes, idx_val, idx_clean_test, idx_atk, mask_edge_index, model = prepare_attack() # 攻击污染节点，prune之后节点

save_dir = "poisoned_files"
os.makedirs(save_dir, exist_ok=True)  
save_filename = f"{args.dataset}_{args.vs_number}_{args.selection_method}_" \
                f"{args.defense_mode}_{args.seed}_backdoor_attack.pth"
save_path = os.path.join(save_dir, save_filename)
                
print(f"Saving backdoor data and model to: {save_path}")

attack_files_dict = {
    "data": data,  # 原始 graph data
    "poison_x": poison_x, # 原始graph node + idx_attach*trigger_size个节点 
    "poison_edge_index": poison_edge_index,
    "poison_edge_weights": poison_edge_weights,
    "poison_labels": poison_labels,
    "idx_attach": idx_attach, # 投毒节点数
    "bkd_tn_nodes": bkd_tn_nodes, # 模型实际训练节点=原训练节点+idx_attach
    "idx_val": idx_val, # val节点
    "idx_clean_test": idx_clean_test, # 干净测试节点， 用于计算CA，test_nodes=idx_clean_test+idx_atk
    "idx_atk": idx_atk, # 中毒测试节点，用于计算ASR
    "mask_edge_index": mask_edge_index,
    "model": model
}

torch.save(attack_files_dict, save_path)
print("Data & model saved successfully.")