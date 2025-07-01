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
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true',
            default=True, help='debug mode')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--model', type=str, default='GCN', help='model',
                        choices=['GCN','GAT','GraphSage','GIN'])
    parser.add_argument('--dataset', type=str, default='Cora', 
                        help='Dataset',
                        choices=['Cora','Pubmed','Flickr','ogbn-arxiv'])
    parser.add_argument('--train_lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=32,
                        help='Number of hidden units.')
    parser.add_argument('--thrd', type=float, default=0.5)
    parser.add_argument('--target_class', type=int, default=0) #是原始类别之一
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--epochs', type=int,  default=300, help='Number of epochs to train benign and backdoor model.')
    parser.add_argument('--trojan_epochs', type=int,  default=400, help='Number of epochs to train trigger generator.')
    parser.add_argument('--inner', type=int,  default=1, help='Number of inner')
    # backdoor setting
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--trigger_size', type=int, default=3,
                        help='tirgger_size')
    parser.add_argument('--use_vs_number', action='store_true', default=True,
                        help="if use detailed number to decide Vs")
    parser.add_argument('--vs_ratio', type=float, default=0,
                        help="ratio of poisoning nodes relative to the full graph")
    parser.add_argument('--vs_number', type=int, default=80,
                        help="number of poisoning nodes relative to the full graph")
    # defense setting
    parser.add_argument('--defense_mode', type=str, default="None",
                        choices=['prune', 'isolate', 'none'],
                        help="Mode of defense")
    parser.add_argument('--prune_thr', type=float, default=0.8,
                        help="Threshold of prunning edges")
    parser.add_argument('--target_loss_weight', type=float, default=1,
                        help="Weight of optimize outter trigger generator")
    parser.add_argument('--homo_loss_weight', type=float, default=100,
                        help="Weight of optimize similarity loss")
    parser.add_argument('--homo_boost_thrd', type=float, default=0.8,
                        help="Threshold of increase similarity")
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

    # GMoE setting
    parser.add_argument('--conv_type', type=str, default='GCN')
    parser.add_argument('--num_experts', type=int, default=6, help='Number of experts')
    parser.add_argument('--topk', type=int, default=2, help='Top-k experts to select')
    parser.add_argument('--router',type=str, default='noisytopk',
                        choices=['noisytopk','cosine'])
    parser.add_argument('--w_div', type=float, default=2)
    parser.add_argument('--w_mi', type=float, default=1) # MI训练判别器
    parser.add_argument('--div_margin', type=float, default=0.5)


    # GPU setting
    parser.add_argument('--device_id', type=int, default=5)
    # args = parser.parse_args()
    args = parser.parse_known_args()[0]
    return args

# 看idx_attach在各个专家上的分布，判断是否所有专家都被污染。
def count_experts_selection(topk_indices: torch.Tensor, num_experts) -> torch.Tensor:
    """
    统计每个专家被选中的次数
    
    Args:
        topk_indices: [num_nodes, topk] 包含每个节点选择的专家索引的张量
        num_experts: 专家总数
    
    Returns:
        experts_chosen: [num_experts, 1] 每个专家被选中的次数
    """
    # 展平张量得到所有被选中的专家索引 [num_nodes * topk]
    all_selected = topk_indices.flatten()
    
    # num_experts = all_selected.max().item() + 1

    # print('zhuanjiashu', num_experts)
    
    # 使用bincount统计每个专家被选中的次数
    counts = torch.bincount(
        all_selected,
        minlength=num_experts  # 确保输出长度为num_experts（即使有专家未被选中）
    )
    
    # 转换为 [num_experts, 1] 形状
    experts_chosen = counts.view(-1, 1)
    
    # 打印结果
    print(" | ".join([f"expert{expert_id}: {counts[expert_id].item()}次" for expert_id in range(num_experts)]))
    # for expert_id in range(num_experts):
    #     print(f"expert{expert_id}: {counts[expert_id].item()}次")
    
    return experts_chosen

def compute_poisoning_indicator(A: torch.Tensor, B: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    计算每个专家的中毒程度指标。
    
    参数:
        A (torch.Tensor): [num_success, num_experts] 张量，表示成功攻击节点的专家权重。
        B (torch.Tensor): [num_nodes, num_experts] 张量，表示所有攻击节点的专家权重。
        epsilon (float): 一个小常数，防止除零错误。
    
    返回:
        P (torch.Tensor): [num_experts] 张量，每个元素为相应专家的中毒程度指标。
                     定义为: P_j = mean(A[:, j]) / (mean(B[:, j]) + epsilon)
    """
    mu_succ = A.mean(dim=0)  # 计算每个专家在成功攻击中的平均权重，形状 [num_experts]
    mu_all  = B.mean(dim=0)   # 计算每个专家在所有攻击节点中的平均权重，形状 [num_experts]
    
    P = mu_succ / (mu_all + epsilon)  # 计算指标，避免除零
    print('专家中毒程度：')
    print(", ".join([f"Expert{i}: {P[i].item():.4f}" for i in range(P.shape[0])]))
    return P

def calculate_asr_fasr_acc(output, idx_atk, idx_clean_test, data, args):
    asr = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()
    # print("ASR: {:.4f}".format(asr)) # attach trigger的污染节点输出y_t的概率（测试集，即为ASR）
    flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()] # flip_idx_at指的是idx_atk中原本标签非y_t的污染节点
    flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()
    # print("Flip ASR: {:.4f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0])) # 反转ASR，原本非y_t的污染节点输出y_t的概率 （idx_attach本身有一部分节点标签就是y_t）
    import utils
    ca = utils.accuracy(output[idx_clean_test], data.y[idx_clean_test])
    # print("CA: {:.4f}".format(ca)) # 干净测试样本上准确率acc, # 这个CA的测试数据上附着了trigger，比较客观
    return asr, flip_asr, ca

# def set_seed(seed):
#     """
#     Set the seed for reproducibility.

#     Parameters:
#     seed (int): The seed to set.
#     设置随机数种子以保证结果的可重复性。

#     参数:
#     seed (int): 要设置的种子值。
#     """
#     # 设置Python的随机数生成器种子
#     random.seed(seed)
    
#     # 设置NumPy的随机数生成器种子
#     np.random.seed(seed)
    
#     # 设置PyTorch的CPU随机数生成器种子
#     torch.manual_seed(seed)
    
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     # print(f"Seed set to: {seed}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def set_logging():
    import logging
    import time

    # 1. 生成带时间戳的日志文件名（格式：YYYYMMDD_HHMMSS.log）
    log_filename = time.strftime("logs/%Y%m%d_%H%M%S.log")

    # 2. 配置 logging，每次运行生成新文件
    logging.basicConfig(
        level=logging.INFO,
        # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
         format='%(message)s',
        handlers=[
            logging.FileHandler(log_filename),  # 自动创建新文件
            logging.StreamHandler()            # 同时输出到终端
        ]
    )

    # 3. 测试日志
    logging.info("This log will be saved in a new file: %s", log_filename)
