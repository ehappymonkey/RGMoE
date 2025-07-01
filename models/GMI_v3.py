# 估计每个节点邻居的MI，并进行多样化约束, 可以work。
import torch
import torch.nn as nn
import torch_scatter
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from typing import List 
import math

class GNNSingleLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout = 0.5):
        super(GNNSingleLayer, self).__init__()
        self.conv = GCNConv(in_dim, out_dim)
        self.discriminator = Discriminator(in_dim, out_dim)
        self.prelu = nn.PReLU()
        self.layernorm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, h_in, edge_index, edge_weight=None):
        h_out = self.conv(h_in, edge_index, edge_weight)
        h_out = self.prelu(h_out)
        h_out = self.layernorm(h_out)
        h_out = self.dropout(h_out)
        return h_out

class GNNTwoLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout = 0.5):
        super(GNNTwoLayer, self).__init__()
        self.conv1 = GCNConv(in_dim, out_dim)
        self.conv2 = GCNConv(out_dim, out_dim)
        self.prelu = nn.PReLU()
        self.layernorm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        h1 = self.conv1(x, edge_index, edge_weight)
        h1 = self.prelu(h1)
        h1 = self.layernorm(h1)
        h1 = self.dropout(h1)
        h2 = self.conv2(h1, edge_index, edge_weight)
        h2 = self.prelu(h2)
        h2 = self.layernorm(h2)
        h2 = self.dropout(h2)
        return h2


class GMIModelSingleLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout = 0.5):
        super(GMIModelSingleLayer, self).__init__()
        self.conv = GCNConv(in_dim, out_dim)
        self.discriminator = Discriminator(in_dim, out_dim)
        self.prelu = nn.PReLU()
        self.layernorm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, h_in, edge_index, edge_weight=None):
        h_out = self.conv(h_in, edge_index, edge_weight)
        h_out = self.prelu(h_out)
        h_out = self.layernorm(h_out)
        h_out = self.dropout(h_out)
        return h_out

    def get_MI(self, h_in, edge_index, edge_weight=None):
        h_out = self.conv(h_in, edge_index, edge_weight)
        h_out = self.prelu(h_out)
        h_out = self.layernorm(h_out)
        h_out = self.dropout(h_out)
        mi_edge, mi_node = self.discriminator(h_in, h_out, edge_index)
        mi_detached_edge, mi_detached_node = self.discriminator(h_in.detach(), h_out.detach(), edge_index.detach())
        return mi_node, mi_detached_node

# 带互信息估计的GCN模型,forward获取h2, get_MI获取互信息。
class GMIModel(nn.Module):
    def __init__(self, in_dim, out_dim, dropout = 0.5):
        super(GMIModel, self).__init__()
        self.conv1 = GCNConv(in_dim, out_dim)
        self.conv2 = GCNConv(out_dim, out_dim)

        self.discriminator = Discriminator(in_dim, out_dim)  # I(h_cur, h_prev)
        # self.discriminator = NodeLevelDiscriminator(in_dim, out_dim)
        self.prelu = nn.PReLU()

        self.layernorm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        h1 = self.conv1(x, edge_index, edge_weight)
        h1 = self.prelu(h1)
        h1 = self.layernorm(h1)
        h1 = self.dropout(h1)
        h2 = self.conv2(h1, edge_index, edge_weight)
        h2 = self.prelu(h2)
        h2 = self.layernorm(h2)
        h2 = self.dropout(h2)
        return h2

    def get_MI(self, x, edge_index, edge_weight=None):
        h1 = self.conv1(x, edge_index, edge_weight)
        h1 = self.prelu(h1)
        h1 = self.layernorm(h1)
        h1 = self.dropout(h1)
        h2 = self.conv2(h1, edge_index, edge_weight)
        h2 = self.prelu(h2)
        h2 = self.layernorm(h2)
        h2 = self.dropout(h2)
        # disc
        mi_edge, mi_node = self.discriminator(x, h2, edge_index) # [num_nodes, degree(node)]的list
        mi_detached_edge, mi_detached_node = self.discriminator(x.detach(), h2.detach(), edge_index.detach())
        return mi_node, mi_detached_node  # mi用于计算div_loss，训练encoder；mi_withno_grad2encoder用于计算MI损失，只更新判别器。

# 按边划分正样本
class Discriminator(nn.Module):
    def __init__(self, n_in, n_h, n_hidden=64, proj_dim=32):
        super().__init__()
        # self.f_k = nn.Bilinear(n_in, n_h, 1, bias=True) # 随后查看是否换成concat
        # self.act = nn.ReLU()
        self.num_neg = 32
        self.proj_prev = nn.Linear(n_in, proj_dim)
        # self.proj_cur = nn.Linear(n_h, proj_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * proj_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )

    # ---------- full MI loss ---------- #
    def forward(self, h_prev, h_cur, edge_index):
        # 计算正负分数
        pos, neg = self.edge_scores(h_prev, h_cur, edge_index)
        
        # 通过正负分数计算MI [E,], difference between positive and negative
        # dv = self.dv_per_edge(pos, neg)
        mi_edge = self.jsd_per_edge(pos, neg) # [E]
        #  mi_edge = self.infonce_per_edge(pos, neg)

        # 把edge-wise互信息转换为node-wise [num_nodes, degree(node)]的list
        num_nodes = h_prev.size(0)
        mi_per_node = edge_mi_to_node_mi(mi_edge, edge_index, num_nodes, use_src=True)
        return mi_edge, mi_per_node # 互信息, 长度为num_nodes的list。
    
        return -dv, dv.detach() # [E], loss和MI
        # 在这里做Gating, 取出专家选中的topk node算


    def edge_scores(self, h_prev, h_cur, edge_index):
        """
        Args
        ----
        h_prev     : (N, n_in)   前一层节点表示
        h_cur      : (N, n_h)    当前层 / 全局表示
        edge_index : (2, E)      边(src, dst) 索引
        num_neg    : int         每条正边配多少负例
        Returns
        -------
        pos_scores : (E,)            每条真实边分数
        neg_scores : (E, num_neg)    打乱 dst 的负分数
        """
        src, dst = edge_index                     # 长度 = E

        # ---------- 正样本分数 ---------- #

        # p_prev = self.proj_prev(h_prev)  # (N, proj_dim)
        # p_cur  = self.proj_cur(h_cur)

        if h_prev.shape[1] == h_cur.shape[1]:
            p_prev = h_prev  # 不变换
        else:
            # if not hasattr(self, 'pca_mapper'):
            #     from sklearn.decomposition import PCA
            #     pca = PCA(n_components=h_cur.shape[1])
            #     self.pca_mapper = pca.fit(h_prev.detach().cpu().numpy())
            
            # h_prev_pca = self.pca_mapper.transform(h_prev.detach().cpu().numpy())
            # p_prev = torch.tensor(h_prev_pca, dtype=h_prev.dtype, device=h_prev.device)
            p_prev = self.proj_prev(h_prev)
        p_cur = h_cur
        

        pos_samples = torch.cat([p_prev[src], p_cur[dst]], dim=-1)  # (E, 2*proj_dim),1个正样本对
        pos_scores = self.mlp(pos_samples).squeeze(-1)  # (E,)

        # ---------- 负样本分数 ---------- #
        num_nodes = h_prev.size(0)
        E = edge_index.size(1)
        neg_src = torch.randint(0, num_nodes, (self.num_neg, E), device=p_prev.device) 
        neg_samples = torch.cat([p_prev[neg_src], p_cur[dst].unsqueeze(0).expand(self.num_neg, -1, -1)], dim=2) # (num_neg, E, 2*proj_dim)
        neg_scores = self.mlp(neg_samples).squeeze(-1)    # (num_neg, E)
        neg_scores = neg_scores.reshape(self.num_neg, -1).transpose(0, 1)  # (E, num_neg)

        # neg_dst = self._shuffle_dst(dst)        # (num_neg, E)
        # src_rep = p_prev[src].unsqueeze(0).expand(self.num_neg, -1, -1) # （num_neg, E, n_proj）
        # x_neg = torch.cat([src_rep, p_cur[neg_dst]], dim=-1)                # (num_neg, E, n_in + n_h)
        # x_neg = x_neg.reshape(-1, x_neg.size(-1))                  # (num_neg * E, input_dim)
        # neg = self.mlp(x_neg).squeeze(-1)                          # (num_neg * E,)
        # neg_scores = neg.reshape(self.num_neg, -1).transpose(0, 1)  # (E, num_neg)
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
        # p_samples: Tensor of shape (E,) - 正样本得分
        # n_samples: Tensor of shape (E, K) - 每个正样本对应的K个负样本得分
        # 输出: shape (E,) 每条边的 InfoNCE loss

        # 把正样本扩展成 shape (E, 1) 然后拼接成 shape (E, K+1)
        all_scores = torch.cat([p_samples.unsqueeze(1), n_samples], dim=1)  # (E, K+1)
        
        # 计算 log-softmax
        log_probs = F.log_softmax(all_scores, dim=1)  # (E, K+1)

        # 正样本在第0个位置，所以我们取第0个log概率
        return log_probs[:, 0]#.mean()  # shape: (E,)

        all_scores = torch.cat([p_samples.unsqueeze(1), n_samples], dim=1)  # (E, K+1)

        # log-softmax over K+1 choices
        log_probs = F.log_softmax(all_scores, dim=1)  # (E, K+1)

        # 取第一个位置（正样本）的 log prob
        mi_per_sample = log_probs[:, 0]  # (E,)

        # 返回均值作为互信息估计值
        return mi_per_sample.mean()




def edge_mi_to_node_mi(dv: torch.Tensor,
                       edge_index: torch.LongTensor,
                       num_nodes: int,
                       use_src: bool = True) -> List[torch.Tensor]:
    """
    将 edge-wise MI 转换为 node-wise MI（按邻居分组），返回 list of Tensor，每个 Tensor shape=(deg(v),)
    Args:
        dv: (E,) 每条边上的 MI 值
        edge_index: (2, E)
        num_nodes: 图中节点数
        use_src: 如果 True，则按 src 节点聚合；否则按 dst 聚合

    Returns:
        List[Tensor]，长度=num_nodes，每个元素是该节点参与边的 MI 值，shape=(deg(v),)
    """
    src, dst = edge_index
    idx = src if use_src else dst

    # 初始化每个节点的 edge-MI 列表
    mi_per_node_list = [[] for _ in range(num_nodes)]
    # 收集每条边的 MI 到对应节点
    for edge_id, node_id in enumerate(idx):
        mi_per_node_list[node_id.item()].append(dv[edge_id])  # 仍保留梯度

    # 将每个节点对应的 MI list 转换成 Tensor（保留梯度）
    mi_per_node_tensor = [
        torch.stack(mi_list) if len(mi_list) > 0 else torch.tensor([], device=dv.device, dtype=dv.dtype)
        for mi_list in mi_per_node_list
    ]

    return mi_per_node_tensor


