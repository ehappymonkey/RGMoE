### 用单个路由+2层GNN的GMoE, 使用DV-MI最大化neighbors和h2的互信息，设置diversity约束不同专家的MI list。配套GMI_v3。
### 可以实现有效的diversity。
### 在v6基础上MoeLayer上使用一个Discriminator。
import torch, gc
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import torch_geometric.transforms as T
from torch.distributions.normal import Normal
import utils
from copy import deepcopy
from models.GMI_v3 import GMIModel, GNNTwoLayer, Discriminator
import numpy as np 
import logging
import math
from attack_utils import calculate_asr_fasr_acc


class RandomRouter(nn.Module):
    """
    在给定的 num_nodes 个节点中，为每个节点随机选出 top_k 个专家，
    并且生成相应的稀疏门控分布 (sparse_gates) 以及完整门控分布 (full_gates)。
    """
    def __init__(self, num_experts, top_k):
        super(RandomRouter, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
    
    def forward(self, num_nodes):
        """
        生成随机的 sparse_gates, topk_indices, full_gates.
        
        :param num_nodes: 节点数量
        :return:
            sparse_gates: [num_nodes, num_experts], 只有随机选出的 top_k 位置非零
            topk_indices: [num_nodes, top_k], 表示每个节点选中的专家
            full_gates:   [num_nodes, num_experts], 一个完整的随机 softmax 分布
        """

        # 1) 生成一个随机分布, 对 [num_nodes, num_experts] 的随机数做 softmax, 得到 full_gates
        random_raw = torch.rand(num_nodes, self.num_experts)
        full_gates = F.softmax(random_raw, dim=1)
        
        # 2) 每个节点随机打乱 experts 的顺序，然后取前 top_k 作为选中专家
        #    random_indices 的形状 [num_nodes, num_experts]
        #    每一行是 experts 的随机排列
        random_indices = torch.rand(num_nodes, self.num_experts).argsort(dim=1)
        topk_indices = random_indices[:, :self.top_k]  # [num_nodes, top_k]
        
        # 3) 对选中的 top_k 专家做 softmax，生成随机门控值
        #    先生成 [num_nodes, top_k] 的随机数，然后对每行做 softmax
        topk_gates_raw = torch.rand(num_nodes, self.top_k)
        topk_gates = F.softmax(topk_gates_raw, dim=1)  # [num_nodes, top_k]
        
        # 4) 构造 sparse_gates, 与 full_gates 形状相同
        sparse_gates = torch.zeros_like(full_gates)  # [num_nodes, num_experts]
        # 在 topk_indices 上 scatter topk_gates
        sparse_gates.scatter_(dim=1, index=topk_indices, src=topk_gates)
        
        return sparse_gates, topk_indices, full_gates

class Cosine_Router(nn.Module):
    def __init__(self, in_dim, num_experts, top_k, d_e=None, gating_type='softmax',
                 init_expert_norm=0.1, init_tau=1.0, load_balance_tau0=1.0, load_balance_weight=0.1, expert_init="orthogonal"):
        """
        Args:
            in_dim (int): 输入隐藏状态的维度 d.
            num_experts (int): 专家数量 N.
            select_topk (int): 最终选中的专家数 k.
            d_e (int, optional): 投影后的维度。如果未指定，通常设为 N/2.
            gating_type (str): 门控函数类型，支持 'softmax' 或 'sigmoid'。默认 'softmax'.
            init_expert_norm (float): 专家嵌入的初始 L2 范数，建议设置为 0.1.
            init_tau (float): 可学习温度参数 τ 的初始值.
            
            load_balance_tau0 (float): 固定温度 τ₀，用于计算负载平衡时的 softmax.
            load_balance_weight (float): 负载平衡损失的权重.
        """
        super(Cosine_Router, self).__init__()
        if d_e is None:
            d_e = num_experts  # 通常设为专家数量的一半
        self.in_dim = in_dim
        self.num_experts = num_experts
        self.d_e = d_e
        self.gating_type = gating_type
        self.select_topk = top_k

        # 线性投影矩阵 W ∈ R^(d_e × d)
        self.W = nn.Parameter(torch.Tensor(d_e, in_dim))
        nn.init.xavier_uniform_(self.W)

        # 专家嵌入矩阵，形状为 [num_experts, d_e]
        self.expert_embeddings = nn.Parameter(torch.Tensor(num_experts, d_e))
        if expert_init == "orthogonal":
            orthonormal_init = self.generate_orthonormal_vectors_svd()
            with torch.no_grad():
                self.expert_embeddings.copy_(
                    orthonormal_init * init_expert_norm)
        else:  
            # 先均匀初始化，再归一化到固定范数（init_expert_norm）
            nn.init.uniform_(self.expert_embeddings, a=-0.1, b=0.1)
            with torch.no_grad():
                self.expert_embeddings.copy_(
                    F.normalize(self.expert_embeddings, p=2, dim=1) * init_expert_norm)


        # 可学习温度参数 τ，用于 gating（与负载平衡中的 τ₀ 区别）
        self.tau = nn.Parameter(torch.tensor(init_tau))

        # 固定温度参数和负载平衡损失权重
        self.load_balance_tau0 = load_balance_tau0
        self.load_balance_weight = load_balance_weight

    def forward(self, h, is_training=True):
        """
        Args:
            h (Tensor): 输入隐藏状态，形状为 [num, in_dim].
        
        Returns:
            selected_gates (Tensor): 经过 top-k 选择后扩展到完整专家维度的 gating 权重，
                                       形状 [num, num_experts]，非 top-k 位置为 0.
            topk_indices (Tensor): 每个 token 选中的 top-k 专家索引，形状 [num, select_topk].
            scores (Tensor): 原始 cosine 路由分数，形状 [num, num_experts].
            h_proj (Tensor): 投影后的 token 表示，形状 [num, d_e].
        """
        # 1. 投影并归一化：f_proj(h) = W h, h_proj 的形状为 [num, d_e]
        h_proj = torch.matmul(h, self.W.t())
        h_norm = F.normalize(h_proj, p=2, dim=1)  # [num, d_e]

        # 2. 专家嵌入归一化（保证专家的 L2 范数固定）
        expert_embed = F.normalize(self.expert_embeddings, p=2, dim=1)  # [num_experts, d_e]

        # 3. 计算 cosine 路由分数: scores = (W h) · e_i，由于均归一化，结果即为余弦相似度
        scores = torch.matmul(h_norm, expert_embed.t())  # [num, num_experts]

        # 4. 使用可学习温度 τ 对 scores 进行缩放，再计算 gating 值
        if self.gating_type == 'softmax':
            full_gates = F.softmax(scores / self.tau, dim=1)
        elif self.gating_type == 'sigmoid':
            full_gates = torch.sigmoid(scores / self.tau)
        
        topk_values, topk_indices = torch.topk(full_gates, self.select_topk, dim=1)
        # # 5. Top-k 选择：选出每个 token 对应的 top-k gating 值及其索引
        # if self.training:
        #     # 训练时：选择 top-k 最大的值（原始逻辑）
        #     topk_values, topk_indices = torch.topk(full_gates, self.select_topk, dim=1, largest=True)
        # else:
        #     # 测试时：选择 top-k 最小的值
        #     topk_values, topk_indices = torch.topk(full_gates, self.select_topk, dim=1, largest=False)
        
        # 对选中的 top-k gating 值归一化（使其在每个 token 内和为 1）, 或者softmax归一化
        normalized_topk = F.softmax(topk_values, dim=1)
        # 6. 将归一化的 top-k gating 值 scatter 回完整的专家维度，其余位置置 0
        sparse_gates = torch.zeros_like(full_gates)
        sparse_gates.scatter_(1, topk_indices, normalized_topk)
        
        return sparse_gates, topk_indices, full_gates

    def cal_load_loss(self, full_gates, topk_indices):
        """
        根据论文中描述计算负载平衡损失。
        
        Args:
            scores (Tensor): 原始的 cosine 路由分数，形状 [B, num_experts].
            topk_indices (Tensor): 每个 token 选中的 top-k 专家索引，形状 [B, select_topk].
            
        Returns:
            loss (Tensor): 标量，负载平衡损失.
        """
        B = full_gates.size(0)
        N = self.num_experts

        # 固定温度 τ₀ 下计算 softmax 得到平滑的专家分布概率 p, 形状 [B, N]
        p = F.softmax(full_gates / self.load_balance_tau0, dim=1)
        
        # 计算离散的 token 分配计数 t_i：
        # 对于每个 token，topk_indices 给出其选中的专家，将其转换为 one-hot 编码，形状 [B, select_topk, N]
        one_hot = F.one_hot(topk_indices, num_classes=N).float()  # [B, select_topk, N]
        # 对于每个 token，在 select_topk 维度上取最大（实际上每个位置都是 0/1），然后对 token 求和，得到每个专家的总计数 t_i，形状 [N]
        t = one_hot.sum(dim=1).sum(dim=0)  # [N] 计算每个专家被选中的 token 绝对数量
        t = t / (B * self.select_topk)  # 归一化，得到相对频率
        
        # 同时计算每个专家的平滑负载 L_i = sum_{x in B} p_i(x)
        L = p.sum(dim=0)  # [N]
        
        # 根据公式，负载平衡损失为： L_balance = (N/|B|) * sum_{i=1}^{N} t_i * L_i
        loss = (N / B) * torch.sum(t * L)
        # 最后乘以权重系数
        loss = self.load_balance_weight * loss
        return loss
    
    def calculate_balancing_loss(self, h, idx):
        _, topk_indices, full_gates = self.forward(h)

        # if idx != None: 只在训练idx上计算负载损失。
        #     full_gates = full_gates[idx]
        #     topk_indices = topk_indices[idx]

        load_loss = self.cal_load_loss(full_gates, topk_indices)
        return load_loss

    def generate_orthonormal_vectors(self):
        """
        生成 shape=[num_experts, d_e] 的行向量，彼此正交，范数=1
        (如果 num_experts > d_e，则无法做到完全正交，但可以近似。)
        """
        num_experts = self.num_experts
        d_e = self.d_e

        # 1) 随机初始化一个矩阵 M, shape=[num_experts, d_e]
        M = torch.randn(num_experts, d_e)
        
        # 2) Gram-Schmidt 正交化
        # 对每一行依次做投影消除，保证 rows 近似正交
        for i in range(num_experts):
            # 取第 i 行
            vec = M[i]
            # 与之前 0..(i-1) 行做正交
            for j in range(i):
                prev_vec = M[j]
                # 减去在 prev_vec 方向上的投影
                vec = vec - (torch.dot(vec, prev_vec) / torch.dot(prev_vec, prev_vec)) * prev_vec
            # 归一化
            M[i] = F.normalize(vec, p=2, dim=0)
        return M  # shape=[num_experts, d_e], 行与行近似正交，范数=1

    def generate_orthonormal_vectors_svd(self):
        """
        用 SVD 得到正交基，然后取其中的 num_experts 行向量。
        若 num_experts <= d_e，则可以取前 num_experts 个奇异向量
        """
        num_experts = self.num_experts
        d_e = self.d_e

        # 随机矩阵 shape=[d_e, d_e]
        # (也可以是 [num_experts, d_e], 根据具体需求做 SVD)
        M = torch.randn(d_e, d_e)
        U, S, V = torch.svd(M)  # torch.svd(M) => U, S, V
        
        # U, V 都是正交矩阵, shape = [d_e, d_e]
        # 这里可以从 U 中取前 num_experts 行，或者从 V 中取前 num_experts 行
        # 例如:
        orthonormal_rows = U[:num_experts, :]  # shape = [num_experts, d_e]
        # 每行范数=1，彼此正交
        
        return orthonormal_rows

class NoisyTopKGate(nn.Module):
    def __init__(self, in_dim, num_experts, top_k, weight_importance, weight_load, noise_std=1.0):
        """
        :param in_dim: 输入特征维度
        :param num_experts: 专家数量
        :param top_k: 每个样本激活的专家数
        :param weight_importance: 重要性损失的权重
        :param weight_load: 负载均衡损失的权重
        :param noise_std: 噪声标准差
        """
        super(NoisyTopKGate, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std

        # 路由器参数：用于计算路由得分
        self.W_g = nn.Parameter(torch.Tensor(in_dim, num_experts))
        self.W_n = nn.Parameter(torch.Tensor(in_dim, num_experts))
        nn.init.xavier_uniform_(self.W_g)
        nn.init.xavier_uniform_(self.W_n)

        self.weight_importance = weight_importance
        self.weight_load = weight_load

        # 初始化标准正态分布，用于计算负载损失
        self.normal = Normal(0, 1)

    def forward(self, h, is_training=True):
        """
        计算 Q(h) = hW_g + ε * SoftPlus(hW_n)
        同时计算完整的门控分布和 top-k 后的稀疏分布。
        
        :param h: 节点特征，形状 [N, in_dim]
        :return: 
            sparse_gates: 稀疏门控分布，只有 top-k 位置非零，形状 [N, num_experts]
            topk_indices: 对应的 top-k 专家索引，形状 [N, top_k]
            full_gates: 完整的 softmax 分布，形状 [N, num_experts]
        """
        # 计算原始得分
        # Q = torch.matmul(h, self.W_g) + self.noise_std * F.softplus(torch.matmul(h, self.W_n))
        # 以下是Graph-MoE论文中实现
        clean_logits = torch.matmul(h, self.W_g)
        noise_std = F.softplus(torch.matmul(h, self.W_n))+1e-2
        Q = clean_logits + (torch.randn_like(clean_logits) * noise_std)

        # 计算完整的门控分布
        full_gates = F.softmax(Q, dim=1)
        # 选取 top-k 得分
        # if is_training:
        topk_values, topk_indices = torch.topk(full_gates, self.top_k, dim=1)
        # else:
        #     topk_values, topk_indices = torch.topk(full_gates, self.top_k, dim=1, largest=False)
        topk_gates = F.softmax(topk_values, dim=1)
        # 构造稀疏门控分布：只有 top-k 位置非零，其余置0
        # sparse_gates = torch.zeros_like(Q)
        sparse_gates = torch.zeros_like(full_gates)
        sparse_gates.scatter_(1, topk_indices, topk_gates)
        return sparse_gates, topk_indices, full_gates

    def cal_importantloss(self, full_gates):
        """
        计算重要性损失：基于完整的门控分布衡量所有专家的激活均衡性。
        :param full_gates: 完整的门控分布，形状 [N, num_experts]
        :return: importance loss（标量）
        """
        # 对于每个专家计算 importance score：所有节点上的 full_gates 之和
        importance = full_gates.sum(dim=0)  # [num_experts]
        mean = importance.mean()
        std = importance.std()
        loss = self.weight_importance * (std / (mean + 1e-8))**2
        return loss

    def cal_load_loss(self, h):
        """
        计算负载损失：利用输入 h 计算每个专家的平滑负载，并计算 CV² 损失。
        :param h: 输入特征，形状 [N, in_dim]
        :return: load loss（标量）
        """
        # 计算 H = hW_g
        H = torch.matmul(h, self.W_g)  # [N, num_experts]
        # 计算 sigma = SoftPlus(hW_n)
        sigma = F.softplus(torch.matmul(h, self.W_n))  # [N, num_experts]
        N, E = H.shape
        T_val = torch.zeros_like(H)
        for i in range(E):
            H_excl = H.clone()
            H_excl[:, i] = float('-inf')
            kth_values, _ = torch.topk(H_excl, self.top_k, dim=1)
            T_val[:, i] = kth_values[:, -1]
        # 计算 z = (H - T_val) / sigma
        z = (H - T_val) / (sigma + 1e-8)
        # 计算概率 P = Phi(z)
        P = self.normal.cdf(z)
        # 每个专家的负载为所有样本上 P 的和
        load = P.sum(dim=0)  # [E]
        mean_load = load.mean()
        std_load = load.std()
        cv2 = (std_load / (mean_load + 1e-8))**2
        loss = self.weight_load * cv2
        return loss


    def calculate_balancing_loss(self, h, idx=None):
        """
        计算综合平衡损失：使用完整门控分布计算重要性损失，加上负载损失。
        :param h: 输入特征，形状 [N, in_dim]
        :return: 综合平衡损失（标量）
        """
        _, _, full_gates = self.forward(h)
        imp_loss = self.cal_importantloss(full_gates)
        l_loss = self.cal_load_loss(h)
        total_loss = imp_loss + l_loss
        return total_loss


# 输出embed和可以计算MI的loss
class GMIExpert(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.5, conv_type='GCN'):
        super(GMIExpert, self).__init__()
        self.model = GNNTwoLayer(in_dim, out_dim) # 以后在往GMI里加其余GNN卷积

    def forward(self, x, edge_index, edge_weight=None):
        h2 = self.model(x, edge_index, edge_weight)
        return h2

# MoE层，采用多个基于GCNConv的专家
class MoELayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, conv_type, num_experts, top_k, router, weight_importance, weight_load, noise_std=0.1):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std

        if router == 'noisytopk':
            self.gate = NoisyTopKGate(in_dim, num_experts, top_k, weight_importance, weight_load, noise_std)
        elif router == 'cosine':
            self.gate = Cosine_Router(in_dim, num_experts, top_k)
        else:
            raise ValueError("Invalid router type" +  router)
  
        self.experts = nn.ModuleList([GMIExpert(in_dim, out_dim, dropout, conv_type) for _ in range(num_experts)])
        self.discriminator = Discriminator(in_dim, out_dim)

    def forward(self, h, edge_index, edge_weight = None, is_training=True, idx=None):
        # 保存传入门控模块的输入 h
        h_in = h
        gates, topk_indices, _ = self.gate(h, is_training)
    
        h2 = [expert(h, edge_index, edge_weight) for expert in self.experts] # num_experts个[num_nodes, degree]的list

        # Fuse h2
        h2_experts = torch.stack(h2, dim=1)
        h_moe = torch.sum(gates.unsqueeze(-1) * h2_experts, dim=1)


        # balance loss
        L_balance = self.gate.calculate_balancing_loss(h_in, idx=idx)

        # calculate MI 
        mi = [self.discriminator(h, h2[i], edge_index) for i in range(self.num_experts)]
        _, mi = zip(*mi)
        mi_detached = [self.discriminator(h.detach(), h2[i].detach(), edge_index) for i in range(self.num_experts)]
        _, mi_detached = zip(*mi_detached)
        # mi loss
        mi_experts = mi_detached  # MI只优化判别器
        # mi_experts = mi  # 也使用MI优化GNN
        mi_loss_per_expert = 0
        for expert_idx, mi_expert_nodes in enumerate(mi_experts):  # expert_idx in 0 ~ num_experts-1
            if False: # 考虑路由
                gate_weights = gates[:, expert_idx]  # [num_nodes]
                mask = gate_weights > 0  # bool tensor
                if mask.sum() == 0:
                    continue  # 没有节点被分配给这个 expert
                selected_indices = torch.where(mask)[0]  # 取出被分配节点的索引
                gate_selected = gate_weights[selected_indices]  # [num_selected_nodes]

                # 从 list 中取出对应的节点互信息向量（每个是 [deg(v)]）
                mi_selected_nodes = [mi_expert_nodes[i] for i in selected_indices]
                # 每个节点MI的均值就是这个专家的MI
                mi_expert = torch.stack([mi.mean() if mi.numel() > 0 else torch.tensor(0., device=mi.device, dtype=mi.dtype) for mi in mi_selected_nodes]) # [num_nodes]
                # 可选加权（根据 gate 权重）
                weighted_mi = mi_expert * gate_selected  # [num_selected_nodes]
                loss = -weighted_mi.mean()
                mi_loss_per_expert += loss
            else: # 不考虑路由
                mi_expert = torch.stack([mi.mean() if mi.numel() > 0 else torch.tensor(0., device=mi.device, dtype=mi.dtype) for mi in mi_expert_nodes]) # [num_nodes]  
                # mi_expert = mi_expert[idx] # 只算训练节点的MI loss
                loss = -mi_expert.mean()
                mi_loss_per_expert += loss
        # L_mi = torch.stack(mi_loss_per_expert).mean()
        L_mi = mi_loss_per_expert*1/self.num_experts

        # diversity loss
        L_div = compute_diversity_loss(mi, topk_indices, idx=idx) # 用mi计算多样损失，可优化GNN。
        # L_div = 0

        return h_moe, gates, topk_indices, h_in, L_balance, L_mi, L_div, mi_experts

# GraphMoE整体架构：初始 GCNConv + 多个 MoE 层 + 分类器
class GraphMoE(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout, conv_type, num_experts, top_k, router, num_moe_layers=1, noise_std=0.1, w_importance = 0.1, w_load = 0.1, w_div = 1, w_MI = 0):
        super(GraphMoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.w_div = w_div
        self.w_MI = w_MI
        self.is_training=True

        self.moe_layer = MoELayer(in_dim, hidden_dim, dropout, conv_type, num_experts, top_k, router, w_importance, w_load, noise_std)
        self.classifier = nn.Linear(hidden_dim, num_classes)


    def forward(self, h, edge_index, edge_weight=None, idx=None):
        routing_info = []  # 每层保存 (gates, topk_indices, h_in, gate_module)

        # 直接用初始节点输入router+2层GNN
        h_moe, gates, topk_indices, h_in, L_balance, L_mi, L_div, mi_experts  = self.moe_layer(h, edge_index, edge_weight, is_training=self.is_training, idx=idx)
        routing_info.append((gates, topk_indices, h_in, L_balance, L_mi, L_div, mi_experts))
        logits = self.classifier(h_moe)
        logits = F.log_softmax(logits, dim=1) # 使用NLL loss要经过softmax
        return logits, routing_info


    # 训练函数，之后可以移入model里
    # def fit(self, poison_x, poison_edge_index, poison_edge_weights, poison_labels, idx_train, idx_val, train_iters=100, verbose=False, SparseLoss = False):
 
    def fit(self, poison_x, poison_edge_index, poison_edge_weights, poison_labels, induct_x, induct_edge_index,induct_edge_weights, idx_train, idx_val, idx_atk, idx_clean_test, data, train_iters=100, verbose=False, margin=0.5, args=None):
        self.train()
        # 获取所有参数
        all_params = list(self.parameters())
        # 获取不包含 discriminator 的参数
        no_disc_params = []
        for name, param in self.named_parameters():
            # print(name)
            if "discriminator" not in name:
                no_disc_params.append(param)

        
        all_param_names = [name for name, _ in self.named_parameters()]
        # 构造 no_disc_params 的 ID 集合
        no_disc_param_ids = set(id(p) for p in no_disc_params)
        # 用 ID 判断 param 是否属于 no_disc_params
        no_disc_param_names = [name for name, param in self.named_parameters() if id(param) in no_disc_param_ids]
        # print("All param names:")
        # print(all_param_names)
        # print("\nParam names without discriminator:")
        # print(no_disc_param_names)

        
        # 初始化两个优化器
        optimizer_all = torch.optim.AdamW(all_params, lr=1e-3, weight_decay=5e-4)
        optimizer_no_disc = torch.optim.AdamW(no_disc_params, lr=1e-3, weight_decay=5e-4)
        # MI 不变，造成优化时Div也不变。
     
        best_acc_val = 0
        best_epoch = -1
        best_l_div = math.inf
        expert_select_count = torch.zeros(self.num_experts, dtype=torch.long).to(poison_x.device)
        for epoch in range(train_iters):
            s1_steps = 1
            s2_steps = 3 # s1=1，s2=3, 100epochs后同时优化不会导致MI loss变正。
            # if epoch < 100: # 先训练好判别器，再优化距离
            for _ in range(s1_steps): # L_pred, balance训练GMoE, L_mi训练判别器。在diversify阶段保持训练判别器也是必要的，需根据特征变化保证判别器学习到比较准确的MI。
                # for expert in self.moe_layer.experts: # 然而， L_mi和L_div会对抗优化，依赖调参。理想优化情况：在diversify阶段MI loss保持稳定，L_div逐渐下降。
                for p in self.moe_layer.discriminator.parameters():
                    p.requires_grad = True
                optimizer_all.zero_grad()
                # forward得到losses
                output, routing_info = self.forward(poison_x, poison_edge_index, poison_edge_weights, idx_train)
                # 计算主任务的交叉熵损失
                L_pred = F.nll_loss(output[idx_train], poison_labels[idx_train])
                # 计算所有 MoE 层的 importance loss，并累加
                L_balance, L_mi, L_div, L_sharp = 0, 0, 0, 0
                for (gates, topk_indices, h_in, l_balance, l_mi, l_div, _) in routing_info:
                    L_balance += l_balance
                    L_mi += l_mi
                    L_div += l_div
                    # L_sharp += l_sharp
                    flattened_indices = topk_indices.flatten()
                    expert_select_count.index_add_(
                        0,                                # 在第0维累加
                        flattened_indices,                # 被选中的专家index
                        torch.ones_like(flattened_indices, dtype=torch.long)
                    )
                    num_nodes = topk_indices.shape[0]
                L_balance = 1/len(routing_info)*L_balance
                L_mi = 1/len(routing_info)*L_mi
                L_s1 = L_pred + L_balance  +self.w_MI*L_mi  # 训练判别器
                # L_s1 = self.w_MI*L_mi
                L_s1.backward()
                optimizer_all.step()
            
            if epoch > 100: # 在前几个epochs先训练好判别器
                for _ in range(s2_steps):
                    # for expert in self.moe_layer.experts:
                    for p in self.moe_layer.discriminator.parameters():
                        p.requires_grad = False
                    optimizer_no_disc.zero_grad()
                    # forward得到losses
                    output, routing_info = self.forward(poison_x, poison_edge_index, poison_edge_weights, idx_train)
                    # 计算主任务的交叉熵损失
                    L_pred = F.nll_loss(output[idx_train], poison_labels[idx_train])
                    # 计算所有 MoE 层的 importance loss，并累加
                    L_balance, L_mi, L_div, L_sharp = 0, 0, 0, 0
                    for (gates, topk_indices, h_in, l_balance, l_mi, l_div, mi_experts) in routing_info:
                        L_balance += l_balance
                        L_mi += l_mi
                        L_div += l_div
                        # L_sharp = l_sharp
                        flattened_indices = topk_indices.flatten()
                        expert_select_count.index_add_(
                            0,                                # 在第0维累加
                            flattened_indices,                # 被选中的专家index
                            torch.ones_like(flattened_indices, dtype=torch.long)
                        )
                        num_nodes = topk_indices.shape[0]
                    L_balance = 1/len(routing_info)*L_balance
                    # L_mi = 1/len(routing_info)*L_mi
                    L_div = 1/len(routing_info)*L_div
                    L_sharp = 1/len(routing_info)*L_sharp
                    # L_s1 = L_pred + L_balance +self.w_MI*L_mi  # 训练判别器
                    # L_s2 = L_pred + L_balance + self.w_div*L_div # 训练Encoder

                    L_s2 = self.w_div*L_div # 训练Encoder
                    L_s2.backward()
                    optimizer_no_disc.step()


            # 验证
            self.eval()
            acc_val = utils.accuracy(output[idx_val], poison_labels[idx_val])
            # loss_div_val = compute_diversity_loss(mi_experts, t)
            if epoch % 1 == 0:
                logging.info(f'Epoch: {epoch:03d}, Classify Loss: {L_pred:.4f}, Balance Loss: {L_balance:.4f}, MI Loss: {self.w_MI*L_mi:.4f}, Diversity Loss: {self.w_div*L_div:.4f}, Val Acc: {acc_val:.4f}')

                # h2_experts
                asr_experts, ca_experts = [],[]
                # 再算一一遍h2_experts
                moe = self.moe_layer
                for expert in moe.experts:
                    h2 = expert(induct_x, induct_edge_index,induct_edge_weights)
                    logits = self.classifier(h2)
                    probs = F.log_softmax(logits, dim=1)
                    asr_i, fasr_i, ca_i = calculate_asr_fasr_acc(probs, idx_atk, idx_clean_test, data, args)
                    asr_experts.append(round(asr_i.item() * 100, 2))
                    ca_experts.append(round(ca_i.item() * 100, 2))
                logging.info('Every ASR: %s', asr_experts)
                logging.info('Everry CA: %s', ca_experts)
            # 统计每个专家在所有MoE层中的选择次数（统计 top-k 索引）
            with torch.no_grad():
                aggregated_counts = torch.zeros(self.num_experts, dtype=torch.int)
                for (gates, topk_indices, _, _, _, _, _) in routing_info:
                    for expert in range(self.num_experts):
                        aggregated_counts[expert] += (topk_indices == expert).sum().item()
                counts_str = ", ".join([f"Expert{idx}: {count}" for idx, count in enumerate(aggregated_counts.tolist())])
                print(f"Epoch {epoch:03d}: {counts_str}")
            
            # if acc_val > best_acc_val:
            #     best_acc_val = acc_val
            #     best_epoch = epoch
            #     weights = deepcopy(self.state_dict())
            if acc_val > 0.80 and self.w_div*L_div < best_l_div: # 保证acc下选div loss最小的。
                best_acc_val = acc_val
                best_l_div = self.w_div*L_div
                best_epoch = epoch
                weights = deepcopy(self.state_dict())

            gc.collect()
            torch.cuda.empty_cache() 
        
        self.load_state_dict(weights)
        print(f"Best Diversity Loss (acc>80): {best_l_div:.4f} at epoch {best_epoch} with Val ACC: {best_acc_val:.4f}")
        total_selections = train_iters * num_nodes * self.top_k
        expert_select_ratio = expert_select_count.float() / total_selections
        print("每个epoch平均专家被选中的次数: ", expert_select_count/train_iters)
        print("专家被选中的比例: ", expert_select_ratio)


    # 测试函数
    def test(self, features, edge_index, edge_weight, labels,idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        with torch.no_grad():
            output, _ = self.forward(features, edge_index, edge_weight)
            acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        # torch.cuda.empty_cache()
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
        return float(acc_test)

    # 前向传播每一个专家，观察每个专家的outputs
    def forward_every_expert(self, h, edge_index, edge_weight=None):
        # h = self.initial_conv(h, edge_index)
        # h = F.relu(h)
        # h, _, _, _, _ = self.initial_moe(h, edge_index, edge_weight)
        # h = F.relu(h) # 初始层也适用MoE
        all_expert_outputs, all_expert_outputs_soft, all_expert_MI= [],[],[]

        # 观察最后一个MoE层的专家的情况
        # for moe in self.moe_layers[:-1]:
        #     h, gates, topk_indices, h_in, _ = moe(h, edge_index, edge_weight)
        #     h = F.relu(h)
        
        # h, gates, topk_indices, h_in, balance_loss = self.moe_layer(h, edge_index, edge_weight)
        
        # 在最后一个 MoE 层, 遍历每个专家，并分类softmax
        # moe = self.moe_layers[-1]
        moe = self.moe_layer
        for expert in moe.experts:
            h2 = expert(h, edge_index, edge_weight)

            mi = [self.moe_layer.discriminator(h.detach(), h2.detach(), edge_index) for i in range(self.num_experts)]
        
            # expert_output = expert.model(h, edge_index, edge_weight)
            logits = self.classifier(h2)
            expert_output = F.log_softmax(logits, dim=1)
            all_expert_outputs.append(expert_output)
            all_expert_outputs_soft.append(logits)
            all_expert_MI.append(mi)
        return all_expert_outputs, all_expert_outputs_soft, all_expert_MI
    
    # 测试中毒节点（或所有节点）random routing的结果
    def forward_random_expert(self, h, edge_index, edge_weight=None):
        num_nodes = h.shape[0]
        random_router = RandomRouter(self.num_experts, self.top_k)
        gates, top_k_indices, _ = random_router(num_nodes)
        gates = gates.to(h.device)
        # 各个专家分别计算输出
        expert_outputs = [expert.model(h, edge_index, edge_weight) for expert in self.moe_layer.experts]
        # 堆叠专家输出，形状 [N, num_experts, out_dim]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        # 对专家输出加权求和
        h_moe = torch.sum(gates.unsqueeze(-1) * expert_outputs, dim=1)
        logits = self.classifier(h_moe)
        logits = F.log_softmax(logits, dim=1)
        return logits


def compute_diversity_loss(mi_per_node, topk_indices, margin=0.5, idx=None):
    """
    mi_per_node: List[List], len = num_experts, [num_nodes, deg]
    topk_indices: Tensor of shape [num_nodes, K]
    """

    # if idx != None:
    #     topk_indices = topk_indices[idx]
    #     mi_per_node = [
    #         [expert_mi[v] for v in idx]  # 从每个 expert 的 mi 列表中取出训练节点
    #         for expert_mi in mi_per_node
    #     ]

    
    num_nodes, K = topk_indices.size()
    loss_all = []

    for v in range(num_nodes):
        expert_ids = topk_indices[v]  # [K]
        mi_vecs = [mi_per_node[e][v] for e in expert_ids]  # list of K vectors, each [deg(v)]

        # 计算任意两两之间的相似度/距离
        pair_count= 0
        loss_expert_wise = 0
        for i in range(K):
            for j in range(i + 1, K):

                x_i = mi_vecs[i]
                x_j = mi_vecs[j]

                # 归一化，防K Expert数值差距导致sim小，造成用减小数值来降低cos sim。使用会导致MI较小<0.01。但效果diverse不错，
                x_i = F.normalize(x_i, p=2, dim=0)
                x_j = F.normalize(x_j, p=2, dim=0)

                # padding 对齐 shape
                # max_len = max(x_i.size(0), x_j.size(0))
                # x_i = F.pad(x_i, (0, max_len - x_i.size(0)))
                # x_j = F.pad(x_j, (0, max_len - x_j.size(0)))

                if len(x_i) == 0 or len(x_j) == 0:
                    continue 

                # 计算距离
                # if distance_fn == "cosine":
                #     sim = F.cosine_similarity(x_i.unsqueeze(0), x_j.unsqueeze(0)).squeeze()
                #     dist = 1 - sim  # 越远越好
                # else:
                #     raise ValueError("Unsupported distance")

                # Margin loss
                # l_ij = F.relu(margin - dist)

                sim = F.cosine_similarity(x_i.unsqueeze(0), x_j.unsqueeze(0)).squeeze()
                l_ij = F.relu(sim-margin)

                loss_expert_wise += l_ij
                pair_count += 1
        if pair_count > 0:
            loss_expert_wise /= (K * (K - 1) / 2) 
            loss_all.append(loss_expert_wise)
    # 聚合
    if len(loss_all) > 0:
        return torch.stack(loss_all).mean()
    else:
        return torch.tensor(0.0, requires_grad=True, device=topk_indices.device)
