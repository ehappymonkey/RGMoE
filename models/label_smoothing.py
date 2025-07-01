import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.GMoE_v6_1 import NoisyTopKGate

class PurifiedGMoE(nn.Module):
    def __init__(self, gmoe, device):
        super(PurifiedGMoE, self).__init__()
        self.gmoe = gmoe
        self.device = device
        self.num_experts = self.gmoe.moe_layer.num_experts
        self.top_k = self.gmoe.moe_layer.top_k
        self.purified_router = None

    # 建立新的路由器，使用soft labels（原始one-hot标签和smooth过的不确定标签）训练。
    def build_router(self, hidden_dim):
        self.purified_router = NoisyTopKGate(
            in_dim=hidden_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            weight_importance = 0.1, weight_load = 0.1
        ).to(self.device)

    def fit(self, h, edge_index, edge_weight, idx_train, soft_labels, epochs=50, lr=0.01):
        """
        使用 soft label (预测分布) 微调 purified router。

        参数:
            h: 输入特征 [N, D]
            edge_index, edge_weight: 图结构
            idx_train: 训练索引 (list[int] or tensor[int])
            soft_labels: [N, C] 的 soft label（由 get_smooth_soft_label 生成）
        """
        self.eval()
        self.gmoe.eval()

        with torch.no_grad():
            h_moe, _, _, _, _, _, _, _ = self.gmoe.moe_layer(h, edge_index, edge_weight)
        h_moe = h_moe.detach()

        if self.purified_router is None:
            self.build_router(h_moe.shape[1])

        optimizer = torch.optim.AdamW(self.purified_router.parameters(), lr=lr)

        idx_train_tensor = (
            torch.tensor(idx_train, device=self.device) if not torch.is_tensor(idx_train) else idx_train
        )

        for epoch in range(epochs):
            self.purified_router.train()
            optimizer.zero_grad()

            # 用 purified router 重新计算路由
            sparse_gates, _, _ = self.purified_router(h_moe)  # [N, E]

            # 组合专家输出
            expert_outputs = []
            for expert in self.gmoe.moe_layer.experts:
                out = expert(h, edge_index, edge_weight)
                out = out[0] if isinstance(out, tuple) else out
                expert_outputs.append(out)
            expert_outputs = torch.stack(expert_outputs, dim=1)  # [N, E, D]
            h_new = torch.sum(sparse_gates.unsqueeze(-1) * expert_outputs, dim=1)  # [N, D]

            # 最终分类输出
            logits = self.gmoe.classifier(h_new)  # [N, C]
            log_probs = F.log_softmax(logits, dim=1)

            # 只在训练集上用 soft label 做 KL loss
            loss = F.kl_div(
                log_probs[idx_train_tensor],
                soft_labels[idx_train_tensor],
                reduction="batchmean"
            )

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"[Epoch {epoch+1}] KLDiv Loss (router + pred): {loss.item():.4f}")

    def forward(self, h, edge_index, edge_weight):
        """
        使用 purified router 推理
        """
        h_moe, _, _, _, _, _, _, _ = self.gmoe.moe_layer(h, edge_index, edge_weight)
        sparse_gates, topk_indices, _ = self.purified_router(h_moe)

        expert_outputs = []
        for expert in self.gmoe.moe_layer.experts:
            out = expert(h, edge_index, edge_weight)
            out = out[0] if isinstance(out, tuple) else out
            expert_outputs.append(out)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [N, E, D]
        h_new = torch.sum(sparse_gates.unsqueeze(-1) * expert_outputs, dim=1)

        routing_info = []
        routing_info.append((sparse_gates, topk_indices))

        logits = self.gmoe.classifier(h_new)
        return logits, routing_info


def get_smooth_soft_label(y, uncertain_indices, num_classes, smoothing=0.9):
    """
    GPU 优化版本：
    - y: tensor[N] (GPU)
    - uncertain_indices: tensor[K] (GPU，存储不确定节点的索引)
    """
    soft_labels = torch.zeros((len(y), num_classes), 
                            dtype=torch.float32, 
                            device=y.device)  # 直接创建在 y 的设备上
    
    # 生成平滑标签（所有节点）    
    smooth_val = smoothing / (num_classes - 1)
    soft_labels.fill_(smooth_val)
    soft_labels[torch.arange(len(y)), y] = 1.0 - smoothing  # 对角线部分增强
    
    # 非不确定节点恢复为 one-hot
    uncertain_indices = torch.tensor([v.item() for v in uncertain_indices], device=y.device)  # 保持和data.y相同设备

    mask = torch.ones(len(y), dtype=torch.bool, device=y.device)
    mask[uncertain_indices] = False  # 标记不确定节点
    print(soft_labels[mask].shape)
    # soft_labels[mask] = torch.zeros(num_classes, device=y.device)
    # soft_labels[mask, y[mask]] = 1.0  # one-hot

    soft_labels[mask] = torch.zeros_like(soft_labels[mask], device=y.device)  # 全部置 0
    soft_labels[mask, y[mask]] = 1.0                          # 设置 one-hot

    
    return soft_labels  # 返回张量，而非字典

def analyze_expert_variance(bkd_tn_nodes, output, all_expert_outputs, num_bins=10):
    """
    分析节点集合中专家预测的方差分布
    
    参数:
        bkd_tn_nodes: 包含干净节点和trigger节点的索引列表
        all_expert_outputs: 各专家对节点的预测logits [num_experts, num_nodes, num_classes]
        num_bins: 方差分布的直方图分箱数
    """
    # 1. 计算每个节点的专家预测方差
    node_variances, node_kls = [],[]
    for node_idx in bkd_tn_nodes:
        logit = output[node_idx].cpu().detach()  # 原始 MoE 输出
        prob = torch.softmax(logit, dim=0)

        # 收集所有专家对该节点的预测logits [num_experts, num_classes]
        probs_experts = torch.stack([torch.softmax(all_expert_outputs[expert_idx][node_idx].cpu().detach(), dim=0)
                                   for expert_idx in range(len(all_expert_outputs))])
        
        kl_divs = [F.kl_div(prob.log(), p_i, reduction='batchmean') for p_i in probs_experts]
        sum_kl = sum(kl_divs) 
        node_kls.append(sum_kl.item())

        # # 计算每个类别的方差 [num_classes]
        # class_variances = torch.var(probs_experts, dim=0)
        # # 取均值作为该节点的总体方差
        # mean_variance = class_variances.mean().item()
        # node_variances.append(mean_variance)
    
    # 转换为numpy数组方便统计
    # variances = np.array(node_variances)
    node_kls = np.array(node_kls)
    variances = node_kls  # 使用KL散度作为方差度量
    
    # 2. 方差分布统计
    print("\n=== 专家预测方差分布统计 ===")
    print(f"分析节点数: {len(bkd_tn_nodes)} (干净:1354 + 中毒:80)")
    print(f"整体方差 - 均值: {variances.mean():.4f}, 标准差: {variances.std():.4f}")
    print(f"最小方差: {variances.min():.4f}, 最大方差: {variances.max():.4f}")
    
    # 方差分箱统计
    hist, bin_edges = np.histogram(variances, bins=num_bins)
    print("\n方差分布直方图:")
    for i in range(len(hist)):
        print(f"[{bin_edges[i]:.3f}, {bin_edges[i+1]:.3f}): {hist[i]}个节点")
    
    return variances