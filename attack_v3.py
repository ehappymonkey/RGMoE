
#!/usr/bin/env python
# coding: utf-8
# 相比于v2增加label smoothing净化路由。
# %% In[1]: 
import numpy as np
import torch
import json
from help_funcs import prune_unrelated_edge
import os, logging

# Training settings
from attack_utils import parse_args, set_seed, set_logging, count_experts_selection, calculate_asr_fasr_acc

args = parse_args()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))
set_logging()
# set_seed(args.seed)
logging.info(args)

def load_poisoned_files():
    save_dir = "poisoned_files"
    os.makedirs(save_dir, exist_ok=True)  
    save_filename = f"{args.dataset}_{args.vs_number}_{args.selection_method}_" \
                    f"{args.defense_mode}_{args.seed}_backdoor_attack.pth"
    save_path = os.path.join(save_dir, save_filename)
    print(f"Loading from: {save_path}")
    loaded_attack_files = torch.load(save_path, map_location=device)

    # 恢复各类数据
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

data, poison_x, poison_edge_index, poison_edge_weights, poison_labels, idx_attach, bkd_tn_nodes, idx_val, idx_clean_test, idx_atk, mask_edge_index, model = load_poisoned_files()

# 先用overall情况下数据注入trigger，然后再做prune
# inject trigger on attack test nodes (idx_atk)'''给idx_atk的节点都注入trigger
induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)]) 
induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights,device)
induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
# do pruning in test datas'''
if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
    induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device)

from models.construct import model_construct
# models = ['GCN','GAT', 'GraphSage']
# args.num_experts = 6 # 这里方便调整
# args.topk = 2
# args.router = 'noisytopk'
# args.router = 'cosine'
logging.info('Gating: %s', args.router)  
models = ['GMoE']
total_overall_asr = 0
total_overall_ca = 0
for test_model in models:
    args.test_model = test_model
    rs = np.random.RandomState(args.seed) # 生成一个固定状态
    seeds = rs.randint(1000,size=1) 
    logging.info("seeds: {}".format(seeds))

    overall_asr = 0
    overall_ca = 0
    for seed in seeds: 
        set_seed(seed)
        
        if args.test_model =='GMoE': # 先只看GMoE的输出
            # for args.w_div in [2]:
            # args.w_div = 1
            print('MI权重 w_div: ', args.w_div)
            logging.info("Testing Model: {}".format(args.test_model))
            test_model = model_construct(args,args.test_model,data,device).to(device) # model_construct中加入我们的GMoE
            test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, induct_x, induct_edge_index,induct_edge_weights, bkd_tn_nodes, idx_val, idx_atk, idx_clean_test, data, train_iters=args.epochs,verbose=False, margin=args.div_margin, args=args)
            if args.test_model =='GMoE':
                output, routing_info = test_model(poison_x,poison_edge_index, poison_edge_weights)

            else:
                output = test_model(poison_x,poison_edge_index,poison_edge_weights)

            if(args.evaluate_mode == 'overall'):
                if args.test_model =='GMoE': # 整个GMoE结果
                    test_model.is_training=True
                    output, routing_info = test_model(induct_x,induct_edge_index,induct_edge_weights)
                    asr, fasr, ca = calculate_asr_fasr_acc(output, idx_atk, idx_clean_test, data, args)
                    logging.info("(TopK) ASR: {:.4f}; Flip_ASR: {:.4f}; CA: {:.4f}".format(asr, fasr, ca))
                    all_expert_outputs, all_expert_outputs_soft, all_expert_MI = test_model.forward_every_expert(induct_x,induct_edge_index,induct_edge_weights)
                    asr_experts, ca_experts = [],[]
                    for i, output_expert in enumerate(all_expert_outputs):
                        asr_i, fasr_i, ca_i = calculate_asr_fasr_acc(output_expert, idx_atk, idx_clean_test, data, args)
                        asr_experts.append(round(asr_i.item() * 100, 2))
                        ca_experts.append(round(ca_i.item() * 100, 2))
                    logging.info('Every ASR: %s', asr_experts)
                    logging.info('Everry CA: %s', ca_experts)

                else:
                    output = test_model(induct_x,induct_edge_index,induct_edge_weights)
                    asr, fasr, ca = calculate_asr_fasr_acc(output, idx_atk, idx_clean_test, data, args)
                    print("ASR: {:.4f}; Flip_ASR: {:.4f}; CA: {:.4f}".format(asr, fasr, ca))

# %%           
                if args.test_model =='GMoE': # 净化路由器过程
                    from models.label_smoothing import get_smooth_soft_label, analyze_expert_variance, PurifiedGMoE
                    variances = analyze_expert_variance(bkd_tn_nodes, output, all_expert_outputs)
                    variances_attached = analyze_expert_variance(idx_attach, output, all_expert_outputs)
                    idx_hat_v_uncertain = [v for v, var in zip(bkd_tn_nodes, variances) if var > variances.mean() + variances.std()]
                    smoothed_soft_labels = get_smooth_soft_label(data.y, idx_hat_v_uncertain, num_classes=7, smoothing=0.9)

                    purified_gmoe = PurifiedGMoE(gmoe=test_model,device=device)
                    # 训练 purified router
                    purified_gmoe.fit(
                        poison_x,
                        poison_edge_index,
                        poison_edge_weights,
                        idx_train=bkd_tn_nodes,
                        soft_labels=smoothed_soft_labels,
                        epochs=100,
                        lr=0.01
                    )
                    output, routing_info_new = purified_gmoe(induct_x, induct_edge_index, induct_edge_weights)
                    asr, fasr, ca = calculate_asr_fasr_acc(output, idx_atk, idx_clean_test, data, args)
                    logging.info("(TopK) ASR: {:.4f}; Flip_ASR: {:.4f}; CA: {:.4f}".format(asr, fasr, ca))
    #     overall_asr += asr
    #     overall_ca += ca
    # overall_asr = overall_asr/len(seeds)
    # overall_ca = overall_ca/len(seeds)
    # print("Overall ASR: {:.4f} ({} model, Seed: {}, Gating: {})".format(overall_asr, args.test_model, seeds, args.router))
    # print("Overall Clean Accuracy: {:.4f}".format(overall_ca))


# %%
