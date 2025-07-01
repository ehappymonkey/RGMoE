
#!/usr/bin/env python
# coding: utf-8


# In[1]: 
import numpy as np
import torch

from help_funcs import prune_unrelated_edge
import os

# Training settings
from attack_utils import parse_args, set_seed, count_experts_selection, calculate_asr_fasr_acc

args = parse_args()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))
# set_seed(args.seed)
print(args)

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


from models.construct import model_construct
# models = ['GCN','GAT', 'GraphSage']
args.num_experts = 8 # 这里方便调整
args.topk = 2
args.router = 'noisytopk'
# args.router = 'cosine'
print('Gating Strategy:', args.router)
models = ['GCN']
total_overall_asr = 0
total_overall_ca = 0
for test_model in models:
    args.test_model = test_model
    rs = np.random.RandomState(args.seed) # 生成一个固定状态
    seeds = rs.randint(1000,size=1) 
    print("seeds: {}".format(seeds))

    overall_asr = 0
    overall_ca = 0
    for seed in seeds: 
        set_seed(seed)

        # if args.test_model =='GMoE': # 
            
        test_model = model_construct(args,args.test_model,data,device).to(device) # model_construct中加入我们的GMoE
        test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val, train_iters=args.epochs,verbose=False)
        if args.test_model =='GMoE':
            output, routing_info = test_model(poison_x,poison_edge_index,poison_edge_weights)

            # print("训练集上污染节点的专家选择:")
            # topk_indices = routing_info[1][1][idx_attach]
            # # topk_indices = routing_info[0][1][idx_attach] # 用单层GMoE时只有一个路由器
            # print(topk_indices.shape)
            # count_experts_selection(topk_indices, args.num_experts)

        else:
            output = test_model(poison_x,poison_edge_index,poison_edge_weights)
        
        train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
        print("ASR on training nodes: {:.4f}".format(train_attach_rate)) # attach trigger的污染节点输出y_t的概率（训练集）
        
        induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
        induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
        clean_acc = test_model.test(poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)

        print("accuracy on clean test nodes: {:.4f}".format(clean_acc))


        if(args.evaluate_mode == '1by1'):
            from torch_geometric.utils  import k_hop_subgraph
            overall_induct_edge_index, overall_induct_edge_weights = induct_edge_index.clone(),induct_edge_weights.clone()
            asr = 0
            flip_asr = 0
            flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
            for i, idx in enumerate(idx_atk):
                idx=int(idx)
                sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask  = k_hop_subgraph(node_idx = [idx], num_hops = 2, edge_index = overall_induct_edge_index, relabel_nodes=True) # sub_mapping means the index of [idx] in sub)nodeset
                ori_node_idx = sub_induct_nodeset[sub_mapping]
                relabeled_node_idx = sub_mapping
                sub_induct_edge_weights = torch.ones([sub_induct_edge_index.shape[1]]).to(device)
                with torch.no_grad():
                    # inject trigger on attack test nodes (idx_atk)'''
                    induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(relabeled_node_idx,poison_x[sub_induct_nodeset],sub_induct_edge_index,sub_induct_edge_weights,device)
                    induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
                    # # do pruning in test datas'''
                    if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
                        induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device,False)
                    # attack evaluation
                    output, _ = test_model(induct_x,induct_edge_index,induct_edge_weights)
                    train_attach_rate = (output.argmax(dim=1)[relabeled_node_idx]==args.target_class).float().mean()
                    asr += train_attach_rate
                    if(data.y[idx] != args.target_class):
                        flip_asr += train_attach_rate
                    induct_x, induct_edge_index,induct_edge_weights = induct_x.cpu(), induct_edge_index.cpu(),induct_edge_weights.cpu()
                    output = output.cpu()
            asr = asr/(idx_atk.shape[0])
            flip_asr = flip_asr/(flip_idx_atk.shape[0])
            print("Overall ASR: {:.4f}".format(asr))
            print("Flip ASR: {:.4f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
        elif(args.evaluate_mode == 'overall'):
            # inject trigger on attack test nodes (idx_atk)'''
            induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights,device)
            induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
            # do pruning in test datas'''
            if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
                induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device)
            # attack evaluation
            if args.test_model =='GMoE':
                output, routing_info = test_model(induct_x,induct_edge_index,induct_edge_weights)
                # 打印测试节点的专家选择

                # print("测试集污染节点的专家选择:")
                # topk_indices_atk = routing_info[1][1][idx_atk] # 第二层moe的第二个对应topk_indices
                # # topk_indices_atk = routing_info[0][1][idx_atk] # 用单层GMoE时只有一个路由器
                # print(topk_indices_atk.shape)
                # count_experts_selection(topk_indices_atk, args.num_experts)
                
                print("测试集所有节点的专家选择:")
                topk_indices = routing_info[0][1]
                print(topk_indices.shape)
                count_experts_selection(topk_indices, args.num_experts)


            else:
                output = test_model(induct_x,induct_edge_index,induct_edge_weights)

            asr, fasr, ca = calculate_asr_fasr_acc(output, idx_atk, idx_clean_test, data, args)
            print("ASR: {:.4f}; Flip_ASR: {:.4f}; CA: {:.4f}".format(asr, fasr, ca))
            
            if args.test_model == 'GMoE':
                # 观察（第二层）每个专家输出结果
                all_expert_outputs = test_model.forward_every_expert(induct_x,induct_edge_index,induct_edge_weights)
                for i, output in enumerate(all_expert_outputs):
                    asr_i, fasr_i, ca_i = calculate_asr_fasr_acc(output, idx_atk, idx_clean_test, data, args)
                    print(f"Expert {i}: ASR = {asr_i*100:.2f}%, Flip_ASR = {fasr_i*100:.2f}%, CA = {ca_i*100:.2f}%")
                    # print(f"Expert {int(i)}: ASR = {(asr_i * 100):.2f}%, Flip_ASR = {(fasr_i * 100):<8}{'':3}", end='')

                # 观察随机路由专家输出结果
                random_expert_output = test_model.forward_random_expert(induct_x,induct_edge_index,induct_edge_weights)
                asr_i, fasr_i, ca_i = calculate_asr_fasr_acc(random_expert_output, idx_atk, idx_clean_test, data, args)
                print(f"Random Expert: ASR = {asr_i*100:.2f}%, Flip_ASR = {fasr_i*100:.2f}%, CA = {ca_i*100:.2f}%")

        overall_asr += asr
        overall_ca += ca

        # 一次seed评估结束，释放内存
        del test_model
        torch.cuda.empty_cache()

    overall_asr = overall_asr/len(seeds)
    overall_ca = overall_ca/len(seeds)
    print("Overall ASR: {:.4f} ({} model, Seed: {}, Gating: {})".format(overall_asr, args.test_model, seeds, args.router))
    print("Overall Clean Accuracy: {:.4f}".format(overall_ca))


# %%
