
#!/usr/bin/env python
# coding: utf-8
# %% In[1]: 
import numpy as np
import torch
import json
from help_funcs import prune_unrelated_edge
import os, logging
import torch.nn.functional as F
from attack_utils import load_poisoned_files
from attack_utils import parse_args, set_seed, calculate_asr_fasr_acc
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
from torch_geometric.utils import to_undirected, coalesce
from od import reconstruct_prune_unrelated_edge


def set_logging(args):
    import logging

    log_filename = f"logs/{args.attack_method}/{args.dataset}_results_N={args.num_experts}.log"

    logging.basicConfig(
        level=logging.INFO,
        # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
         format='%(message)s',
        handlers=[
            logging.FileHandler(log_filename),  
            logging.StreamHandler()            
        ]
    )

def edge_to_sparse_adj(edge_index, edge_weight, num_nodes, device,
                make_undirected=True):

    if make_undirected:
        edge_index, edge_weight = to_undirected(edge_index, edge_weight, num_nodes=num_nodes)

    edge_index, edge_weight = coalesce(edge_index, edge_weight, num_nodes, num_nodes)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device, dtype=torch.float32)
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes),
                                dtype=torch.float32, device=device).coalesce()
    return adj
        

if __name__ == '__main__':
    args = parse_args()
    device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))
    # set_logging(args)
    if args.logging:
        logging.info(f"Dataset: {args.dataset}, Num Experts: {args.num_experts}, Top K: {args.topk}, w_div: {args.w_div}, w_certain: {args.w_certain}, Topk-Re: {args.topk_rerouting}")

    set_seed(args.seed)
    data, poison_x, poison_edge_index, poison_edge_weights, poison_labels, idx_attach, bkd_tn_nodes, idx_val, idx_clean_test, idx_atk, mask_edge_index, model = load_poisoned_files(args, device)

    # inject trigger on attack test nodes (idx_atk)''
    induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
    induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)]) 
    induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights,device)
    induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()

    # do pruning in test datas'''
    if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
        poison_edge_index,poison_edge_weights = prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,device,large_graph=False)
        induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device)

    elif(args.defense_mode == 'reconstruct'):
        poison_edge_index,poison_edge_weights = reconstruct_prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,data.x,data.edge_index,device, idx_attach, large_graph=True)

    # %% 
    from models.construct import model_construct  
    total_overall_asr = 0
    total_overall_ca = 0
    args.test_model = args.model_name
    rs = np.random.RandomState(args.seed) 
    seeds = rs.randint(1000,size=3)  
    overall_asr = 0
    overall_ca = 0
    fasr_list, ca_list = [],[]
    fasr_random_list, ca_random_list = [],[]
    robust_expert_rates, route_rates = [], []
    # for seed in seeds: 
    for seed in seeds[0:3]: 
        set_seed(int(seed)) # 
     
        save_dir = f"pre_trained"
        model_name = f"gmoe_model_{args.dataset}_{args.router}_{args.num_experts}_{args.topk}_w={args.w_div}_mar={args.margin}_{seed}.pth" # 265
        save_path = Path(save_dir) / model_name
        if args.test_model =='GMoE': 
            # if False:
            if save_path.exists():
                print(f"Loading pre-trained model from {save_path}...")
                test_model = model_construct(args,args.test_model,data,device).to(device)
                test_model.load_state_dict(torch.load(save_path))
            else:
                print(f"Model {model_name} does not exist, training GMoE model...")
                from utils import get_split
                if args.dataset == 'Cora':
                    _,bkd_tn_nodes,_,_,_ = get_split(args, data, device, train_ratio=0.7) 
                
                # in_channels = poison_x.size(1)
                # num_classes = int(poison_labels.max().item()) + 1
                # from models.GMoE_vanilla import GraphMoE
                # graphmoe = GraphMoE(in_channels, args.hidden, num_classes, args.dropout, conv_type=None, num_experts=args.num_experts, top_k=args.topk, router=args.router)
                # graphmoe.to(device)
                # graphmoe.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, induct_x, induct_edge_index,induct_edge_weights, bkd_tn_nodes, idx_val, idx_atk, idx_clean_test, data)
                test_model = model_construct(args,args.test_model,data,device).to(device)  
                test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, induct_x, induct_edge_index,induct_edge_weights, bkd_tn_nodes, idx_val, idx_atk, idx_clean_test, data, train_iters=args.epochs,verbose=False, margin=args.margin, args=args)
                torch.save(test_model.state_dict(), save_path)
                print(f"🔒 Pre-trained model saved to {save_path}. Before re-routing.")  
        elif args.model_name == 'GCN':
            # from models.GMI_v3 import GCN
            from models.GCN import GCN      
            test_model = GCN(poison_x.size(1), args.hidden, int(poison_labels.max().item()) + 1, dropout=0.5, lr=0.01, weight_decay=5e-4, layer=2,device=device).to(device)
            # test_model = GCN(data.num_node_features, args.hidden, data.y.max().item() + 1).to(device)
            from utils import get_split
            if args.dataset == 'Cora':
                _,bkd_tn_nodes,_,_,_ = get_split(args, data, device, train_ratio=0.7) 
            test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes)

# %%    # Evauation
        if args.test_model =='GMoE': 
            test_model.eval()
            with torch.no_grad():
                output, routing_info = test_model(induct_x,induct_edge_index,induct_edge_weights, w_mi=0, w_div=0)
            asr, fasr, ca = calculate_asr_fasr_acc(output, idx_atk, idx_clean_test, data, args)
            logging.info("(Before Re-Routing) ASR: {:.4f}; Flip_ASR: {:.4f}; CA: {:.4f}".format(asr, fasr, ca))
            test_model.check_every_expert(induct_x, induct_edge_index, induct_edge_weights, idx_atk, idx_clean_test, data, args, log=args.logging)
            # _, all_expert_logits, all_expert_probs = test_model.get_all_expert_outputs(induct_x, induct_edge_index, induct_edge_weights)
            # all_expert_outputs = all_expert_probs

        else:
            output = test_model(induct_x,induct_edge_index,induct_edge_weights)
            asr, fasr, ca = calculate_asr_fasr_acc(output, idx_atk, idx_clean_test, data, args)
            print("ASR: {:.4f}; Flip_ASR: {:.4f}; CA: {:.4f}".format(asr, fasr, ca))

# %%    # Perform Re-routing if GMoE
        fasr_random, ca_random = torch.tensor(0).to(device), torch.tensor(0).to(device)
        if args.test_model =='GMoE': 
            test_model.load_state_dict(torch.load(save_path))
            print(f"Loading from: {save_path}")

            test_model.eval()
            test_model.to(device)
            asr_experts, _ = test_model.check_every_expert(induct_x, induct_edge_index, induct_edge_weights, idx_atk, idx_clean_test, data, args)
            output, routing_info = test_model(induct_x,induct_edge_index,induct_edge_weights,w_mi=0, w_div=0)
            asr, fasr, ca = calculate_asr_fasr_acc(output, idx_atk, idx_clean_test, data, args)
            if args.logging == True:
                logging.info("(Before Re-Routing) ASR: {:.4f}; Flip_ASR: {:.4f}; CA: {:.4f}".format(asr, fasr, ca))
            else:
                print("(Before Re-Routing) ASR: {:.4f}; Flip_ASR: {:.4f}; CA: {:.4f}".format(asr, fasr, ca))

            output, gates_random = test_model.pred_with_random_experts(induct_x, induct_edge_index, induct_edge_weights, args.topk_rerouting)
            asr_random, fasr_random, ca_random = calculate_asr_fasr_acc(output, idx_atk, idx_clean_test, data, args)
            if args.logging ==True:
                logging.info("(Random Routing) ASR: {:.4f}; Flip_ASR: {:.4f}; CA: {:.4f}".format(asr_random, fasr_random, ca_random))
            else:
                print("(Random Routing) ASR: {:.4f}; Flip_ASR: {:.4f}; CA: {:.4f}".format(asr_random, fasr_random, ca_random))

            print("Re-routing...")
            all_expert_h2, all_expert_logits, all_expert_probs = test_model.get_all_expert_outputs(induct_x, induct_edge_index, induct_edge_weights)
            all_expert_outputs = all_expert_probs # log_softmax probs
            output = F.log_softmax(output, dim=-1)
            from models.label_smoothing import analyze_expert_variance, PurifiedGMoE, get_smooth_soft_label_2
            variances = analyze_expert_variance(bkd_tn_nodes, output, all_expert_outputs)
            variances_attached = analyze_expert_variance(idx_attach, output, all_expert_outputs)
            idx_hat_v_uncertain = [v for v, var in zip(bkd_tn_nodes, variances) if var > variances.mean() + 1*variances.std()]
            # idx_hat_v_uncertain = idx_hat_v_uncertain.to(torch.long)
            # idx_hat_v_uncertain = torch.cat([t.reshape(-1) for t in idx_hat_v_uncertain]).long().to(device)
            print(len(idx_hat_v_uncertain))
            num_classes = data.y.max().item() + 1
            print("y_c shape:", output.shape)
            print("all_expert_probs shape:", all_expert_probs[0].shape)
            
            smoothed_soft_labels_mean, smoothed_soft_labels = get_smooth_soft_label_2(poison_labels, all_expert_probs, output, idx_hat_v_uncertain, smoothing=0.0)
            # smoothed_soft_labels = get_smooth_soft_label(poison_labels, idx_hat_v_uncertain, num_classes=num_classes, smoothing=0.99) 
            purified_gmoe = PurifiedGMoE(gmoe=test_model, top_k=args.topk_rerouting, device=device) 
            purified_gmoe.fit(
                poison_x,
                poison_edge_index,
                poison_edge_weights,
                bkd_tn_nodes,
                idx_hat_v_uncertain,
                soft_labels=smoothed_soft_labels,
                w_certain=args.w_certain,
                epochs=args.re_epochs,
                lr=args.re_lr
            )
            output, routing_info_new = purified_gmoe.forward(induct_x, induct_edge_index, induct_edge_weights, args, rerouting=True)
            asr, fasr, ca = calculate_asr_fasr_acc(output, idx_atk, idx_clean_test, data, args)
            if args.logging ==True:
                logging.info("(After Re-Routing)  ASR: {:.4f}; Flip_ASR: {:.4f}; CA: {:.4f}".format(asr, fasr, ca))
            else:
                print("(After Re-Routing)  ASR: {:.4f}; Flip_ASR: {:.4f}; CA: {:.4f}".format(asr, fasr, ca))

        fasr_list.append(fasr.cpu().numpy())
        ca_list.append(ca.cpu().numpy())
        fasr_random_list.append(fasr_random.cpu().numpy())
        ca_random_list.append(ca_random.cpu().numpy())

    overall_asr = np.mean(fasr_list)  
    overall_ca = np.mean(ca_list)    
    overall_fasr_random = np.mean(fasr_random_list)
    overall_ca_random = np.mean(ca_random_list)


    std_asr = np.std(fasr_list)  
    std_ca = np.std(ca_list)    
    std_fasr_random = np.std(fasr_random_list)
    std_ca_random = np.std(ca_random_list)


    print("Overall Random Routing ASR: {:.2f} ± {:.2f}".format(overall_fasr_random*100, std_fasr_random*100))
    print("Overall Random Routing Clean Accuracy: {:.2f} ± {:.2f}".format(overall_ca_random*100, std_ca_random*100))
    if args.logging ==True:
        logging.info("Overall ASR: {:.2f} ± {:.2f} ({} model, Seed: {})".format(overall_asr*100, std_asr*100, args.test_model, seeds))
        logging.info("Overall Clean Accuracy: {:.2f} ± {:.2f}".format(overall_ca*100, std_ca*100))
    else:
        print("Overall ASR: {:.2f} ± {:.2f} ({} model, Seed: {})".format(overall_asr*100, std_asr*100, args.test_model, seeds))
        print("Overall Clean Accuracy: {:.2f} ± {:.2f}".format(overall_ca*100, std_ca*100))




    # %%
