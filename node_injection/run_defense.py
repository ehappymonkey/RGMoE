
import torch
import torch.nn.functional as F
import numpy as np
from tdgia_utils import parse_args
from gmoe import GraphMoE, calculate_acc
from pathlib import Path
import logging


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

 


args = parse_args()
# set_logging(args)
if args.model_name == 'GMoE':
    logging.info(f"Dataset: {args.dataset}, Num Experts: {args.num_experts}, Top K: {args.topk}, w_div: {args.w_div}, w_certain: {args.w_certain}, Topk-Re: {args.topk_rerouting}")
else: 
    logging.info(f"Dataset: {args.dataset}, Model: {args.model_name}, N Injection Max: {args.n_inject_max}, N Edge Max: {args.n_edge_max}")
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))
# load ptb graph
save_path = f'./tdgia_poisoning_{args.dataset}_{args.n_inject_max}_{args.n_edge_max}.pt'
print(f'Loading poisoned graph from {save_path}...')
all_data = torch.load(save_path)
features,labels,edge_index,edge_weight,idx_train,idx_val,idx_test = all_data['features'],all_data['labels'],all_data['edge_index'],all_data['edge_weight'],all_data['idx_train'],all_data['idx_val'],all_data['idx_test']
features,edge_index,edge_weight, labels = features.to(device),edge_index.to(device),edge_weight.to(device),labels.to(device)
idx_train,idx_val,idx_test = idx_train.to(device),idx_val.to(device),idx_test.to(device)

rs = np.random.RandomState(args.seed) 
seeds = rs.randint(1000,size=3) 
acc_ptb,acc_ptb_randomr = [],[]
robust_expert_rate, route_rate = [],[]
for seed in seeds:
    print(f"Using seed: {seed}")
    set_seed(int(seed))
    if args.model_name == 'GCN':
        from GCN import GCN
        test_model = GCN(nfeat=features.shape[1], nhid=32, nclass = max(labels).item()+1, dropout=0.5, lr=0.01, weight_decay=5e-4, layer=2,device=device).to(device)
        test_model.fit(features, edge_index, edge_weight, labels, idx_train)
        ptb_acc = test_model.test(features, edge_index, edge_weight, labels, idx_test)
        print(f"Acc of GCN After Ptb: {ptb_acc:.4f}")
        acc_ptb_randomr.append(0)


    if args.model_name == 'GMoE':
        save_dir = f"pre_trained/{args.dataset}"
        model_name = f"gmoe_model_{args.n_inject_max}_{args.n_edge_max}_{args.router}_{args.num_experts}_{args.topk}_w={args.w_div}_mar={args.margin}_{seed}.pth" # 265
        save_path = Path(save_dir) / model_name
        if save_path.exists():
            print(f"Loading pre-trained model from {save_path}...")
            test_model = GraphMoE(in_dim=features.shape[1],hidden_dim=32,num_classes=labels.max().item()+1,num_experts=args.num_experts, dropout=args.dropout,w_bala=args.w_bala, w_div=args.w_div,w_mi=args.w_mi).to(device)
            test_model.load_state_dict(torch.load(save_path))
        else:
            print("No pre-trained model found, training from scratch...")
            test_model = GraphMoE(in_dim=features.shape[1],hidden_dim=32,num_classes=labels.max().item()+1,num_experts=args.num_experts, dropout=args.dropout,w_bala=args.w_bala, w_div=args.w_div,w_mi=args.w_mi).to(device)
            test_model.fit(features, edge_index, edge_weight, labels, idx_train, idx_val, idx_test, train_iters=200)
            torch.save(test_model.state_dict(), save_path)
            print(f"🔒 Pre-trained model saved to {save_path}.")     
    
    # Evaluation & Re-routing
    if args.model_name == 'GMoE':
        _, ca_experts = test_model.check_every_expert(features, edge_index, edge_weight, labels, idx_test, log=True)
        output, _ = test_model(features, edge_index, edge_weight)
        acc = calculate_acc(output, labels, idx_test)
        print(f"Acc of GMoE Before Re-routing: {acc:.4f}")
        output, gates_random = test_model.pred_with_random_experts(features, edge_index, edge_weight, args.topk_rerouting)

        ptb_acc_random = calculate_acc(output, labels, idx_test)
        print(f"(Random Routing) Acc: {ptb_acc_random*100:.2f}")  
        acc_ptb_randomr.append(ptb_acc_random)

        print("Re-routing...")
        all_expert_h2, all_expert_logits, all_expert_probs = test_model.get_all_expert_outputs(features, edge_index, edge_weight)
        all_expert_outputs = all_expert_probs 
        output = F.log_softmax(output, dim=-1)
        from label_smoothing import get_smooth_soft_label, get_smooth_soft_label_2, analyze_expert_variance, PurifiedGMoE
        # idx_all = torch.cat([idx_train, idx_val, idx_test])
        idx_all = idx_train
        variances = analyze_expert_variance(idx_all, output, all_expert_outputs)
        variances_attached = analyze_expert_variance(idx_test, output, all_expert_outputs)
        idx_hat_v_uncertain = [v for v, var in zip(idx_all, variances) if var > variances.mean() + 1*variances.std()]
        #idx_hat_v_uncertain = idx_attach
        print(len(idx_hat_v_uncertain))
        num_classes = labels.max().item() + 1
        smoothed_soft_labels, _ = get_smooth_soft_label_2(labels, all_expert_probs, output, idx_hat_v_uncertain, smoothing=0.1)  
        # smoothed_soft_labels = get_smooth_soft_label(labels, idx_hat_v_uncertain, num_classes=num_classes, smoothing=0.99)  
        purified_gmoe = PurifiedGMoE(gmoe=test_model, top_k=args.topk_rerouting, device=device) 
        purified_gmoe.fit(
            features,
            edge_index,
            edge_weight,
            idx_train,
            idx_hat_v_uncertain,
            soft_labels=smoothed_soft_labels,
            w_certain=args.w_certain,
            epochs=200,
            lr=1e-3
        )
        output, routing_info_new = purified_gmoe.forward(features, edge_index, edge_weight, args, rerouting=True)
        ptb_acc = calculate_acc(output, labels, idx_test)
        print("(After Re-Routing) Acc: {:.4f}".format(ptb_acc))
  
    # Note that the values here a bit more noisy than in the evasion case:
    print(f'Ptbed Acc of TDGIA {ptb_acc:.3f}')
    acc_ptb.append(ptb_acc)

acc_ptb_mean = np.mean(acc_ptb)
acc_ptb_std = np.std(acc_ptb)
acc_ptb_randomr_mean = np.mean(acc_ptb_randomr)
acc_ptb_randomr_std = np.std(acc_ptb_randomr)
logging.info("Overall Ptb ACC (Poisoning): {:.2f} ± {:.2f} (Seed: {})".format(acc_ptb_mean*100, acc_ptb_std*100, seeds))
logging.info("Overall Ptb ACC (Random Routing): {:.2f} ± {:.2f} (Seed: {})".format(acc_ptb_randomr_mean*100, acc_ptb_randomr_std*100, seeds))

