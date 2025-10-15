# This is a demo script for training and evaluating the RGMoE model.

python run_defense.py --dataset=ogbn-arxiv --n_inject_max=1000 --n_edge_max=50 --model_name=GCN
python run_defense.py --dataset=ogbn-arxiv --n_inject_max=1000 --n_edge_max=50 --model_name=GMoE --num_experts=48 --topk=32 --w_div=0.001 -