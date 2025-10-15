# This is a demo script for training and evaluating the RGMoE under PRBCD attack.

python rbcd_defense.py --dataset=ogbn-arxiv --ptb_rate=0.05 --model_name=GCN
python rbcd_defense.py --dataset=ogbn-arxiv --ptb_rate=0.05 --model_name=GMoE --num_experts=48 --topk=32 --topk_rerouting=8 --w_div=0.01 