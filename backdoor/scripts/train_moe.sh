#!/usr/bin/env bash

# python defense.py --dataset=Cora        --num_experts=48 --topk=24  --w_div=10    --topk_rerouting=8
python defense.py --dataset=Pubmed      --num_experts=48 --topk=36 --w_div=1  --topk_rerouting=8
python defense.py --dataset=Flickr      --num_experts=48 --topk=24 --w_div=6    --topk_rerouting=8
python defense.py --dataset=ogbn-arxiv --vs_ratio=0.03 --num_experts=48 --topk=24 --w_div=0.2 --topk_rerouting=8
