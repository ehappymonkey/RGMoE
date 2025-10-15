# Cora
# python -u attack_prepare.py \
#     --dataset=Cora\
#     --homo_loss_weight=50\
#     --vs_ratio=0.05\
#     --selection_method=cluster_degree\
#     --homo_boost_thrd=0.5\
#     --epochs=200\
#     --trojan_epochs=400


# # Pubmed
# python -u attack_prepare.py \
#     --dataset=Pubmed\
#     --homo_loss_weight=50\
#     --vs_ratio=0.05\
#     --selection_method=cluster_degree\
#     --homo_boost_thrd=0.1\
#     --epochs=200\
#     --trojan_epochs=2000


# # Flickr

python -u attack_prepare.py \
    --dataset=Flickr\
    --homo_loss_weight=100\
    --vs_ratio=0.05\
    --selection_method=cluster_degree\
    --homo_boost_thrd=0.8\
    --epochs=200\
    --trojan_epochs=400

# # OGBN-Arixv
# python -u attack_prepare.py  \
#     --dataset=ogbn-arxiv\
#     --homo_loss_weight=200\
#     --vs_ratio=0.03\
#     --selection_method=cluster_degree\
#     --homo_boost_thrd=0.8\
#     --epochs=800\
#     --trojan_epochs=800
