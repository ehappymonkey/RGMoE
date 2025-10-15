# %%
from models.GCN import GCN
from models.GCN_Encoder import GCN_Encoder
from models.GMoE_v9 import GraphMoE
#from models.GMoE_vanilla import GraphMoE

def model_construct(args,model_name,data,device):
    if(args.dataset == 'Reddit2'):
        use_ln = True
        layer_norm_first = False
    else:
        use_ln = False
        layer_norm_first = False
    if (model_name == 'GCN'):
        model = GCN(nfeat=data.x.shape[1],\
                    nhid=args.hidden,\
                    nclass= int(data.y.max()+1),\
                    dropout=args.dropout,\
                    lr=args.train_lr,\
                    weight_decay=args.weight_decay,\
                    device=device,
                    use_ln=use_ln,
                    layer_norm_first=layer_norm_first)
    elif(model_name == 'GCN_Encoder'):
        model = GCN_Encoder(nfeat=data.x.shape[1],                    
                            nhid=args.hidden,                    
                            nclass= int(data.y.max()+1),                    
                            dropout=args.dropout,                    
                            lr=args.train_lr,                    
                            weight_decay=args.weight_decay,                    
                            device=device,
                            use_ln=use_ln,
                            layer_norm_first=layer_norm_first)
    elif(model_name == 'GMoE'):
        model = GraphMoE(in_dim=data.x.shape[1],\
                         hidden_dim=args.hidden,\
                         num_classes=int(data.y.max()+1),\
                         dropout=args.dropout,\
                         conv_type=args.conv_type,\
                         num_experts = args.num_experts,\
                        #  top_k=args.topk,\
                         router=args.router,\
                         num_moe_layers=1,\
                         w_bala=args.w_bala, \
                         w_div=args.w_div, \
                         w_mi=args.w_mi,  
                        )
                         
    else:
        print("Not implement {}".format(model_name))
    return model