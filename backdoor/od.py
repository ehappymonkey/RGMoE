import torch
import numpy as np
import torch.nn as nn

import torch.optim as optim

# Define the Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 2 * input_size // 3),
            nn.ReLU(True),
            nn.Linear(2 * input_size // 3, input_size // 3),
            nn.ReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(input_size // 3, 2 * input_size // 3),
            nn.ReLU(True),
            nn.Linear(2 * input_size // 3, input_size),
            nn.Sigmoid() # Use Sigmoid if the input data is normalized between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class MLPAE(nn.Module):
    def __init__(self, ori_x, trigger, device, epochs):
        super(MLPAE, self).__init__()
        self.device = device
        self.model = Autoencoder(len(ori_x[0])).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.epochs = epochs
        self.ori_x = ori_x
        self.trigger = trigger

    def fit(self):
        for epoch in range(self.epochs):
            output = self.model(self.ori_x)
            loss = self.criterion(output, self.ori_x)
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
           
    # def inference(self, input):
    #     self.model.eval()
    #     reconstruction_errors = []
    #     with torch.no_grad():
    #         for sample in input:
    #             reconstructed = self.model(sample)
    #             loss = self.criterion(reconstructed, sample)
    #             reconstruction_errors.append(loss.item())
    #     return reconstruction_errors
    def inference(self, input):
        self.model.eval()
        reconstruction_errors = []
        with torch.no_grad():
            for sample in input:
                reconstructed = self.model(sample)
                loss = self.criterion(reconstructed, sample)
                reconstruction_errors.append(loss)

        # Convert the list of tensors to a single tensor
        reconstruction_errors_tensor = torch.stack(reconstruction_errors)
        return reconstruction_errors_tensor



def reconstruct_prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,ori_x,ori_edge_index,device, idx=None, large_graph=True):
    poison_x = poison_x.to(device)
    # AE = MLPAE(poison_x, poison_x[len(ori_x):], device, args.rec_epochs)
    AE = MLPAE(poison_x, poison_x[len(ori_x):], device, args.rec_epochs)
    AE.fit()
    rec_score_ori = AE.inference(poison_x)
    # print(torch.mean(rec_score_ori))
    rec_score_triggers = AE.inference(poison_x[len(ori_x):])
    # print(rec_score)
    # print(torch.mean(rec_score_triggers))
    poison = rec_score_ori[len(ori_x):].detach().cpu().numpy()
    # Calculate the threshold for the top 3% largest values in rec_score_ori
    threshold = np.percentile(rec_score_ori.detach().cpu().numpy(), args.threhold)
    mask = rec_score_ori>threshold
    keep_edges_mask = ~(mask[poison_edge_index[0]] | mask[poison_edge_index[1]])
    # Filter the edge_index by the edges we want to keep
    filtered_poison_edge_index = poison_edge_index[:, keep_edges_mask]
    # Filter the edge weights similarly
    filtered_poison_edge_weights = poison_edge_weights[keep_edges_mask]
    # Check each element in poison against this threshold
    top_3_percent_flag = poison >= threshold
    # Calculate the percentage of poison elements that are in the top 3%
    percentage_in_top_3 = np.mean(top_3_percent_flag) * 100  # Convert to percentage
    print('Percentage of Triggers in Top3 Reconstruction Loss:',percentage_in_top_3)
    return filtered_poison_edge_index,filtered_poison_edge_weights