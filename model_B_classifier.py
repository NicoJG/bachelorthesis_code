import torch
from torch import nn

# Define the model
# similar to https://gitlab.cern.ch/nnolte/DeepSetTagging/-/blob/cf286579b7db4ff21341eb4b06fc8f726168a8c9/model.py
class DeepSetModel(nn.Module):
    def __init__(self, n_features, n_latent_features):
        super(DeepSetModel, self).__init__()
        
        self.n_features = n_features
        self.n_latent_features = n_latent_features
        
        # Neural Network for the tracks:
        self.phi_stack = nn.Sequential(
            nn.Linear(n_features, n_features*2),
            nn.ReLU(),
            nn.Linear(n_features*2, n_latent_features),
            nn.ReLU()
        )
        
        # Sum up all outputs of the phi_stack for each event
        # done in the forward function
        
        # Neural Network for the events
        self.rho_stack = nn.Sequential(
            nn.Linear(n_latent_features, n_latent_features*2),
            nn.ReLU(),
            nn.Linear(n_latent_features*2, n_latent_features),
            nn.ReLU(),
            nn.Linear(n_latent_features, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, event_ids):
        # x must have shape (tracks, features)
        # event_ids must have shape (tracks,)
        x = self.phi_stack(x)
        
        # get an event_index_by_track array
        idxs = torch.zeros_like(event_ids)
        idxs[1:] = torch.cumsum(event_ids[1:] != event_ids[:-1], 0)
        
        # sum up the latent features of all tracks per event
        temp = torch.zeros(idxs[-1]+1, self.n_latent_features)
        x = temp.index_add(0, idxs, x)
    
        x = self.rho_stack(x)
        return x

        
        