import torch.nn as nn
from models.mlp_GAT import AttentionBasedMLP
from models.mlp_GAT import MLPActor
from models.mlp_GAT import MLPCritic
import torch.nn.functional as F
from models.MHA_Based_Encoding import MultiHeadAttentionLayer
import torch

import time

class ActorCritic(nn.Module):
    def __init__(self,
                 n_j,
                 n_m,
                 # feature extraction net unique attributes:
                 n_heads,
                 n_nodes,
                 # num_layers,
                 # num_mlp_layers,
                 # learn_eps,
                 # neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 # feature extraction net MLP attributes:
                 # num_mlp_layers_feature_extract,
                 # actor net MLP attributes:
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 # actor net MLP attributes:
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 # actor/critic/feature_extraction shared attribute
                 conv_layers,
                 device
                 ):
        super(ActorCritic, self).__init__()
        # job size for problems, no business with network
        self.n_j = n_j
        # machine size for problems, no business with network
        self.n_m = n_m
        self.n_ops_perjob = n_m
        self.device = device
        self.hidden_dim = hidden_dim


        self.feature_extract = nn.ModuleList([MultiHeadAttentionLayer(hidden_dim,
                                                                      n_heads,
                                                                      n_nodes,
                                                                      input_dim,
                                                                      ) for i in range(conv_layers)]).to(
            device)

        
        self.actor = MLPActor(num_mlp_layers_actor, hidden_dim * 2, hidden_dim_actor, 1).to(device)
        self.critic = MLPCritic(num_mlp_layers_critic, hidden_dim, hidden_dim_critic, 1).to(device)
        self.IniEmbed = nn.Linear(input_dim, hidden_dim).to(device)
        self.Linear2 = nn.Linear(257, 256).to(device)

        
    def forward(self,
                x,
                graph_pool,
                candidate,
                mask,
                device,):


        x = self.IniEmbed(x)
        # x = F.leaky_relu(x)


        for conv in self.feature_extract:
            h_nodes = conv(x)

        h_pooled = graph_pool.mm(h_nodes)
        
  
        dummy = candidate.unsqueeze(-1).expand(-1, self.n_j, h_nodes.size(-1))
        candidate_feature = torch.gather(h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy)

        
        h_pooled_repeated = h_pooled.unsqueeze(1).expand_as(candidate_feature)

        # prepare policy feature: concat row work remaining feature
        durfea2mat = x[:, 1].reshape(shape=(-1, self.n_j, self.n_m))
        mask_right_half = torch.zeros_like(durfea2mat)
        mask_right_half.put_(candidate, torch.ones_like(candidate, dtype=torch.float))
        mask_right_half = torch.cumsum(mask_right_half, dim=-1)
        # calculate work remaining and normalize it with job size
        wkr = (mask_right_half * durfea2mat).sum(dim=-1, keepdim=True)/self.n_ops_perjob

        # concatenate feature
        concateFea = torch.cat((wkr, candidate_feature, h_pooled_repeated), dim=-1)
        # concateFea = torch.cat((candidate_feature, h_pooled_repeated), dim=-1)
        concateFea = self.Linear2(concateFea)
        concateFea.squeeze(0)
        candidate_scores = self.actor(concateFea)


        # perform mask
        mask_reshape = mask.reshape(candidate_scores.size())
        candidate_scores[mask_reshape] = float('-inf')


        pi = F.softmax(candidate_scores, dim=1)
        # pi = torch.nn.DataParallel(pi)
        # pi = pi.cuda()
        v = self.critic(h_pooled)


        return pi, v


if __name__ == '__main__':
    print('Go home')