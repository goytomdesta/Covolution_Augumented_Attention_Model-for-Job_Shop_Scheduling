from models.mlp_GAT import MLP
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np


# import sys
# sys.path.append("models/")

class MultiHeadAttention(nn.Module):
    # def __init__(self, hidden_dim, n_nodes, input_dim, n_heads, attn_dropout=0.1, dropout=0):
    def __init__(self, hidden_dim, n_nodes, input_dim, n_heads,dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_nodes = n_nodes
        self.n_heads = n_heads
        self.head_dim = self.hidden_dim / self.n_heads
        self.norm = 1 / math.sqrt(self.head_dim)

        self.IniEmbed = nn.Linear(input_dim, hidden_dim)

        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, h):

        h = h.reshape(-1, self.hidden_dim)
        n_nodes, hidden_dim = h.size()
        # h = F.leaky_relu(h)
        Q = self.q(h).view(n_nodes, self.n_heads, -1)
        K = self.k(h).view(n_nodes, self.n_heads, -1)
        V = self.v(h).view(n_nodes, self.n_heads, -1)

        # compatibility =  torch.matmul(Q.reshape(self.n_heads, n_nodes, -1), K.reshape(self.n_heads, -1, n_nodes)) * 0.5
        torch.cuda.empty_cache()
        # einsum multiplication
        K = K.reshape(n_nodes, -1, self.n_heads)
        compatibility = torch.einsum('ijk,ikl->ijl', Q, K) / (hidden_dim // self.n_heads)* 0.5

        scores = F.softmax(compatibility, dim=-1)
        # scores = F.softmax(adaptive_weights, dim=1).unsqueeze(-1)   # (batch_size,n_heads,n_nodes)
        # scores = adaptive_weights.unsqueeze(-1)
        del compatibility

        # # einsum multiplication
        V = V.reshape(n_nodes, -1, self.n_heads)
        out_put = torch.einsum("blh,bdh->bld", scores, V)
        out_put = out_put.transpose(1, 2).contiguous().view(n_nodes, hidden_dim)
        out_put = self.fc(out_put)

        return out_put   # (batch_size,hidden_dim)

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, hidden_dim, n_heads, n_nodes, input_dim,dropout=0.00, batch_norm=True, residual=True,
                 use_bias=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_nodes = n_nodes
        self.input_dim = input_dim
        # self.edge_index = edge_index
        self.dropout = dropout
        self.residual = residual
        # self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        feed_forward_hidden = 512

        self.attentions = MultiHeadAttention(hidden_dim, n_nodes, input_dim, n_heads,dropout=0.00)
        self.LinTra = nn.Linear(input_dim, hidden_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(hidden_dim)
        # FFN for h
        self.FFN_h_layer1 = nn.Linear(hidden_dim, feed_forward_hidden)
        self.FFN_h_layer2 = nn.Linear(feed_forward_hidden, hidden_dim)
        self.FFN_h_layer3 = nn.Linear(hidden_dim, feed_forward_hidden)
        self.FFN_h_layer4 = nn.Linear(feed_forward_hidden, hidden_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(hidden_dim)

        self.linear_w_r = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_u_r = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_w_z = nn.Linear(hidden_dim, hidden_dim, bias=False)  ### Giving bias to this layer (will count as b_g so can just initialize negative)
        self.linear_u_z = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_w_g = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_u_g = nn.Linear(hidden_dim, hidden_dim, bias=False)

        if self.batch_norm:
            self.batch_norm3_h = nn.BatchNorm1d(hidden_dim)

        self.conv1d = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.depthwise_conv1d = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, groups=hidden_dim)
        # self.Glu = GLUActivation(hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h):

        # Multi Head Attention
        # h_in = h

        h_a = h
        # h = self.layer_norm(h)
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        # h = h * torch.sigmoid(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)
        # h = h_a + h
        h = h * torch.sigmoid(h_a)
        # h = h_a + (h) * 0.5

        # h = h_a * torch.sigmoid(h)
        # h = self.layer_norm(h)

        re = h
        xc = h.unsqueeze(-1)
        # residual = xc
        xc = self.conv1d(xc)
        # xc = xc + residual
        h_c = xc.squeeze(-1)

        h_c = h_c * torch.sigmoid(h)

        h_c = h_c.unsqueeze(-1)
        h_c = self.depthwise_conv1d(h_c)
        h_c = h_c.squeeze(-1)

        if self.batch_norm:
            h_c = self.batch_norm2_h(h_c)

        h_c = F.relu(h_c)
        h_c = h_c.unsqueeze(-1)
        h_c = self.conv1d(h_c)
        h_c = h_c.squeeze(-1)

        h_c = re + h_c
        # if self.batch_norm:
        #     h_c = self.batch_norm2_h(h_c)

        h_attn_out = self.attentions(h_c)
        h_attn_out = F.dropout(h_attn_out, self.dropout, training=self.training)

        # Gating Mechanism starts here
        z = torch.sigmoid(self.linear_w_z(h_c) + self.linear_u_z(h_attn_out))  # MAKE SURE THIS IS APPLIED ON PROPER AXIS
        r = torch.sigmoid(self.linear_w_r(h_c) + self.linear_u_r(h_attn_out))
        h_hat = torch.tanh(self.linear_w_g(h_c) + self.linear_u_g(r * h_attn_out))  # Note elementwise multiplication of r and x
        h = (1. - z) * h_attn_out + z * h_hat
        h = h * torch.sigmoid(h_attn_out)

        h = self.fc1(h)

        return h
