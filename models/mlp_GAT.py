import torch
import torch.nn as nn
import torch.nn.functional as F
import math


###MLP with lienar output


class MultiHeadAttentionActor(nn.Module):
    def __init__(self,hidden_dim, n_nodes, n_heads, attn_dropout=0.1, dropout=0):
        super(MultiHeadAttentionActor, self).__init__()

        # self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_nodes = n_nodes
        self.n_heads = n_heads
        self.head_dim = self.hidden_dim / self.n_heads
        self.norm = 1 / math.sqrt(self.head_dim)

        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # self.h = MLP(self, num_layers, hidden_node_dim, hidden_node_dim)
        # self.a = nn.Linear(hidden_dim*2, hidden_dim)

    def forward(self, h, mask=None):
        h = h.squeeze(0)
        # h = self.a(h)
        # n_nodes, hidden_node_dim = h.size()
        n_nodes = h.size(0)
        hidden_node_dim = h.size(1)
        # h = h.transpose(1,0)
        # batch_size, n_nodes,n_nodes, hidden_edge_dim  = e.size()
        Q = self.q(h).view(n_nodes, self.n_heads, -1)
        K = self.k(h).view(n_nodes, self.n_heads, -1)
        V = self.v(h).view(n_nodes, self.n_heads, -1)

        compatibility = self.norm * torch.matmul(Q.reshape(self.n_heads, n_nodes, -1), K.reshape(self.n_heads, -1,
                                                                                                 n_nodes))  # (batch_size,n_heads,1,hidden_dim)*(batch_size,n_heads,hidden_dim,n_nodes)
        # new_compatibility = new_compatibility.squeeze(3)  # (batch_size,n_heads,n_nodes)
        # mask = mask.unsqueeze(1).expand_as(new_compatibility)
        # u_i = new_compatibility.masked_fill(mask.bool(), float("-inf"))

        scores = F.softmax(compatibility, dim=-1)  # (batch_size,n_heads,n_nodes)
        # scores = scores.unsqueeze(2)
        out_put = torch.matmul(scores, V.view(self.n_heads, n_nodes,
                                              -1))  # (batch_size,n_heads,1,n_nodes )*(batch_size,n_heads,n_nodes,head_dim)
        out_put = out_put.view(-1, self.hidden_dim)  # （batch_size,n_heads,hidden_dim）
        out_put = self.fc(out_put).unsqueeze(0)
        # pooled_h = (h_nodes.mean(dim=0).reshape(1, self.hidden_node_dim)).to(self.device)

        return out_put

class AttentionBasedMLP(nn.Module):
    def __init__(self, hidden_dim, out_put_dim, n_nodes, n_heads,dropout=0.0
                 ):
        super().__init__()

        # self.input_node_dim = input_node_dim
        self.hidden_node_dim = hidden_dim
        self.out_put_dim = out_put_dim
        self.dropout = dropout
        self.FFN_h_layer1 = nn.Linear(hidden_dim, 128)
        self.FFN_h_layer2 = nn.Linear(128, 1)

        self.attention = MultiHeadAttentionActor(hidden_dim, n_nodes, n_heads)

    def forward(self, h):
        h=self.attention(h)
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)
        return h


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class MLPActor(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLPActor, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            '''
            self.batch_norms = torch.nn.ModuleList()
            '''

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            '''
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            '''

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                '''
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                '''
                h = torch.tanh(self.linears[layer](h))
                # h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)


class MLPCritic(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLPCritic, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            '''
            self.batch_norms = torch.nn.ModuleList()
            '''

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            '''
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            '''

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                '''
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                '''
                h = torch.tanh(self.linears[layer](h))
                # h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)


