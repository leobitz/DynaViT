import torch
import torch.nn as nn
import torchsummary
import random
import numpy as np
from torch.distributions import Categorical


class Skipper(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers, dropout=0.4, inference=False) -> None:
        super(Skipper, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(hidden_size, 2) for x in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        if inference:
            self.forward_x = self.forward_inference
        else:
            self.forward_x = self.forward_train

    def forward_train(self, x, hidden, layer_index):
        #print(x.device,  self.linears[layer_index].weight.device)
        # x = x.to(self.linears[layer_index].weight.device)
        self.dropout(x)
        x = self.linears[layer_index](x)

        skip_pred = torch.softmax(x, axis=1)
        m = Categorical(skip_pred)
        action = m.sample()
        log_action = m.log_prob(action)
        return x, action, log_action, hidden

    def forward_inference(self, x, hidden, layer_index):
        self.dropout(x)
        x = self.linears[layer_index](x)
        x = (x[:, 1] > x[:, 0]).long()
        return x 
    
    def forward(self, x, hidden, layer_index):
        return self.forward_x(x, hidden, layer_index)


class Baseline(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers, dropout=0.4) -> None:
        super(Baseline, self).__init__()
        # self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc2 = nn.ModuleList([nn.Linear(input_size, hidden_size) for x in range(n_layers)])
        self.linears = nn.ModuleList([nn.Linear(hidden_size, 1) for x in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        # self.fc1 = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden, layer_index):
        # x, hidden = self.lstm(x, hidden)
        # x = x.to(self.linears[layer_index].weight.device)
        x = self.fc2[layer_index](x)
        self.dropout(x)
        x = torch.relu(x)
        x = self.linears[layer_index](x)
        # x = self.fc1(x)
        return x, hidden

        

