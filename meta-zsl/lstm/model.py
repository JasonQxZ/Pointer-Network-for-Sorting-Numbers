#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikai Wang
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter


class net_LSTM(nn.Module):

    def __init__(self, num_inputs, num_hidden, num_outputs, num_layers):
        super(net_LSTM, self).__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=num_inputs,
                            hidden_size=num_hidden,
                            num_layers=num_layers)
        self.fc = nn.Linear(in_features = num_hidden, 
                out_features = num_outputs)
        # The hidden state is a learned parameter
        self.lstm_h_bias = Parameter(torch.randn(self.num_layers,1, self.num_hidden) * 0.05)
        self.lstm_c_bias = Parameter(torch.randn(self.num_layers,1, self.num_hidden) * 0.05)
        self.reset_parameters()

    def create_new_state(self, batch_size):
        # Dimension: (num_layers * num_directions, batch, hidden_size)
        lstm_h = self.lstm_h_bias.clone().repeat(1, batch_size, 1)
        lstm_c = self.lstm_c_bias.clone().repeat(1, batch_size, 1)
        return lstm_h, lstm_c

    def reset_parameters(self):
        for p in self.lstm.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.num_inputs +  self.num_hidden))
                nn.init.uniform_(p, -stdev, stdev)


    def forward(self, x):
        x = x.unsqueeze(0)
        x, self.previous_state = self.lstm(x, self.previous_state)
        x = x.squeeze(0)
        x = F.sigmoid(self.fc(x))
        return x
       
    
    def init_sequence(self, batch_size):
        """Initializing the state."""
        self.batch_size = batch_size
        self.previous_state = self.create_new_state(batch_size)
