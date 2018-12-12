# -*- coding: UTF-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class Pointer_network(nn.Module):

    def __init__(self, input_dim, hidden_dim, device):
        super(Pointer_network, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.encoder = nn.LSTMCell(self.input_dim, self.hidden_dim)
        self.decoder = nn.LSTMCell(self.input_dim, self.hidden_dim)
        self.W1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, x):
        batch_size, seq_len = x.shape
        x = torch.from_numpy(x).float().to(self.device)
        h,c = torch.zeros([batch_size, self.hidden_dim]).to(self.device),torch.zeros([batch_size, self.hidden_dim]).to(self.device)
        encoder_x = []
        for i in range(seq_len):
            h, c = self.encoder(x[:,i].unsqueeze(1), (h, c))
            encoder_x.append(h.unsqueeze(1))
        encoder_x = torch.cat(encoder_x, 1)
        decoder_input = torch.zeros(batch_size,self.input_dim).to(self.device)
        output =[]
        h, c = self.decoder(decoder_input, (h,c))
        runner = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).long().to(self.device)
        mask = torch.ones(batch_size, seq_len).to(self.device)
        We = self.W1(encoder_x)
        for _ in range(seq_len):
            Wd = self.W2(h).unsqueeze(1).repeat(1, self.hidden_dim, 1)
            u_i = self.v(torch.tanh(We+Wd))
            a_i = torch.softmax(u_i, dim=1)
            output.append(a_i.unsqueeze(1))
            pointer = (runner == torch.argmax(a_i * mask.unsqueeze(2),1)).float()
            mask = mask * (1-pointer)
            decoder_input = x[pointer.byte()].view(batch_size, self.input_dim)
            h, c = self.decoder(decoder_input, (h,c))
        output = torch.cat(output, dim=1).squeeze(3)
        return output