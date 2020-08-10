'''
Author: wxin
Date: 2020-08-09 09:36:32
LastEditTime: 2020-08-09 10:32:04
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /lab/src/model.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, embedding_weights, config):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=False)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out