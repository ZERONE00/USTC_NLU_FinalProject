'''
Author: wxin
Date: 2020-08-09 09:43:09
LastEditTime: 2020-08-09 10:30:38
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /lab/src/dataset.py
'''
import torch
import torch.utils.data

class dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]