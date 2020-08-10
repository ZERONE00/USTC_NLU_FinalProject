'''
Author: wxin
Date: 2020-08-09 10:04:57
LastEditTime: 2020-08-09 10:11:27
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /lab/src/config.py
'''
import torch

class Config(object):

    """配置参数"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        self.dropout = 0.5                                              
        self.num_classes = 3                         
        self.num_epochs = 20                                           
        self.batch_size = 128                                                                           
        self.learning_rate = 1e-3                                       
        self.embed = 300
        self.filter_sizes = (2, 3, 4)   # 卷积核尺寸
        self.num_filters = 256          # 卷积核数量(channels数)
