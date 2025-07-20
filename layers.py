import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

#自注意力机制
class selfattention(nn.Module):
    def __init__(self, sample_size, d_k, d_v):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.query = nn.Linear(sample_size, d_k)
        self.key = nn.Linear(sample_size, d_k)
        self.value = nn.Linear(sample_size, d_v)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        att = torch.matmul(q, k.transpose(0, 1)) / np.sqrt(self.d_k)
        att = torch.softmax(att, dim=1)
        output = torch.matmul(att, v)
        return output




class GraphSAGELayer(nn.Module):
    def __init__(self, in_features, out_features, agg_method="mean"):
        super(GraphSAGELayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.agg_method = agg_method

        # 线性层和批量归一化层
        self.linear = nn.Linear(in_features * 2, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)

        # 池化层
        self.pool = nn.AdaptiveAvgPool1d(1)

        # LSTM层
        self.lstm = nn.LSTM(in_features, out_features, batch_first=True)

    def forward(self, x, adj):
        if self.agg_method == "mean":
            out = torch.spmm(adj, x)
        elif self.agg_method == "sum":
            out = torch.spmm(adj, x)
        elif self.agg_method == "max":
            out = torch.spmm(adj, x)
            #out = torch.max(out, dim=1, keepdim=True)[0]
            out, _ = torch.max(out, dim=1, keepdim=True)
            #out = out.repeat(1, x.shape[1])  # 扩展维度
            out = out.expand(-1, x.shape[1])  # 让维度匹配 x
        elif self.agg_method == "pool":
            out = torch.spmm(adj, x)
            out = self.pool(out.unsqueeze(0)).squeeze(0)
        elif self.agg_method == "lstm":
            out = torch.spmm(adj, x)
            # out, _ = self.lstm(out.unsqueeze(0))
            # out = out.squeeze(0)
            out = out.unsqueeze(1)  # 增加 seq_len 维度
            out, _ = self.lstm(out)
            out = out.squeeze(1)
        else:
            raise NotImplementedError

        # 将节点自身的特征与邻居节点的特征进行拼接
        out = torch.cat([x, out], dim=1)

        # 线性变换和批量归一化
        out = self.linear(out)
        out = self.bn(out)
        return out