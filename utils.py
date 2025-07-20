import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import scale

def multiomics_data():

    #se_fea
    seHIN = np.genfromtxt("feature/se_hin_diease.csv", delimiter=',', skip_header=1, dtype=np.dtype(str))
    sese = np.genfromtxt("feature/se_self.csv", delimiter=',', skip_header=1, dtype=np.dtype(str))
    seHIN = np.array(seHIN)
    seHIN = scale(np.array(seHIN[:, 1:], dtype=float))
    sese = np.array(sese)
    sese = scale(np.array(sese[:, 1:], dtype=float))
    sesei_adj = get_adj_array("feature/Adjacency_matrix_se.txt")


    dHIN = np.genfromtxt("feature/drug_hin_diease.csv", delimiter=',', skip_header=1, dtype=np.dtype(str))
    dse = np.genfromtxt("feature/drug_self.csv", delimiter=',',skip_header=1,dtype=np.dtype(str))
    dHIN = np.array(dHIN)
    dHIN = scale(np.array(dHIN[:, 1:], dtype=float))
    dse = np.array(dse)
    dse = scale(np.array(dse[:, 1:], dtype=float))
    ddi_adj = get_adj_array("feature/Adjacency_matrix_drug.txt")

    # 加载label
    labellist = []
    with open('data/label_end.txt', 'r') as file:
        lines = file.readlines()
    for line in lines:
        line = line.strip()  # 移除行首和行尾的空白字符
        elements = line.split(" ")  # 使用空格分隔元素
        processed_elements = [int(elements[1]), int(elements[0]), int(elements[2])]  # 互换位置
        labellist.append(processed_elements)
    labellist = torch.Tensor(labellist)
    print("drug protein lable:", labellist.shape)


    se_HIN, se_se, se_adj = torch.FloatTensor(seHIN), torch.FloatTensor(sese), torch.FloatTensor(sesei_adj)
    drug_HIN, drug_PC, drug_adj = torch.FloatTensor(dHIN), torch.FloatTensor(dse), torch.FloatTensor(ddi_adj)
    return se_HIN, se_se, se_adj, drug_HIN, drug_PC, drug_adj, labellist


def get_adj_array(file_path):
    # 读取txt文件
    file_path = file_path  # 替换为你的文件路径
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 提取邻接矩阵信息
    adj_matrix = []
    for line in lines:
        row = [float(x) for x in line.strip().split()]  # 假设邻接矩阵中的元素以空格分隔
        adj_matrix.append(row)

    # 将邻接矩阵转换为NumPy数组
    adj_array = np.array(adj_matrix)
    return adj_array