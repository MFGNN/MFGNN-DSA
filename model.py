import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *

class MultiDeep(nn.Module):
    def __init__(self,nse, ndrug, nseHIN, nseSE, ndrugHIN, ndrugPC, nhid_GAT, nheads_GAT, nhid_MHSA, nheads_MHSA, alpha):

        super(MultiDeep, self).__init__()
        self.se_attentions1 = nn.ModuleList(
            [GraphSAGELayer(nseSE, nhid_GAT * nheads_GAT, agg_method="mean")])
        for i, att in enumerate(self.se_attentions1):
            self.add_module(f'Se_GraphSAGE1_{i}', att)

        self.se_MultiHead1 = [selfattention(nhid_GAT * nheads_GAT + nseHIN + nseSE, nhid_MHSA, nhid_GAT * nheads_GAT + nseHIN + nseSE) for _ in range(nheads_MHSA)]
        for i, attention in enumerate(self.se_MultiHead1):
            self.add_module('Self_Attention_Se1_{}'.format(i), attention)
        self.se_prolayer1 = nn.Linear(nhid_GAT * nheads_GAT, nhid_GAT * nheads_GAT, bias=False)
        self.se_LNlayer1 = nn.LayerNorm(nhid_GAT * nheads_GAT + nseHIN + nseSE)

        self.drug_attentions1 = nn.ModuleList([GraphSAGELayer(ndrugPC, nhid_GAT * nheads_GAT, agg_method="mean")])
        for i, att in enumerate(self.drug_attentions1):
            self.add_module(f'Drug_GraphSAGE1_{i}', att)

        self.drug_MultiHead1 = [selfattention(nhid_GAT * nheads_GAT + ndrugHIN + ndrugPC, nhid_MHSA, nhid_GAT * nheads_GAT + ndrugHIN + ndrugPC) for _ in range(nheads_MHSA)]
        for i, attention in enumerate(self.drug_MultiHead1):
            self.add_module('Self_Attention_Drug1_{}'.format(i), attention)
        self.drug_prolayer1 = nn.Linear(nhid_GAT * nheads_GAT, nhid_GAT * nheads_GAT, bias=False)
        self.drug_LNlayer1 = nn.LayerNorm(nhid_GAT * nheads_GAT + ndrugHIN + ndrugPC)

        self.FClayer1 = nn.Linear(nhid_GAT * nheads_GAT + ndrugHIN + ndrugPC + nhid_GAT * nheads_GAT + nseHIN + nseSE, (nhid_GAT * nheads_GAT + ndrugHIN + ndrugPC) * 2)
        self.FClayer2 = nn.Linear((nhid_GAT * nheads_GAT + ndrugHIN + ndrugPC) * 2, (nhid_GAT * nheads_GAT + ndrugHIN + ndrugPC) * 2)
        self.FClayer3 = nn.Linear((nhid_GAT * nheads_GAT + ndrugHIN + ndrugPC) * 2, 1)
        self.output = nn.Sigmoid()

    def forward(self, seHIN, seSE, se_adj, drugHIN, drug_PC, drug_adj, idx_se_drug, device):
        sex = torch.cat([att(seSE, se_adj) for att in self.se_attentions1], dim=1)
        sex = self.se_prolayer1(sex)
        semax = torch.cat([sex, seSE, seHIN],dim=1)
        temp = torch.zeros_like(semax)
        for selfatt in self.se_MultiHead1:
            temp = temp + selfatt(semax)
        sex = temp +semax
        sex = self.se_LNlayer1(sex)

        drugx = torch.cat([att(drug_PC, drug_adj) for att in self.drug_attentions1], dim=1)
        drugx = self.drug_prolayer1(drugx)
        drugmax = torch.cat([drugx, drug_PC, drugHIN],dim=1)
        temp = torch.zeros_like(drugmax)
        for selfatt in self.drug_MultiHead1:
            temp = temp + selfatt(drugmax)
        drugx = temp + drugmax
        drugx = self.drug_LNlayer1(drugx)


        se_drug_x = torch.cat((sex[idx_se_drug[:, 0]], drugx[idx_se_drug[:, 1]]), dim=1)
        se_drug_x = se_drug_x.to(device)
        se_drug_x = self.FClayer1(se_drug_x)
        se_drug_x = F.relu(se_drug_x)
        se_drug_x = self.FClayer2(se_drug_x)
        se_drug_x = F.relu(se_drug_x)
        se_drug_x = self.FClayer3(se_drug_x)
        se_drug_x = se_drug_x.squeeze(-1)
        return se_drug_x



