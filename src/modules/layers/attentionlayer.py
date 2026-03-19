import torch.nn as nn
import torch.nn.functional as F
import torch

class SelfAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask

        '''建立三个线性层，用于计算multi-head的Q、K、V矩阵'''
        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x):

        b, t, e = x.size()
        # 从输入的张量（x）中取出第一、二、三个维度的大小（b, t, e），分别赋值给b, t, e。其中，b是批量大小，t是序列长度，e是嵌入维度

        h = self.heads
        # h代表注意力的头数

        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        dot = torch.bmm(queries, keys.transpose(1, 2))
        #对Q和K进行批量矩阵乘法（torch.bmm），得到点积注意力矩阵（dot），其形状为[bs * h, seq_len, seq_len]

        assert dot.size() == (b * h, t, t)
        # 声明一下点积dot的形状[bs * h, seq_len, seq_len]

        dot = F.softmax(dot, dim=2)

        out = torch.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        # 将自注意力输出第二个和第三个维度转置，并将头的维度移到最后一个维度，得到形状为[bs, seq_len, h * embed_dim]的张量

        return self.unifyheads(out)