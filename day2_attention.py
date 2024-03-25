import torch
import torch.nn as nn
import torch.nn.functional as F

class Attn(nn.Module):
    def __init__(self,query_size,key_size,value_size1,value_size2,output_size):
        # query_size: Q的最后一个维度
        # key_size: K的最后一个维度
        # V的尺寸表示(1,value_size1,value_size2)
        # output_size: 输出的最后一个维度
        super(Attn, self).__init__()

        self.query_size = query_size
        self.key_size = key_size
        self.value_size1 = value_size1
        self.value_size2 = value_size2
        self.output_size = output_size

        # 初始化注意力机制实现中第一步的线性层
        self.attn = nn.Linear(self.query_size + self.key_size, value_size1)

        # 初始化注意力机制实现中第三步的线性层
        self.attn_combine = nn.Linear(self.query_size + self.value_size2, self.output_size)

    def forward(self, query, key, value):
        # 假定Q,K,V都是三维张量
        # 1.将Q,K进行纵轴拼接，然后通过线性层attn进行变换，最后使用softmax函数得到注意力权重
        attn_weights = F.softmax(
            self.attn(torch.cat((query[0], key[0]), 1)), dim=1)

        # 2.将注意力权重与V进行加权求和
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), value)

        output = torch.cat((query[0], attn_applied[0]), 1)
        # 3.将Q与加权后的V进行拼接，然后通过线性层attn_combine进行变换,再扩展成三维张量
        output = self.attn_combine(output).unsqueeze(0)

        return output, attn_weights
    

query_size = 32
key_size = 32
value_size1 = 32
value_size2 = 64
output_size =  64

attn = Attn(query_size,key_size,value_size1,value_size2,output_size)

query = torch.randn(1,1,query_size)
key = torch.randn(1,1,key_size)
value = torch.randn(1,value_size1,value_size2)

output,attn_weights = attn(query,key,value)
print(output)
print(attn_weights)
# 输出：
