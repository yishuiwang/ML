import torch 
import torch.nn as nn

# 实例化rnn对象
# 第一个参数: input_size: 输入张量x的维度
# 第二个参数: hidden_size: 隐藏层的维度（神经元的个数）
# 第三个参数: num_layers: rnn的层数
rnn=nn.RNN(5,6,1)

# 定义一个输入张量x
# 第一个维度: 序列长度
# 第二个维度: batch_size  (批次样本数)
# 第三个维度: 输入张量的维度
input=torch.randn(1,3,5)

# 定义一个初始的隐藏层张量h0
# 第一个维度: 隐藏层的层数
# 第二个维度: batch_size  (批次样本数)
# 第三个维度: 隐藏层的维度
h0=torch.randn(1,3,6)

# 将输入张量x和初始的隐藏层张量h0传入rnn对象中
output,hn=rnn(input,h0)

# print("output: ",output)
# print("hn: ",hn)

# 实例LSTM对象
# 第一个参数: input_size: 输入张量x的维度
# 第二个参数: hidden_size: 隐藏层的维度（神经元的个数）
# 第三个参数: num_layers: 隐藏层的层数
lstm=nn.LSTM(5,6,2)

# 定义一个输入张量x
# 第一个参数: 序列长度
# 第二个参数: batch_size  (批次样本数)
# 第三个参数: 输入张量的维度
input=torch.randn(1,3,5)

# 定义一个初始的隐藏层张量h0
# 第一个参数: 隐藏层的层数
# 第二个参数: batch_size  (批次样本数)
# 第三个参数: 隐藏层的维度
h0=torch.randn(2,3,6)
c0=torch.randn(2,3,6)

# 将输入张量x和初始的隐藏层张量h0、c0传入lstm对象中
output,(hn,cn)=lstm(input,(h0,c0))
print("output: ",output)
print("hn: ",hn)
print("cn: ",cn)
print("output.shape: ",output.shape)
print("hn.shape: ",hn.shape)
print("cn.shape: ",cn.shape)