import jieba

content="今天天气真好，我想出去打球"

# jieba分词
content_cut = jieba.cut(content)

# 转换成列表
content_list = list(content_cut)

# 列表转换成字符串
content_str = " ".join(content_list)


# hanlp分词
# import hanlp

# tokenizer = hanlp.load('CTB6_CONVSEG')

# tokenizer(content)

# 词性标注 动词 v 名词 n 形容词 a

# import jieba.posseg as pseg

# pseg.lcut(content)


# 文本预处理基本方法

# 文本张量化
# 1. one-hot编码
# 2. word2vec
# 3. word embedding

import torch
import json
from torch.utils.tensorboard import SummaryWriter

# 1. 实例化一个SummaryWriter
writer = SummaryWriter()

embedded = torch.rand(100, 50)

