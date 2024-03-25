import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from torch.utils.data.dataset import random_split   # 划分数据集

# 指定batch_size
batch_size = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TextSentiment(nn.Module):
    '''文本分类模型'''

    def __init__(self, vocab_size, embed_dim, num_class):
        # vocab_size为词典大小，embed_dim为词向量维度，num_class为分类类别数
        super(TextSentiment, self).__init__()
        # 实例化embedding层，vocab_size为词典大小，embed_dim为词向量维度，sparse=True表示梯度更新时使用稀疏更新
        self.embedding = nn.Embedding(vocab_size, embed_dim,sparse=True)
        # 实例化卷积层
        self.fc = nn.Linear(embed_dim, num_class)

        self.init_weights()

    def init_weights(self):
        '''初始化权重'''
        initrange = 0.5
        # 各层的权重参数初始化为均匀分布
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        # 偏置初始化为0
        self.fc.bias.data.zero_()

    def forward(self, text):
        # text代表数字化映射后的张量

        # 对输入的文本进行词嵌入
        embedded = self.embedding(text)
        c = embedded.size(0) // batch_size
        embedded = embedded[:c * batch_size]

        # 平均池化的张量需要传入三维张量，因此需要对嵌入张量进行维度转换
        embedded = embedded.transpose(0, 1).unsqueeze(0)

        # 进行平均池化的操作
        # 池化的目的是为了减少特征的数量，减少计算量
        embedded = F.avg_pool1d(embedded, kernel_size=c)


def generate_batch(batch):

    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    text=torch.cat(text)

    return text, label




def train(train_data):

    tran_loss = 0
    tran_acc = 0

    data = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)

    for i ,(text, cls) in enumerate(data):
        # 梯度清零
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, cls)
        # 进行反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 计算损失
        loss += loss.item()
        # 计算准确率
        acc += (output.argmax(1) == cls).sum().item()


    # 进行整个轮次的优化器学习率调整
    scheduler.step()

    # 返回平均损失和准确率
    return tran_loss / len(train_data), tran_acc / len(train_data)


def valid(valid_data):

    valid_loss = 0
    valid_acc = 0

    data = DataLoader(valid_data, batch_size=batch_size, collate_fn=generate_batch)

    for i, (text, cls) in enumerate(data):
        # 不进行梯度更新
        with torch.no_grad():
            output = model(text)
            loss = criterion(output, cls)
            valid_loss += loss.item()
            valid_acc += (output.argmax(1) == cls).sum().item()

    return valid_loss / len(valid_data), valid_acc / len(valid_data)


def setup_datasets(ngrams=2, vocab_train=None, vocab_test=None, include_unk=False):

    train_csv_path = 'data/ag_news_csv/train.csv'
    test_csv_path = 'data/ag_news_csv/test.csv'

    if vocab_train is None:
        vocab_train = build_vocab_from_iterator(_csv_iterator(train_csv_path, ngrams))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")

    if vocab_test is None:
        vocab_test = build_vocab_from_iterator(_csv_iterator(test_csv_path, ngrams))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")

    train_data, train_labels = _create_data_from_iterator(
        vocab_train, _csv_iterator(train_csv_path, ngrams, yield_cls=True), include_unk)
    test_data, test_labels = _create_data_from_iterator(
        vocab_test, _csv_iterator(test_csv_path, ngrams, yield_cls=True), include_unk)

    if len(train_labels ^ test_labels) > 0:
        raise ValueError("Training and test labels don't match")
    return (TextClassificationDataset(vocab_train, train_data, train_labels),
            TextClassificationDataset(vocab_test, test_data, test_labels))


# 调用函数, 加载本地数据
train_dataset, test_dataset = setup_datasets()
print("train_dataset", train_dataset)















