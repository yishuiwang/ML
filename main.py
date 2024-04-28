import numpy as np
import torch
import os
import tqdm
import config
from processor import RaceProcessor, download_dataset, select_field, simple_accuracy
from transformers import AlbertTokenizer

from transformers import *
# from transformers.modeling_albert import *
# from transformers.modeling_bert import *
from torch.utils.data import RandomSampler, DataLoader,TensorDataset
from torch.utils.data import SequentialSampler

from transformers.modeling_albert import AlbertForMultipleChoice


def transfer_to_features(
        examples: list,
        label_list: list,
        max_length: int,
        tokenizer,
)->list:
    
    lab_map = {label: i for i, label in enumerate(label_list)}
    features = []

    for example_index, example in enumerate(tqdm.tqdm(examples, desc="Converting examples to features")):
        # 将问题和文章分词
        context_tokens = tokenizer.tokenize(example.context)
        start_ending_tokens = tokenizer.tokenize(example.question)
        # 使用预训练的语言模型（如BERT、RoBERTa等）进行文本处理时，输入的序列长度通常是固定的，
        # 但是不同的句子长度可能会有所不同。因此，为了对不同长度的输入序列进行处理，需要使用填充（padding）或截断（truncation）等技术，
        # 使得所有输入序列的长度保持一致。
        # 将context_tokens和ending_tokens截断到最大长度
        def _truncate_seq_pair(tokens_a, tokens_b, max_length):
            while True:
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_length:
                    break
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()

        choices_features = []
        for ending_index, ending in enumerate(example.endings):
            # 将问题和选项拼接起来
            ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
            _truncate_seq_pair(context_tokens, ending_tokens, max_length - 3)
            # 添加特殊标记符号，将问题、文章、选项拼接起来
            # CLS classification 在文本开始处添加[CLS]标记符号，SEP separate 在每个文本之间添加[SEP]标记符号
            tokens = ["[CLS]"] + context_tokens + ["[SEP]"] + ending_tokens + ["[SEP]"]
            # input_ids是将每个单词转换为一个ID
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # 使用attention_mask 用于告知模型哪些部分是填充的、哪些部分是真实的文本内容的技术
            attention_mask = [1] * len(input_ids)
            # token type ids 用于区分单词属于那个句子
            token_type_ids = [0] * (1 + len(context_tokens) + 1) + [1] * (len(ending_tokens) + 1)

            padding_length = max_length - len(input_ids)
            input_ids += [0] * padding_length
            attention_mask += [0] * padding_length
            token_type_ids += [0] * padding_length
            choices_features.append((input_ids, attention_mask, token_type_ids))
        label = lab_map[example.label]
        features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))

    return features



def load_dataset(traning):
    cached_features_file = config.train_features_file
    if traning:
        cached_features_file = config.train_features_file
    else:
        cached_features_file = config.test_features_file
    # 1. 获取features
    if os.path.exists(cached_features_file):
        print("Loading features from cached file:", cached_features_file)
        features = torch.load(cached_features_file)
        # 确认特征加载成功
        print("Number of training examples:", len(features))
    else:
        # Raceprocessor 
        process = RaceProcessor()

        # 获取训练数据 返回的是InputExample对象
        # InputExample对象包含了问题、文章、选项、答案等信息
        if traning:
            train_examples = process.get_train_examples(config.data_dir)
        else:
            train_examples = process.get_test_examples(config.data_dir)
        print("Train examples: ", len(train_examples))

        # 获取训练标签
        label_list = ["0", "1", "2", "3"]

        # 使用albert tokenizer
        tokenizer = AlbertTokenizer.from_pretrained(config.pretrained_model)

        # 将数据转换为特征
        # features是一个列表，列表中的每个元素是一个InputFeatures对象
        # InputFeatures对象包含了问题、文章、选项、答案等信息
        features = transfer_to_features(train_examples, label_list, 512, tokenizer)

        # 保存特征
        print("Saving features into cached file ", cached_features_file)
        torch.save(features, cached_features_file)

    # 2. 将特征转换为张量
    print("Converting features to tensors")
    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_attention_mask = torch.tensor(select_field(features, 'attention_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, 'token_type_ids'), dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
    # all_attention_mask = torch.tensor([f.choices_features[0]['attention_mask'] for f in features], dtype=torch.long)
    # all_segment_ids = torch.tensor([f.choices_features[0]['token_type_ids'] for f in features], dtype=torch.long)
    # all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)

    print("all_input_ids: ", all_input_ids.shape)
    print("all_attention_mask: ", all_attention_mask.shape)
    print("all_segment_ids: ", all_segment_ids.shape)
    print("all_label_ids: ", all_label_ids.shape)

    # 3. 创建数据集
    print("Creating TensorDataset")
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_segment_ids, all_label_ids)
    
    return dataset

def train(model, dataset, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    batch_size = 4
    train_sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)
    
    model.zero_grad()

    total_acc, total_count = 0, 0

    
    train_steps = len(dataset) // batch_size

    print("Total steps:", train_steps)

    for epoch in range(2):
        print("Epoch", epoch)
        for step, batch in enumerate(dataloader):
            # 将数据转移到GPU上
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            token_type_ids = batch[2].to(device)
            label = batch[3].to(device)
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids, 'labels': label}

            # 将文章、问题、选项拼接起来得到X，X本身乘以三个矩阵W 得到Q,K,V
            # Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V
            # 的到的结果再乘以一个矩阵W得到最终的输出
            # 输出的第一个元素是loss，第二个元素是logits预测值

            # AlbertForMultipleChoice
            # parameters:
            # input_ids:
                # Indices of input sequence tokens in the vocabulary.
            # attention_mask: 
                # Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]:
                # 1 for tokens that are NOT MASKED, 0 for MASKED tokens.
            # token_type_ids
                # Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]:
                # 0 corresponds to a sentence A token, 1 corresponds to a sentence B token
            # position_ids
                # Indices of positions of each input sequence tokens in the position embeddings.
            # labels
                # Labels for computing the multiple choice classification loss. Indices should be in [0, ..., num_choices] where num_choices is the size of the second dimension of the input tensors.
            # Return:
            # loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided):
            # classification_scores (torch.FloatTensor of shape (batch_size, num_choices)):
            outputs = model(**inputs)
            loss = outputs[0]
            preds = outputs[1].detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
            # print("preds", preds)
            # print("out_label_ids", out_label_ids)
            # preds [[ 0.37252483  0.35667765  0.37862685  0.36976168]
            #  [-0.03206023 -0.03527563 -0.04461297 -0.02163452]
            #  [ 0.1736255   0.17404702  0.16252017  0.16390753]
            #  [ 0.32777974  0.33433586  0.33356354  0.33603436]]
            # out_label_ids [1 1 0 1]
            
            # 返回每行中最大值的索引
            preds = np.argmax(preds, axis=1)
            # print("preds", preds)
            # preds [2 3 1 3]
            
            # 比较预测值和标签值是否相等，mean()计算为True的比例
            acc = simple_accuracy(preds, out_label_ids)

            total_acc += acc * input_ids.size(0)
            total_count += input_ids.size(0)

            loss.backward() # 反向传播
            optimizer.step()    # 更新参数
            model.zero_grad()   # 梯度归零

            if step % 100 == 0:
                # 保存模型
                print("Saving model")   
                torch.save(model.state_dict(), config.model_file)


            print("Step", step, "Loss", loss.item(), "Accuracy", acc)
   
    print("Total accuracy", total_acc / total_count)
    print("Finished training")

    return 0

def evaluate(dataset, device, model):
    print("Evaluating")

    # model = AlbertForMultipleChoice.from_pretrained("albert-base-v2")
    # model.to(device)
    # model.load_state_dict(torch.load("model.pth"))


    batch_size = 2
    eval_sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size)

    train_steps = len(dataset) // batch_size

    print("Total steps:", train_steps)
    total_acc, total_count = 0, 0
    for step, batch in enumerate(dataloader):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        token_type_ids = batch[2].to(device)
        label = batch[3].to(device)
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids, 'labels': label}

        outputs = model(**inputs)
        loss = outputs[0]
        preds = outputs[1].detach().cpu().numpy()
        out_label_ids = inputs['labels'].detach().cpu().numpy()

        preds = np.argmax(preds, axis=1)
        acc = simple_accuracy(preds, out_label_ids)
        total_acc += acc * input_ids.size(0)
        total_count += input_ids.size(0)

        print("Step", step, "Loss", loss.item(), "Accuracy", acc)

    print("Total accuracy", total_acc / total_count)

    return 0

def init():
    if not os.path.exists(config.data_dir):
        download_dataset()
    return 0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))


    # 1. 训练模型
    model = AlbertForMultipleChoice.from_pretrained(config.pretrained_model)
    model.to(device)
    # dataset = load_dataset(traning=True)

    # train(model, dataset, device)

    # 2. 评估模型
    model.load_state_dict(torch.load(config.model_file))
    dataset = load_dataset(traning=False)
    evaluate(dataset, device, model)

    return 0

if __name__ == "__main__":
    main()
