import glob
import json
import numpy as np
import torch
import os
import tqdm
from transformers import AlbertTokenizer

from transformers import *
from transformers.modeling_albert import *
from transformers.modeling_bert import *
from torch.utils.data import RandomSampler, DataLoader,TensorDataset


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def convert_examples_to_features(
        examples: list,
        label_list: list,
        max_length: int,
        tokenizer,
)->list:
    
    lab_map = {label: i for i, label in enumerate(label_list)}
    features = []

    for example_index, example in enumerate(tqdm.tqdm(examples, desc="Converting examples to features")):
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
            ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
            _truncate_seq_pair(context_tokens, ending_tokens, max_length - 3)
            tokens = ["[CLS]"] + context_tokens + ["[SEP]"] + ending_tokens + ["[SEP]"]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # 使用attention_mask 用于告知模型哪些部分是填充的、哪些部分是真实的文本内容的技术
            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * (1 + len(context_tokens) + 1) + [1] * (len(ending_tokens) + 1)
            padding_length = max_length - len(input_ids)
            input_ids += [0] * padding_length
            attention_mask += [0] * padding_length
            token_type_ids += [0] * padding_length
            choices_features.append((input_ids, attention_mask, token_type_ids))
        label = lab_map[example.label]
        features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))

    return features


class InputExample:
    """A single training/test example for multiple choice."""

    def __init__(self, example_id, question, context, endings, label=None):
        """Constructs a InputExample."""
        self.example_id = example_id
        self.question = question
        self.context = context
        self.endings = endings
        self.label = label


class InputFeatures:
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids, # 将每个单词转换为一个ID 
                # 由整数构成的列表，表示输入文本的token索引 eg {'hello': 0, 'world': 1}，句子"hello world"将被转换为[0, 1]。
                'attention_mask': attention_mask, # 由0和1构成的列表，表示哪些token是填充的，哪些是真实的文本内容。eg [1, 1]。
                'token_type_ids': token_type_ids  # 标记每个单词属于哪个句子
            }
            for input_ids, attention_mask, token_type_ids in choices_features
        ]
        self.label = label

class RaceProcessor:
    """Processor for the Race data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        high_dir = os.path.join(data_dir, "train/high")
        middle_dir = os.path.join(data_dir, "train/middle")
        print("get_train_examples")
        high_examples = self._read_data(high_dir,"train")
        middle_examples = self._read_data(middle_dir,"train")
        return high_examples + middle_examples
    
    def _read_data(self, input_dir, set_type):
        """Read a json file into a list of examples."""
        examples = []
        # 读取文件夹下所有的txt文件
        files = glob.glob(input_dir + "/*txt")
        # 使用tqdm显示进度条
        for file in tqdm.tqdm(files,desc="Reading data"):
            with open(file, "r", encoding="utf-8") as reader:
                data_raw = json.load(reader)
                article = data_raw['article']
                for i in range(len(data_raw['answers'])):
                    truth = str(ord(data_raw['answers'][i]) - ord('A'))
                    question = data_raw['questions'][i]
                    options = data_raw['options'][i]
                    examples.append(
                        InputExample(
                            example_id = data_raw['id'],
                            question=question,
                            context=article,
                            endings=[options[0], options[1], options[2], options[3]],
                            label=truth,
                        )
                    )

        return examples

def load_dataset():
    cached_features_file = "features.pt"
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
        train_examples = process.get_train_examples("RACE")
        print("Train examples: ", len(train_examples))

        # 获取训练标签
        label_list = ["0", "1", "2", "3"]

        # 使用albert tokenizer
        tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

        # 将数据转换为特征
        # features是一个列表，列表中的每个元素是一个InputFeatures对象
        # InputFeatures对象包含了问题、文章、选项、答案等信息
        features = convert_examples_to_features(train_examples, label_list, 512, tokenizer)

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    batch_size = 4
    train_sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)
    
    model.zero_grad()

    for epoch in range(2):
        print("Epoch", epoch)
        for step, batch in enumerate(dataloader):
            # 将数据转移到GPU上
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            token_type_ids = batch[2].to(device)
            label = batch[3].to(device)
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids, 'labels': label}

            outputs = model(**inputs)   # 前向传播
            loss = outputs[0]
            logits = outputs[1]

            preds = None
            if preds is None: 
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:  
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
            preds = np.argmax(preds, axis=1)
            acc = simple_accuracy(preds, out_label_ids)

            loss.backward() # 反向传播
            optimizer.step()    # 更新参数
            model.zero_grad()   # 梯度归零

            print("Step", step, "Loss", loss.item(), "Accuracy", acc)
   
    return 0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))
    model = AlbertForMultipleChoice.from_pretrained("albert-base-v2")

    model.to(device)
    # load_dataset()
    dataset = load_dataset()

    train(model, dataset,device)

    return 0


if __name__ == "__main__":
    main()