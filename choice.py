import glob
import json
import torch
import os
import tqdm
from transformers import AlbertTokenizer


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
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
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
    # Raceprocessor 
    process = RaceProcessor()

    # 获取训练数据
    train_examples = process.get_train_examples("RACE")
    print("Train examples: ", len(train_examples))

    # 获取训练标签
    label_list = ["0", "1", "2", "3"]

    # 使用albert tokenizer
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

    # 将数据转换为特征
    features = convert_examples_to_features(train_examples, label_list, 512, tokenizer)

    # 保存特征
    cached_features_file = "features.pt"
    print("Saving features into cached file ", cached_features_file)
    torch.save(features, cached_features_file)

    return 0

def load_features():
    cached_features_file = "features.pt"
    print("Loading features from cached file ", cached_features_file)
    features = torch.load(cached_features_file)
    # 确认特征加载成功
    print("Number of training examples:", len(features))
    return features

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))
    # load_dataset()
    features = load_features()


    return 0


if __name__ == "__main__":
    main()