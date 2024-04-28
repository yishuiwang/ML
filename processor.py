import json
import os
import glob
import tqdm
import requests
import tarfile

class InputExample:
    """A single training/test example for multiple choice."""

    def __init__(self, example_id, question, context, endings, label=None):
        """Constructs a InputExample."""
        self.example_id = example_id
        self.question = question
        self.context = context
        self.endings = endings  # 代表四个选项
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
    
    def get_dev_examples(self, data_dir):
        """See base class."""
        high_dir = os.path.join(data_dir, "dev/high")
        middle_dir = os.path.join(data_dir, "dev/middle")
        high_examples = self._read_data(high_dir,"dev")
        middle_examples = self._read_data(middle_dir,"dev")
        return high_examples + middle_examples
    
    def get_test_examples(self, data_dir):
        """See base class."""
        high_dir = os.path.join(data_dir, "test/high")
        middle_dir = os.path.join(data_dir, "test/middle")
        high_examples = self._read_data(high_dir,"test")
        middle_examples = self._read_data(middle_dir,"test")
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

def download_dataset():
    print("Downloading the dataset...")
    url = "http://www.cs.cmu.edu/~glai1/data/race/RACE.tar.gz"
    target_path = 'RACE.tar.gz'

    response = requests.get(url, stream=True)
    file_size = int(response.headers.get('Content-Length', 0))
    progress = tqdm(response.iter_content(1024), f'Downloading {target_path}', total=file_size, unit='B', unit_scale=True, unit_divisor=1024)

    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            for data in progress.iterable:
                f.write(data)
                progress.update(len(data))

        if target_path.endswith("tar.gz"):
            tar = tarfile.open(target_path, "r:gz")
            tar.extractall()
            tar.close()
        elif target_path.endswith("tar"):
            tar = tarfile.open(target_path, "r:")
            tar.extractall()
            tar.close()

        os.remove(target_path)
    else:
        print("Failed to download the dataset. HTTP response code: ", response.status_code)