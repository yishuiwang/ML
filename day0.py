from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
import jieba
import pandas as pd


def datasets_demo():
    # 获取数据集
    iris = load_iris()
    # print("鸢尾花数据集：\n", iris)
    # print("查看数据集描述：\n", iris["DESCR"])
    # print("查看特征值的名字：\n", iris.feature_names)
    # print("查看特征值：\n", iris.data, iris.data.shape)

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("训练集的特征值：\n", x_train, x_train.shape)
    print("测试集的特征值：\n", x_test, x_test.shape)

    return None

def dict_demo():
    """
    字典特征抽取: DictVectorizer
    """
    data = [
        {'city': '北京', 'temperature': 20},
        {'city': '上海', 'temperature': 30},
        {'city': '广州', 'temperature': 25},
    ]

    transfer = DictVectorizer(sparse=False)

    data_new = transfer.fit_transform(data)
    # one-hot编码
    print("data_new:\n", data_new)
    print("特征名字：\n", transfer.get_feature_names_out())

    return None

def count_demo():
    """
    文本特征抽取: CountVectorizer
    """

    data = [
        "life is short, i like python",
        "life is too long, i dislike python"
    ]

    transfer = CountVectorizer()

    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())

    return None

def cut_word(text):
    """
    中文分词
    """
    return " ".join(list(jieba.cut(text)))

def count_chinese_demo2():
    """
    中文文本特征抽取, jieba分词
    """

    data = [
        "今天天气真好，我想出去打球",
        "今天我在家学习，不想出去"
    ]

    data_new = []
    for sentence in data:
        data_new.append(cut_word(sentence))
    
    transfer = CountVectorizer()

    data_final = transfer.fit_transform(data_new)
    print("data_new:\n", data_final.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())

    return None

def minmax_demo():
    """
    归一化 使用MinMaxScaler将原始数据映射到[0, 1]区间
    """
    # 1. 使用pd读取数据
    data = pd.read_csv("dating.txt")
    print(data)

    # 2. 实例化一个转换器类
    transfer = MinMaxScaler(feature_range=(0, 1))

    # 3. 调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)


    return None

def stand_demo():
    """
    标准化 使用StandardScaler将原始数据映射到均值为0，方差为1的区间
    """

    data = pd.read_csv("dating.txt")
    print("data:\n", data)

    transfer = StandardScaler()

    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)

    return None


    return None

if __name__ == "__main__":
    # datasets_demo()
    # dict_demo()
    # count_demo()
    # text = "今天天气真好，我想出去打球"
    # text = cut_word(text)
    # print(text)
    # count_chinese_demo2()
    minmax_demo()
