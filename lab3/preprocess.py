import json
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import os

def preprocess(input_path, output_path):
    # 读取数据
    with open(input_path, 'r') as f:
        ori_data = f.readlines()
        ori_data = [i.strip().split('\t') for i in ori_data]
    name = ori_data[0]
    data = [dict(zip(name, i)) for i in ori_data[1:]]

    # split
    random.shuffle(data)
    size = len(data)
    train_size = int(0.9 * size)

    train_set = data[: train_size]
    test_set = data[train_size: ]

    print('train size = ', len(train_set))
    print('test size = ', len(test_set))
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for data, split in zip([train_set, test_set], ['train', 'test']):  
        with open(f"{output_path}/{split}.json", 'w') as f:
            json.dump(data, f, indent=4, separators=[',', ':'])

if __name__ == '__main__':
    preprocess("exp3-reviews.csv", 'data')