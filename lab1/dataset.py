from nltk import word_tokenize
import os
import json
from tqdm import tqdm
import random
import email.charset
from email import policy, message_from_binary_file
from encodings.aliases import aliases
from multiprocessing import Pool

email.charset.ALIASES.update(dict(aliases))
email.charset.ALIASES.update({
    'default': 'utf-8',
    'x-unknown': 'utf-8',
    '%charset': 'utf-8',
    '': 'utf-8'
})


def extract_sender(msg):
    # return ''
    try:
        return msg['From'].split('@')[-1].split('>')[0]
    except:
        return ''

def extract_body(msg):
    content = ''
    if msg.is_multipart():
        # print('is_multipart')
        for part in msg.get_payload():
            is_txt = part.get_content_type() == 'text/plain'
            if not content or is_txt:
                content += extract_body(part)
            if is_txt: # 后面的似乎就是重复的
                break
    elif msg.get_content_type().startswith("text/"): # text/plain, text/html
        charset = msg.get_param('charset', 'utf-8').lower()
        charset = email.charset.ALIASES.get(charset, charset)
        msg.set_param('charset', charset)
        try:
            content =  msg.get_content()
        except AssertionError as e:
            print('Parsing failed.')
            print(e)
        except LookupError as fk:
            #直接忽略
            pass
            # print('unkonw encoding')
            # print('lookupError = ', fk)
            # msg.set_param('charset', 'utf-8')
            # print('=== <UNKOWN ENCODING POSSIBLY SPAM> ===')
            # content = msg.get_content()
    # else:
    #      # content = msg.get_content()
        #  raise Exception(f"unknown type {msg.get_content()}")
    return content

def filter(text):
    if text:
        text = text.encode('ascii', "ignore").decode('utf-8')
    return text

def tokenize(path):
    tokens = []
    msg = message_from_binary_file(open(path, "rb"), policy=policy.default)
    sender = filter(extract_sender(msg))
    body = filter(extract_body(msg))  # TODO: 去停用词, 无意义的标点符号(如逗号)
    if sender:
        tokens.append(sender)
    if body:
        tokens += word_tokenize(body)
    return tokens

def process_one(data):
    root_path = data['root_path']
    index = data['index']
    label, sample_name = index.split()
    sample_name = sample_name[3:]
    sample_path = os.path.join(root_path, sample_name)
    sample_data = tokenize(sample_path)
    return {"label": label, "data": sample_data}


class Email:
    def __init__(self, path, ratio=1):
        with open(path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        if ratio == 1:
            self.dataset = dataset
        else:
            random.shuffle(dataset)
            size = len(dataset)
            sample_size = int(size * ratio)
            self.dataset = dataset[: sample_size]
    
    def __len__(self):
        return len(self.dataset)

    def get_label_freq(self):
        result = {'spam': 0, 'ham': 0}
        for sample in self.dataset:
            result[sample['label']] += 1
        return result
    
    def transfer(self):
        result = {'spam': [], 'ham': []}
        for sample in self.dataset:
            result[sample['label']] += sample['data']
        return result

    @staticmethod
    def preprocess(root_path, output_path):
        '''
        root_path: trec06p原始数据的路径
        output_path: 经过读取, 分词后，得到一个list of dict, 导出到json中
                        dict的格式是
                        {
                            "label": "spam",
                            "data": [word1, word2, ...]
                        }
        '''
        print('init dataset')
        index_path = os.path.join(root_path, 'label/index')
        with open(index_path, 'r', encoding='utf8') as f:
            indexes = f.readlines()

        input = [{"root_path": root_path, "index": index} for index in indexes]
        data = []
        with Pool(os.cpu_count()) as pool:
            iter = pool.imap(process_one, input)
            for i in tqdm(iter):
                data.append(i)
            
        print('total size = ', len(data))
        with open(output_path, 'w', encoding='utf8') as f:
            json.dump(data, f, indent=4, separators=[',', ':'])
        # else:
        #     print('find file, skip preprocessing')
    
    @staticmethod
    def split(path, ratio):
        '''
        按比例切分训练集测试集
        '''
        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)
        random.shuffle(data)
        size = len(data)
        test_size = int(size * ratio)
        train_data = data[:-test_size]
        test_data = data[-test_size:]
        print('train size = ', len(train_data))
        print('test size = ', len(test_data))
        with open(f"data/train.json", 'w', encoding='utf8') as f:
            json.dump(train_data, f, indent=4, separators=[',', ':'])
        with open(f"data/test.json", 'w', encoding='utf8') as f:
            json.dump(test_data, f, indent=4, separators=[',', ':'])
        print('finish dataset split')
    
    @staticmethod
    def split_kfold(path, k):
        '''
        kfold切分训练集测试集
        '''
        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)
        random.shuffle(data)
        size = len(data)
        ratio = 1 / k
        block_size = int(size * ratio)
        base_path = f'data/{k}_fold'
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        for i in range(k):
            start = i * block_size
            end = start + block_size
            test_data = data[start: end]
            train_data = data[:start] + data[end:]

            print(f'{i}_fold train size = ', len(train_data))
            print(f'{i}_fold test size = ', len(test_data))
            with open(f"data/{k}_fold/{i}_train.json", 'w', encoding='utf8') as f:
                json.dump(train_data, f, indent=4, separators=[',', ':'])
            with open(f"data/{k}_fold/{i}_test.json", 'w', encoding='utf8') as f:
                json.dump(test_data, f, indent=4, separators=[',', ':'])
        print('finish dataset split')


if __name__ == '__main__':
    random.seed(2021)
    path = 'data/ori_data_w_sender_suffix_only.json'
    # Email.preprocess('data/trec06p', path) # 由于multiprocess不保序, 建议直接使用预处理好的文件。
    Email.split(path, 0.1)
    # Email.split_kfold(path, 5)