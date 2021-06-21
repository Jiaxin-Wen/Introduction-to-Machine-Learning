from sys import path
from dataset import Email
import json
from nltk import FreqDist
import math
import os
import numpy as np


class Trainer:
    
    def cal_metric(self, preds, labels):
        preds = np.array(preds)
        labels = np.array(labels)
        
        acc = np.mean(preds == labels)

        TP, FP, FN = 0, 0, 0
        for pred, label in zip(preds, labels):
            if pred == 1 and label == 1:
                TP += 1
            elif pred == 1 and label == 0:
                FP += 1
            elif pred == 0 and label == 1:
                FN += 1

        precision = TP / (TP + FP) if (TP + FP) else 0
        recall = TP / (TP + FN) if (TP + FN) else 0
        F1 = (2 * precision * recall) / (precision + recall)

        print('acc = ', acc)
        print('precision = ', precision)
        print('recall = ', recall)
        print('F1 = ', F1)

    def normalize(self, freq, cond_flag=False):
        if cond_flag:
            for cls in freq:
                total = sum(list(freq[cls].values()))
                for word in freq[cls]:
                    freq[cls][word] /= total
        else:
            total = sum(list(freq.values()))
            for k in freq:
                freq[k] /= total
        return freq
    
    def fit(self, data_path, save_path, sample_rate):
        dataset = Email(data_path, sample_rate)
        print('train len = ', len(dataset))
        label_freq = dataset.get_label_freq()
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # if os.listdir(save_path) and not overwrite: # 非空
        #     print(f'find param in {save_path}, skip training')
        # else:
        with open(os.path.join(save_path, "label_freq.json"), 'w', encoding='utf-8') as f:
            json.dump(label_freq, f)

        dataset = dataset.transfer() # 变成dict of list
        params = {'ham': [], 'spam': []}
        params['ham'] = dict(FreqDist(dataset['ham']))
        params['spam'] = dict(FreqDist(dataset['spam']))
        with open(os.path.join(save_path, "cond_freq.json"), 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=4, separators=[',', ': '])
        print(f"save params to: {save_path}")
        
    def test(self, data_path, param_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            test_dataset = json.load(f)
        with open(os.path.join(param_path, 'label_freq.json'), 'r', encoding='utf-8') as f:
            label_freq = json.load(f)
        with open(os.path.join(param_path, 'cond_freq.json'), 'r', encoding='utf-8') as f:
            cond_freq = json.load(f)

        label_freq = self.normalize(label_freq)
        cond_freq = self.normalize(cond_freq, cond_flag=True)    

        classes = ['ham', 'spam']  
        labels = []
        preds = []
        for sample in test_dataset:
            label, tokens = sample['label'], sample['data']
            temp_logits = []
            for cls in classes:
                prob = math.log(label_freq[cls])
                for token in tokens:
                    if token in cond_freq[cls]:
                        prob += math.log(cond_freq[cls][token])
                    else:
                        prob += math.log(1e-10)
                temp_logits.append(prob)
            pred = np.argmax(temp_logits)
            labels.append(classes.index(label))
            preds.append(pred)

        self.cal_metric(preds, labels)
                        
    def run(self, train_path, test_path, save_path, k=1, sample_rate=1):
        '''
        train_path: 训练集路径
        test_path: 测试集路径
        save_path: 参数保存的路径
        k: kfold
        sample_rate: 测试训练集size的影响
        '''
        if k == 1:
            self.fit(train_path, save_path, sample_rate)
            self.test(test_path, save_path)
        else:
            for i in range(k):
                tmp_train_path = os.path.join(train_path, f'{i}_train.json')
                tmp_test_path = os.path.join(test_path, f'{i}_test.json')
                tmp_save_path = os.path.join(save_path, f"{i}_param")
                if not os.path.exists(tmp_save_path):
                    os.mkdir(tmp_save_path)
                self.run(tmp_train_path, tmp_test_path, tmp_save_path, 1, sample_rate)
                print('='*100)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.run(train_path="data/train.json", test_path="data/test.json", save_path="data/param", sample_rate=1)
    # trainer.test('data/test.json', 'data/param')
    # trainer.run(train_path='data/5_fold', test_path='data/5_fold', save_path='data/5_fold/', k=5)
    # trainer.fit('data/train.json')
    # trainer.test(param_path='data/param', data_path="data/test.json")
