import json
from numpy.lib.function_base import _percentile_dispatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree, svm
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.utils import resample
from sklearn.ensemble import AdaBoostRegressor as CheckRegressor

import numpy as np
import pickle
import random
import os
from tqdm import tqdm


def read_data(data_path, split, vectorizer):
    with open(f'{data_path}/{split}.json', 'r') as f:
        data = json.load(f)
    print("dataset size = ", len(data))

    text = []
    labels = []
    for sample in data:
        text.append(sample['reviewText'])
        labels.append(float(sample['overall']))

    if split != 'test':
        ids = vectorizer.fit_transform(text)
    else:
        ids = vectorizer.transform(text)

    return ids, labels

def evaluate(preds, labels, log_path):
    print('preds.shape = ', preds.shape)
    print('labels.shape = ', np.array(labels).shape)
    mae = mean_absolute_error(labels, preds)
    rmse = np.sqrt(mean_squared_error(labels, preds))
    print("mae = ", mae)
    print('rmse = ', rmse)
    with open(f'{log_path}/result.txt', 'w') as f:
        f.write(f"mae = {mae}\n")
        f.write(f"rmse = {rmse}\n")

class BaggingRegressor:
    def __init__(self, n, ratio, base_regressor, *, max_depth=None):
        '''
        n: 预测器数量
        ratio: 抽样比例
        base_regressor: svm or tree
        '''
        self.n = n
        self.ratio = ratio
        self.base_regressor = base_regressor
        self.max_depth = max_depth
        assert(self.base_regressor !='tree' or self.max_depth is not None) # 指定base_regressor为tree时需要指定max_depth
        
        self.models = []
        self.vectorizer = None

        self.name = f"bagging_regression_{self.base_regressor}_n{self.n}_ratio{self.ratio}" + f"_depth{self.max_depth}" if self.max_depth else ""
    
    def fit(self, split='train'):
        self.vectorizer = TfidfVectorizer()
        ids, labels = read_data('data', split, self.vectorizer)
        
        size = len(labels)
        sample_size = int(size * self.ratio)
        for i in tqdm(range(self.n)): # 训练n个分类器
            # bootstrap抽样: 从ids, labels中随机选取ratio比例的数据训练本次的分类器
            tmp_ids, tmp_labels = resample(ids, labels, replace=True, n_samples=sample_size) # replace设成True, 参考sklearn源码
            if self.base_regressor == 'svm':
                regressor = svm.LinearSVR()
            elif self.base_regressor == 'tree':
                regressor = tree.DecisionTreeRegressor(max_depth=self.max_depth) # set max_depth to 10
            regressor.fit(tmp_ids, tmp_labels)
            self.models.append(regressor)
        
        # save
        if not os.path.exists('param'):
            os.makedirs('param')
        base_path = f"param/{self.name}"
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        pickle.dump(self.vectorizer, open("param/vectorizer.pickle", 'wb'))
        for i, model in enumerate(self.models):
            pickle.dump(model, open(f"{base_path}/{i}.pickle", 'wb'))


    def test(self, split='test'):
        # load
        if self.models == []: 
            self.vectorizer = pickle.load(open("param/vectorizer.pickle", 'rb'))
            base_path = f"param/{self.name}"
            for i in range(self.n):
                self.models.append(pickle.load(open(f"{base_path}/{i}.pickle", 'rb')))

        # get data
        ids, labels = read_data('data', split, self.vectorizer)
        
        preds = []
        for model in self.models:
            scores = model.predict(ids) 
            preds.append(scores)

        preds = np.mean(np.array(preds).T, axis=-1) # confidence取平均
        evaluate(preds, labels, f"param/{self.name}")

class AdaBoostRegressor:
    def __init__(self, n, ratio, base_regressor, *, max_depth=None):
        '''
        n: 预测器数量
        ratio: 抽样比例
        base_regressor: svm or tree
        '''
        self.n = n
        self.ratio = ratio
        self.base_regressor = base_regressor
        self.max_depth = max_depth
        assert(self.base_regressor !='tree' or self.max_depth is not None) # 指定base_regressor为tree时需要指定max_depth
        
        self.models = []
        self.error_rate = None
        self.vectorizer = None

        self.name = f"adaboost_{self.base_regressor}_n{self.n}_ratio{self.ratio}" + (f"_depth{self.max_depth}" if self.max_depth else "")
        print('self.name = ', self.name)

    def update(self, i, sample_weight, scores, labels, mask):
        '''
        '''
        error_vec = np.abs(scores - labels)
        error_vec /= error_vec.max() # 用默认的linear loss
        error = (sample_weight[mask] * error_vec).sum()
        if error <= 0:
            return sample_weight, 1, 0
        elif error >= 0.5:
            return None, None, None # 退出循环，当前模型不加入结果
        
        # alpha = 0.5 * np.log((1. - error) / error) # ori adaboost
        # model_weight = alpha
        beta = error / (1. - error)
        model_weight = np.log(1. / beta)


        if i != self.n - 1:
            # sample_weight[mask] *= np.exp(-1 * alpha) # ori adaboost
            sample_weight[mask] *= np.power(beta, (1 - error_vec))

        return sample_weight, model_weight, error

    def kkk(self):
        vectorizer = TfidfVectorizer()
        regressor = CheckRegressor(svm.LinearSVR(), 10)
        train_ids, train_labels = read_data('data', 'train', vectorizer)
        test_ids, test_labels = read_data('data', 'test', vectorizer)
        regressor.fit(train_ids, train_labels)
        preds = regressor.predict(test_ids)
        evaluate(preds, test_labels, 'param/wtf')

    def fit(self, split='train'):
        self.vectorizer = TfidfVectorizer()
        ids, labels = read_data('data', split, self.vectorizer)
        labels = np.array(labels)

        size = len(labels)
        sample_size = int(size * self.ratio)
        sample_weight = np.ones(size) / size # 初始化权重
        self.model_weight = [] # 初始化每个模型的权重
        
        for i in tqdm(range(self.n)): # 训练n个分类器
            index = np.random.choice(np.arange(size), size=sample_size, replace=True, p=sample_weight)
            tmp_ids = ids[index]
            tmp_labels = labels[index]
            if self.base_regressor == 'svm':
                regressor = svm.LinearSVR()
            elif self.base_regressor == 'tree':
                regressor = tree.DecisionTreeRegressor(max_depth=self.max_depth) # set max_depth to 10
            regressor.fit(tmp_ids, tmp_labels)
            tmp_scores = regressor.predict(tmp_ids)
            tmp_sample_weight, tmp_model_weight, error = self.update(i, sample_weight, tmp_scores, tmp_labels, index) # 更新self.weight和self.error_rate
            if error is None: # error >= 0.5
                break
            elif error == 0:
                break
            else:
                tmp_sum = np.sum(tmp_sample_weight)
                if tmp_sum <= 0:
                    break
                if i < self.n - 1:
                    tmp_sample_weight /= tmp_sum
                sample_weight = tmp_sample_weight
                self.model_weight.append(tmp_model_weight)
                self.models.append(regressor)
        
        # save
        if not os.path.exists('param'):
            os.makedirs('param')
        base_path = f"param/{self.name}"
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        pickle.dump(self.vectorizer, open("param/vectorizer.pickle", 'wb'))
        for i, model in enumerate(self.models):
            pickle.dump(model, open(f"{base_path}/{i}.pickle", 'wb'))
        pickle.dump(self.model_weight, open(f"{base_path}/model_weight.pickle", 'wb'))


    def test(self, split='test'):
        # load
        if self.models == []: 
            self.vectorizer = pickle.load(open("param/vectorizer.pickle", 'rb'))
            base_path = f"param/{self.name}"
            for file in os.listdir(base_path): # 可能会提前终止, listdir
                if file != 'model_weight.pickle':
                    print('file = ', file)
                    self.models.append(pickle.load(open(f"{base_path}/{file}", 'rb')))
            self.model_weight = pickle.load(open(f"{base_path}/model_weight.pickle", 'rb')) # 要额外load一个error_rate
        
        # get data
        ids, labels = read_data('data', split, self.vectorizer)
        
        preds = []
        for model in self.models:
            scores = model.predict(ids) 
            preds.append(scores)
        preds = np.array(preds).T
        
        # 找到中位数作为预测值
        sorted_idx = np.argsort(preds, axis=1)
        self.model_weight = np.array(self.model_weight)
        weight_cdf = np.cumsum(self.model_weight[sorted_idx], axis=1)
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis=1)
        median_models = sorted_idx[np.arange(len(labels)), median_idx]
        final_preds = preds[np.arange(len(labels)), median_models]

        evaluate(final_preds, labels, f"param/{self.name}")

if __name__ == '__main__':
    random.seed(2021)
    np.random.seed(2021)

    regressor = BaggingRegressor(n=5, ratio=0.8, base_regressor='svm')
    # regressor = BaggingRegressor(n=1, ratio=1, base_regressor='svm')
    # regressor = BaggingRegressor(n=5, ratio=0.8, base_regressor='tree', max_depth=20)
    # regressor = BaggingRegressor(n=1, ratio=1, base_regressor='tree', max_depth=20)
    regressor.fit()
    regressor.test()

    # regressor = AdaBoostRegressor(n=5, ratio=0.8, base_regressor='svm')
    # regressor = AdaBoostRegressor(n=5, ratio=0.8, base_regressor='tree', max_depth=20)
    # regressor.fit()
    # regressor.test()
