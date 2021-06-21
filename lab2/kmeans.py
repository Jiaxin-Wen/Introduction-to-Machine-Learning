import os
from torchvision import datasets, transforms
import numpy as np
import random
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from munkres import Munkres
import munkres
from copy import copy
import json

class Kmeans:

    def __init__(self, k, max_step=None):
        '''
        n: 聚类数目
        max_step: 最大迭代次数
        '''
        self.k = k
        self.max_step = max_step
        self.centers = None
        self.load_data()
        self.m = Munkres()

    def km(self, matrix, max=1e9):
        '''km匹配'''
        matrix = munkres.make_cost_matrix(
            matrix, lambda cost: max - cost  # 最大化转化为最小化
        )
        indices = self.m.compute(matrix)
        result = [i[1] for i in indices]
        return result

    def load_data(self):
        '''加载MNIST数据'''
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
        data = datasets.MNIST(root='.', train=True, transform=trans, download=True)
        # data = [(data, target) for (data, target) in data][:50]
        self.n = len(data)
        print('dataset size = ', self.n)
        self.data = [None for _ in range(self.n)]
        self.labels = [None for _ in range(self.n)]
        self.preds = [None for _ in range(self.n)]
        self.clusters = [None for _ in range(self.n)]
        for i, (tensor, label) in enumerate(data):
            self.data[i] = tensor.squeeze().flatten().numpy()
            self.labels[i] = label
        self.data = np.array(self.data)
        self.tsne = TSNE(n_components=2, verbose=False) # 2维 

    def init_centers(self):
        '''随机初始化中心点'''
        idxs = np.random.permutation(self.n)[:self.k]
        self.centers = self.data[idxs]

    def assign_cluster(self):
        '''更新聚类结果'''
        for i, sample in enumerate(self.data):
            self.clusters[i] = int(np.argmin(np.linalg.norm(self.centers - sample, axis=-1))) # 重新计算类别, default是2范数
        return copy(self.clusters) # 注意copy, 否则收敛会判断错误

    def update_centers(self):
        '''更新中心点'''
        for i in range(self.k):
            mdata = [sample for j, sample in enumerate(self.data) if self.clusters[j] == i]
            self.centers[i] = np.mean(mdata, axis=0)

    def test(self):
        kmeans = KMeans(n_clusters=self.k, max_iter=300, n_init=5, init='k-means++') # n_init 初始化5次，选结果最好的
        self.clusters = kmeans.fit_predict(self.data)
        return self.clusters

    def fit(self, save_path=None):
        '''训练，参数保存到save_path'''
        # random initialize
        self.init_centers()
        
        # main loop
        prev_cluster_result = None
        for step in tqdm(range(self.max_step)):
            cur_cluster_result = self.assign_cluster() # 更新各数据点的分类
            self.update_centers() # 更新中心点
            if prev_cluster_result is not None and (prev_cluster_result == cur_cluster_result): # 判断收敛
                break
            else:
                prev_cluster_result = cur_cluster_result
        if save_path is not None:
            with open(save_path, 'w') as f:
                json.dump(self.clusters, f)
        return self.clusters

    def predict(self, mode="km"):
        '''预测，模式支持km和voting两种'''
        if mode == "km": # km
            cost_matrix = [[len([id for id in range(self.n) if self.clusters[id] == cluster and self.labels[id] == label]) for label in range(self.k)] for cluster in range(self.k)]
            pred = self.km(cost_matrix)
            for i in range(self.n):
                self.preds[i] = pred[self.clusters[i]]
        elif mode == 'voting': # majority voting majority voting (投票可能重复)
            for cluster in range(self.k): # 第i类
                pred = np.argmax([len([id for id in range(self.n) if self.clusters[id] == cluster and self.labels[id] == label]) for label in range(self.k)])
                for i in range(self.n):
                    if self.clusters[i] == cluster:
                        self.preds[i] = pred
        else:
            raise Exception("invalid model")

        self.evaluate()
        self.visualize()

    def evaluate(self):
        '''测自动指标'''
        cls_result = classification_report(self.labels, self.preds, target_names=[f"number {i}" for i in range(10)])
        print(cls_result)
        acc = np.mean(np.array(self.preds) == np.array(self.labels))
        print('acc = ', acc)

    def visualize(self):
        '''可视化'''
        sampled_idx = random.sample(list(range(self.n)), k=500)
        sampled_data = self.data[sampled_idx]
        sampled_clusters = np.array(self.clusters)[sampled_idx]
        weights = self.tsne.fit_transform(sampled_data) # 降维
        plt.scatter(weights[:, 0], weights[:, 1], c=sampled_clusters)
        plt.savefig('cluster.png')

    def load(self, load_path):
        '''加载数据'''
        with open(load_path, 'r') as f:
            self.clusters = json.load(f)
        
        
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def train(k, max_step):
    kmeans = Kmeans(k, max_step)
    kmeans.fit(save_path=f"param_k{k}.json")
    kmeans.predict(mode='voting')


def test(k, load_path):
    kmeans = Kmeans(k=k)
    kmeans.load(load_path)
    print('param loaded!')
    kmeans.predict(mode='voting')

if __name__ == '__main__':
    set_seed(2021)
    train(k=10, max_step=300)
    # test(k=10, load_path='param_k10.json')