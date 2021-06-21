import csv
import numpy as np
import json


def bagging(input_files, output_file):
    scores = []
    for file in input_files:
        with open(file, 'r') as f:
            scores.append(json.load(f))
    # for file in input_files:
    #     with open(file, 'r') as f:
    #         reader = csv.reader(f)
    #         content = [i for i in reader]
    #         scores.append([i[1] for i in content[1:]])

    scores = np.array(scores)
    scores = np.mean(scores, axis=0)
    index = np.arange(5) + 1
    index = np.expand_dims(index, axis=0)
    index = np.tile(index, (scores.shape[0], 1))
    preds = np.sum(scores * index, axis=-1)
    
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Predicted'])
        for i, pred in enumerate(preds):
            writer.writerow([i+1, float(pred)])


if __name__ == '__main__':
    files = [
        "roberta_heu_version_21.json",
        "roberta_heu_version_20.json"
    ]
    bagging(files, 'roberta20_roberta21.csv')
    


