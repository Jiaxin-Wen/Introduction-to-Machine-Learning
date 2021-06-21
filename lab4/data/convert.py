import json

def convert(input_path, output_path):
    with open(input_path, 'r') as f:
            ori_data = f.readlines()
            ori_data = [i.strip().split('\t') for i in ori_data]
    name = ori_data[0]
    data = [dict(zip(name, i)) for i in ori_data[1:]]
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4, separators=[',', ':'])

    print('size = ', len(data))


if __name__ == '__main__':
    convert('test.csv', 'test.json')