import csv
import collections

def print_label_raito(path):
    with open(path) as f:
        reader = csv.reader(f)
        # ヘッダーの削除処理
        labels = [row[1] for row in reader][1:]

    c = collections.Counter(labels)
    print(c)


path_list = [
    '/media/syu/7d1c582d-4eb8-4836-9f6e-6a8bc8609413/data/signate/431/train_images.csv',
    '/home/syu/src/signate/431/logs/k-fold_benchmark/submit.csv',
    '/home/syu/src/signate/431/logs/20210218-1613660289/submit.csv',
    '/home/syu/src/signate/431/logs/test_raito/submit.csv'
]

for p in path_list:
    print_label_raito(p)

