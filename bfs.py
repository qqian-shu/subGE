import numpy as np
import pandas as pd
import os
from collections import defaultdict

def bfs(adj_data, node, sub_node, nodes):
    # 挑选出node节点连接的所有节点，并记录其degree
    adj_degree = defaultdict(int)
    neg_nodes = adj_data[adj_data[node]==1].index
    for i in neg_nodes:
        adj_degree[i] = adj_data[adj_data[i]==1].shape[0]

    # 根据degree排序
    sort_degree = sorted(adj_degree.items(), key = lambda x : x[1])
    next_nodes = []
    for i in sort_degree:
        if i[0] in nodes:
            continue
        nodes.append(i[0])
        sub_node -= 1
        # 得到所需子图点数就可以退出
        if sub_node <= 0: return sub_node
        # 把当前未纳入的节点作为后续目标节点
        next_nodes.append(i[0])
    # 使用bfs搜寻后续节点
    for node in next_nodes:
        sub_node = bfs(adj_data, node, sub_node, nodes)
        if sub_node <= 0: return sub_node
    return sub_node


def dfs(adj_data, node, sub_node, nodes):
    # 挑选出node节点连接的所有节点，并记录其degree
    adj_degree = defaultdict(int)
    neg_nodes = adj_data[adj_data[node]==1].index
    for i in neg_nodes:
        adj_degree[i] = adj_data[adj_data[i]==1].shape[0]

    # 根据degree排序
    sort_degree = sorted(adj_degree.items(), key = lambda x : x[1])
    for i in sort_degree:
        if i[0] in nodes:
            continue
        nodes.append(i[0])
        sub_node -= 1
        # 得到所需子图点数就可以退出
        if sub_node <= 0: return sub_node
        # 深入探索
        sub_node = dfs(adj_data, i[0], sub_node, nodes)
        if sub_node <= 0: return sub_node

    return sub_node


root_path = 'bbbp_data'
img_num = 2050
img_count = 28
node_num = 5
fea_set = set()
max_num = 0
mole_num = 0

for id in range(0, img_num):
    adj_name = f'{id}_adj.csv'
    fea_name = f'{id}_feature.csv'
    if os.path.exists(os.path.join(root_path, f'origin', adj_name)):
        print(id)
    else:
        continue
    adj_data = pd.read_csv(os.path.join(root_path, f'origin', adj_name), header=None)
    fea_data = pd.read_csv(os.path.join(root_path, f'origin', fea_name), header=None)
    if adj_data.shape[0] < 9 or adj_data.shape[0] > 35:
        continue
    mole_num += 1
    fea_set |= set(fea_data[0].to_list())
    max_num = max(max_num, adj_data.shape[0])
print(fea_set)

print(mole_num, max_num)

label = {1: [0, 1], -1: [1, 0]}

fea_one_hot = np.eye(len(fea_set))
feas = {}
for a, oh in zip(fea_set, fea_one_hot):
    feas[a] = oh
print(feas)
# one-hot

adj = np.zeros((mole_num, max_num, max_num))
sub_adjs = np.zeros((mole_num, img_count, node_num, node_num))
graphs_label = np.zeros((mole_num, 2))
features = np.zeros((mole_num, img_count, node_num, len(fea_set)))


label_data = pd.read_csv(os.path.join(root_path, r'origin', 'target.csv'), header=None)

mole_index = 0
# print("fea_data.shape[0]:", fea_data.shape[0])
for id in range(1, img_num):
    adj_name = f'{id}_adj.csv'
    fea_name = f'{id}_feature.csv'
    if os.path.exists(os.path.join(root_path, f'origin', adj_name)):
        print("id:", id)
    else:
        continue
    if id==6549:
        continue
    adj_data = pd.read_csv(os.path.join(root_path, r'origin', adj_name), header=None)
    fea_data = pd.read_csv(os.path.join(root_path, r'origin', fea_name), header=None)

    # if adj_data.shape[0] < 9 or adj_data.shape[0] > 28:
    if adj_data.shape[0] < 9 or adj_data.shape[0] > img_count:
        continue

    sub_count = img_count
    if sub_count > fea_data.shape[0]:
        sub_count = fea_data.shape[0]

    print("fea_data.shape[0]:", fea_data.shape[0])
    # adj
    adj[mole_index, 0:fea_data.shape[0], 0:fea_data.shape[0]] = adj_data
    # graphs_label
    graphs_label[mole_index, ...] = label[label_data.iat[id, 0]]
    get_nodes = []
    for si in range(sub_count):
        sub_node = node_num
        if sub_node > fea_data.shape[0]:
            sub_node = fea_data.shape[0]
        nodes = []
        node = np.random.randint(fea_data.shape[0])
        while node in get_nodes:
            node = np.random.randint(fea_data.shape[0])
        nodes.append(node)
        sub_node -= 1

        dfs(adj_data, node, sub_node, nodes)

        sub_node = min(fea_data.shape[0], node_num)
        for i in range(sub_node):
            sub_adjs[mole_index, si, i, i] = 1
            if i >= fea_data.shape[0]:
                features[mole_index, si, i, :] = feas['C']
                continue
            features[mole_index, si, i, :] = feas[fea_data.iat[nodes[i], 0]]

            for j in range(i + 1, sub_node):
                sub_adjs[mole_index, si, i, j] = adj_data.iat[nodes[i], nodes[j]]
                sub_adjs[mole_index, si, j, i] = adj_data.iat[nodes[i], nodes[j]]
    for si in range(sub_count, img_count):
        for i in range(node_num):
            features[mole_index, si, i, :] = feas['C']
            sub_adjs[mole_index, si, i, i] = 1
    mole_index += 1

print(np.where(graphs_label[:, 0] == 0)[0].shape)


indexs1 = np.where(graphs_label[:, 1] == 1)[0]
indexs2 = np.where(graphs_label[:, 0] == 1)[0][:indexs1.shape[0]]
index = np.append(indexs1, indexs2)
np.random.shuffle(index)


root_path = r'tox21_data\NR-AhR'
np.save(os.path.join(root_path, r'origin', 'adj.npy'), adj[index, ...])
np.save(os.path.join(root_path, r'origin', 'sub_adj.npy'), sub_adjs[index, ...])
np.save(os.path.join(root_path, r'origin', 'features.npy'), features[index, ...])
np.save(os.path.join(root_path, r'origin', 'graphs_label.npy'), graphs_label[index, ...])


root_path = r'SUGAR_CODE\dataset\MUTAG'
adj_m = np.load(os.path.join(root_path, 'adj.npy'))
sub_adj_m = np.load(os.path.join(root_path, 'sub_adj.npy'))
features_m = np.load(os.path.join(root_path, 'features.npy'))
graphs_label_m = np.load(os.path.join(root_path, 'graphs_label.npy'))

root_path = r'bbbp_data\origin'
adj_b = np.load(os.path.join(root_path, 'adj.npy'))
sub_adj_b = np.load(os.path.join(root_path, 'sub_adj.npy'))
features_b = np.load(os.path.join(root_path, 'features.npy'))
graphs_label_b = np.load(os.path.join(root_path, 'graphs_label.npy'))



# id1 = 25
# plt.figure(1, figsize=(7,4))
# for i in range(sub_adj_m.shape[1]):
#     plt.subplot(4, 7, i+1)
#     plt.imshow(sub_adj_m[id1, i, ...])
# id2 = 12
# plt.figure(2, figsize=(7,4))
# for i in range(sub_adj_b.shape[1]):
#     plt.subplot(4, 7, i+1)
#     plt.imshow(sub_adj_b[id2, i, ...])