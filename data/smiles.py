from pysmiles import read_smiles
import networkx as nx
import pandas as pd

data = pd.read_csv('bace.csv')
# print(dataset)
target = data['Class']
smiles = data['mol']
# print(len(target))
# print(len(smiles))
m = len(target)
y = []
for i in range(m):
    if target[i] == 0:
        y.append(-1)
    else:
        y.append(1)
y = pd.DataFrame(y)
y.to_csv('./bace_data/target.csv', index=0, header=0)

n = len(smiles)
element = []
all_len = 0
max_len = 0

for i in range(n):
    smile = smiles[i]
    mol = read_smiles(smile)
    # adjacency matrix
    adj = nx.to_numpy_matrix(mol)
    adj = pd.DataFrame(adj)
    adj.to_csv('./bace_data/'+str(i)+'_adj.csv', index=0, header=0)
    e = nx.get_node_attributes(mol, name='element')
    e = list(e.values())
    l = len(e)
    all_len += l
    if l > max_len:
        max_len = l
    # print(l)
    # print(e)
    feature = []
    for j in range(l):
        feature.append(e[j])
        if e[j] not in element:
            element.append(e[j])
    feature = pd.DataFrame(feature)
    feature.to_csv('./bace_data./'+str(i)+'_feature.csv', index=0, header=0)

# n = 2050
average_len = all_len / n
print('num of elements:', len(element))
print('elements:', element)
print('average_len:', average_len)
print('max_len:', max_len)




