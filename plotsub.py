import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd

path = 'D:/first_one/SUGAR-master/bbbp_data/subgcn'
adj_ids = np.load(path + '/adj_ids_0.npy', allow_pickle=True)
sub_adj = np.load(path + '/k_sub_adj_0.npy', allow_pickle=True)
sub_feature = np.load(path + '/k_sub_feature_0.npy', allow_pickle=True)

# # 224
# print(adj_ids.shape, adj_ids.ndim)
# # 224*19*5*5
# print(sub_adj.shape, sub_adj.ndim)
# # 224*19*5*13
# print(sub_feature.shape, sub_feature.ndim)

id = adj_ids[0]
origin_path = 'D:/first_one/SUGAR-master/bbbp_data/origin/'
id_adj = pd.read_csv(origin_path + str(id) + '_adj.csv', header=None)
feature = pd.read_csv(origin_path + str(id) + '_feature.csv', header=None)
# print(origin_adj)
# print(feature)
# print(sub_adj[0][0], sub_adj[0][1], sub_adj[0][2])

colors_dict = {
    "Cl": "khaki",
    "C": "cornflowerblue",
    "N": "yellowgreen",
    "O": "peru"
}


def draw(adj, f):
    adj_size = adj.shape[0]
    # print(adj_size)
    colors = ["" for i in range(adj_size)]
    nlist = ["" for i in range(adj_size)]
    for i in range(adj_size):
        colors[i] = colors_dict[f[i][0]]
        nlist[i] = f[i][0]
    # print(colors)
    # print(nlist)

    G = nx.Graph()
    G.add_nodes_from(range(adj_size))
    # G.add_nodes_from(nlist)
    # print(G.nodes)
    edge_list = []
    for i in range(adj_size):
        for j in range(adj_size):
            if adj[i][j]:
                edge_list.append((i, j))
                G.add_edge(i, j)
    # print(G.edges, G.number_of_edges())
    pos = nx.kamada_kawai_layout(G)
    pos = nx.circular_layout(G)
    pos = nx.spectral_layout(G)

    c_list = [v for v in range(adj_size) if nlist[v] == 'C']
    nx.draw_networkx_nodes(G, pos=pos, nodelist=c_list, node_size=100, node_color=colors_dict['C'], label='C')
    cl_list = [v for v in range(adj_size) if nlist[v] == 'Cl']
    nx.draw_networkx_nodes(G, pos=pos, nodelist=cl_list, node_size=100, node_color=colors_dict['Cl'], label='CL')
    n_list = [v for v in range(adj_size) if nlist[v] == 'N']
    nx.draw_networkx_nodes(G, pos=pos, nodelist=n_list, node_size=100, node_color=colors_dict['N'], label='N')
    # o_list = [v for v in range(adj_size) if nlist[v] == 'O']
    # nx.draw_networkx_nodes(G, pos=pos, nodelist=o_list, node_size=100, node_color=colors_dict['O'], label='O')
    nx.draw_networkx_edges(G, pos=pos, edgelist=edge_list)
    # plt.legend()
    plt.axis('off')
    # nx.draw(G, pos=pos, node_color=colors, node_size=100)
    plt.show()


# print(id_adj.values, feature.values, type(feature.values))
# draw(id_adj.values, feature.values)

sub1 = sub_adj[id][0]
sub1_f = sub_feature[id][0]
sub1_f = np.array([['C'], ['N'], ['C'], ['C'], ['Cl']])
draw(sub1, sub1_f)
#
#
# sub2 = sub_adj[id][2]
# sub2_f = sub_feature[id][2]
# sub2_f = np.array([['O'], ['C'], ['C'], ['C'], ['O']])
# # print(sub2, type(sub2))
# # print(sub2_f, type(sub2_f))
# draw(sub2, sub2_f)


# sub3 = sub_adj[id][7]
# sub3_f = sub_feature[id][7]
# sub3_f = np.array([['C'], ['O'], ['C'], ['O'], ['C']])
# print(sub3, type(sub3))
# print(sub3_f, type(sub3_f))
# draw(sub3, sub3_f)







