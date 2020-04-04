import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from GNNs import GNN

def load_data(data_file):
    graphs = pickle.load(open(data_file, 'rb'))
    node_types = set()
    label_types = set()
    for graph in graphs:
        #print(graph)
        # raise TypeError
        edges = list(graph[0])
        nodes = list(graph[1])
        label = graph[2]
        node_types |= set([x[1] for x in nodes]) ## output the kinds of nodes
        label_types.add(label)
    node2index = {n: i for i, n in enumerate(node_types)}
    label2index = {l: i for i, l in enumerate(label_types)}
    adj_lists = []
    features = []
    labels = torch.zeros(len(graphs), len(label_types))
    for graph in graphs:
        edges = list(graph[0])
        nodes = list(graph[1])
        label = graph[2]
        adjlist = defaultdict(set)
        feature = torch.zeros(len(nodes), len(node_types))
        for i, j in edges:
            adjlist[i].add(j)
            adjlist[j].add(i)
        for i, n in nodes:
            feature[i, node2index[n]] = 1
        labels[len(adj_lists), label2index[label]] = 1
        adj_lists.append(adjlist)
        features.append(torch.FloatTensor(feature).to(device))

if __name__ == "__main__":
    load_data("graphsage/data/fda_data.pkl")