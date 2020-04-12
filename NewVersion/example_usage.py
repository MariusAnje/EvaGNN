import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from GNNs import GNN


def load_pubmed():
    #hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty(num_nodes, dtype=np.int64)
    node_map = {}
    with open("pubmed/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1])-1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])

    row = []
    col = []
    with open("pubmed/Pubmed-Diabetes.DIRECTED.cites.tab", 'r') as fp:
        fp.readline()
        fp.readline()
        for i, line in enumerate(fp):
            # if i == 0: continue
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            row.extend([paper1, paper2])
            col.extend([paper2, paper1])
    row = np.asarray(row)
    col = np.asarray(col)
    adj_matrix = csr_matrix((np.ones(len(row)), (row, col)), shape=(len(node_map), len(node_map)))

    return feat_data, labels, adj_matrix


def load_cora():
    """load cora dataset"""
    cora_content_file = 'cora/cora.content'
    cora_cite_file = 'cora/cora.cites'
    # load features and labels
    feat_data = []
    labels = [] # label sequence of node
    node_map = {} # map node to Node_ID
    label_map = {} # map label to Label_ID
    with open(cora_content_file, 'r') as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data.append([float(x) for x in info[1:-1]])
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels.append(label_map[info[-1]])
    raw_features = np.asarray(feat_data)
    labels = np.asarray(labels, dtype=np.int64)
    # load adjacency matrix
    row = []
    col = []
    with open(cora_cite_file, 'r') as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            assert len(info) == 2
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            row.extend([paper1, paper2])
            col.extend([paper2, paper1])
    row = np.asarray(row)
    col = np.asarray(col)
    adj_matrix = csr_matrix((np.ones(len(row)), (row, col)), shape=(len(node_map), len(node_map)))

    """
    Example of using GraphSAGE for supervised learning.
    using CPU and do not print training progress
    """

    return raw_features, labels, adj_matrix


def GNN_run(raw_features, labels, adj_matrix, index):

    sample_size = 10
    batch_size = 20
    layer_num = index
    
    gnn = GNN(adj_matrix, features=raw_features, labels=labels, supervised=True, model='graphsage', n_layer = layer_num, device='cuda', batch_size=batch_size, sample_size = sample_size)
    # train the model
    gnn.fit()
    # make predictions with the built-in MLP classifier and evaluate
    preds = gnn.predict()
    f1 = f1_score(labels, preds, average='micro')
    print(f'F1 score for supervised learning: {f1:.4f}')
    embs = gnn.generate_embeddings()

    fw = open('results.tsv', 'a')
    fw.write('{}\t{}\t{}\t{}\n'.format(str(layer_num),str(sample_size), str(batch_size), str(f1)))

    # """
    # Example of using Graph Attention Network for unsupervised learning.
    # using CUDA and print training progress
    # """
    # gnn = GNN(adj_matrix, features=raw_features, supervised=False, model='gat', device='cuda')
    # # train the model
    # gnn.fit()
    # # get the node embeddings with the trained GAT
    # embs = gnn.generate_embeddings()
    # # evaluate the embeddings with logistic regression
    # lr = LogisticRegression(penalty='l2', random_state=0, solver='liblinear')
    # preds = lr.fit(embs, labels).predict(embs)
    # f1 = f1_score(labels, preds, average='micro')
    # print(f'F1 score for unsupervised learning: {f1:.4f}')

if __name__ == "__main__":

    feat_data, labels, adj_matrix = load_pubmed()
    # feat_data, labels, adj_matrix = load_cora()

    for i in [1,3,4]:
        GNN_run(feat_data, labels, adj_matrix, i)