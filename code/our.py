import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from node2vec import Node2Vec

edges_list = []
with open("data/test.csv", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        v, u = parts[0], parts[1]
        u = int(u)
        v = int(v)
        edges_list.append((u, v))

G_full = nx.Graph()
G_full.add_edges_from(edges_list)

nodes = list(G_full.nodes())
mapping = {node: i for i, node in enumerate(sorted(nodes))}
G_full = nx.relabel_nodes(G_full, mapping)
edges_list = [(mapping[u], mapping[v]) for u, v in edges_list]

num_nodes = G_full.number_of_nodes()

node2vec = Node2Vec(G_full, dimensions=32, walk_length=10, num_walks=100, workers=4)
n2v_model = node2vec.fit()
node_embeddings = n2v_model.wv
node_attr_dim = 32
node_features = np.array([node_embeddings[str(node)] for node in range(num_nodes)])

edges_array = np.array(edges_list, dtype=int)
perm = np.random.permutation(len(edges_array))
train_size = int(0.8 * len(edges_array))
val_size = int(0.1 * len(edges_array))
train_edges = edges_array[perm[:train_size]]
val_edges = edges_array[perm[train_size : train_size + val_size]]
test_edges = edges_array[perm[train_size + val_size :]]

train_edge_index = torch.tensor(train_edges, dtype=torch.long).t().contiguous()
val_edge_index = torch.tensor(val_edges, dtype=torch.long).t().contiguous()
test_edge_index = torch.tensor(test_edges, dtype=torch.long).t().contiguous()

train_edge_index = torch.cat([train_edge_index, train_edge_index.flip(0)], dim=1)
val_edge_index = torch.cat([val_edge_index, val_edge_index.flip(0)], dim=1)
test_edge_index = torch.cat([test_edge_index, test_edge_index.flip(0)], dim=1)

train_edge_index = torch.unique(train_edge_index, dim=1)
val_edge_index = torch.unique(val_edge_index, dim=1)
test_edge_index = torch.unique(test_edge_index, dim=1)

x = torch.tensor(node_features, dtype=torch.float)
data = Data(x=x, edge_index=train_edge_index)

train_edge_set = set(tuple(sorted(e)) for e in train_edges.tolist())
full_edge_set = set(tuple(sorted(e)) for e in edges_list)


def sample_negative_edges(num_samples, forbidden_set):
    neg_edges = []
    while len(neg_edges) < num_samples:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        if u == v:
            continue
        edge_tuple = (u, v) if u < v else (v, u)
        if edge_tuple in forbidden_set:
            continue
        neg_edges.append([u, v])
        forbidden_set.add(edge_tuple)
    return np.array(neg_edges, dtype=int)


train_neg_edges = sample_negative_edges(len(train_edges), set(train_edge_set))
val_neg_edges = sample_negative_edges(len(val_edges), set(full_edge_set))
test_neg_edges = sample_negative_edges(len(test_edges), set(full_edge_set))

train_pairs = np.vstack([train_edges, train_neg_edges])
val_pairs = np.vstack([val_edges, val_neg_edges])
test_pairs = np.vstack([test_edges, test_neg_edges])
train_labels = np.hstack([np.ones(len(train_edges)), np.zeros(len(train_neg_edges))])
val_labels = np.hstack([np.ones(len(val_edges)), np.zeros(len(val_neg_edges))])
test_labels = np.hstack([np.ones(len(test_edges)), np.zeros(len(test_neg_edges))])
train_pairs_tensor = torch.tensor(train_pairs, dtype=torch.long)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.float)


class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super(GATEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.2)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=heads, dropout=0.2)
        self.conv3 = GATConv(out_channels * heads, out_channels, heads=heads, dropout=0.2)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        return x


in_dim = node_attr_dim
hidden_dim = 128
embed_dim = 64
model = GATEncoder(in_dim, hidden_dim, embed_dim, heads=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=100, verbose=True)

best_val_score = -1
best_epoch = 0
best_model = None
num_epochs = 1500

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    z = model(data.x, data.edge_index)
    u = train_pairs_tensor[:, 0]
    v = train_pairs_tensor[:, 1]
    scores = (z[u] * z[v]).sum(dim=1)
    preds = torch.sigmoid(scores)
    loss = torch.nn.functional.binary_cross_entropy(preds, train_labels_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            z = model(data.x, data.edge_index)
        val_u = torch.tensor(val_pairs[:, 0], dtype=torch.long)
        val_v = torch.tensor(val_pairs[:, 1], dtype=torch.long)
        val_scores = (z[val_u] * z[val_v]).sum(dim=1)
        val_preds = torch.sigmoid(val_scores)
        val_pred_labels = (val_preds > 0.5).int()
        precision = precision_score(val_labels, val_pred_labels, zero_division=0)
        recall = recall_score(val_labels, val_pred_labels, zero_division=0)
        f1 = f1_score(val_labels, val_pred_labels, zero_division=0)
        auc = roc_auc_score(val_labels, val_preds)
        accuracy = accuracy_score(val_labels, val_pred_labels)
        val_score = precision + recall + f1 + auc + accuracy
        print(f"Epoch {epoch}, Validation Score: {val_score:.4f}, Loss: {loss.item():.4f}")
        scheduler.step(val_score)
        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            best_model = model.state_dict()

model.load_state_dict(best_model)
model.eval()
with torch.no_grad():
    z = model(data.x, data.edge_index)
test_u = torch.tensor(test_pairs[:, 0], dtype=torch.long)
test_v = torch.tensor(test_pairs[:, 1], dtype=torch.long)
test_scores = (z[test_u] * z[test_v]).sum(dim=1)
test_preds = torch.sigmoid(test_scores)
test_pred_labels = (test_preds > 0.5).int()
test_precision = precision_score(test_labels, test_pred_labels, zero_division=0)
test_recall = recall_score(test_labels, test_pred_labels, zero_division=0)
test_f1 = f1_score(test_labels, test_pred_labels, zero_division=0)
test_auc = roc_auc_score(test_labels, test_preds)
test_accuracy = accuracy_score(test_labels, test_pred_labels)

print(f"Best Epoch: {best_epoch}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall:    {test_recall:.4f}")
print(f"Test F1-score:  {test_f1:.4f}")
print(f"Test AUC:       {test_auc:.4f}")
print(f"Test Accuracy:  {test_accuracy:.4f}")
