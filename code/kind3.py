import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv, GATv2Conv, GINConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

##########################################
# 数据加载与预处理
##########################################
def load_usair_data(file_path):
    edge_list = []
    nodes = set()
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                v, u = map(int, parts[:2])
                edge_list.append((u, v))
                nodes.add(u)
                nodes.add(v)
    nodes = sorted(list(nodes))
    node_map = {node: i for i, node in enumerate(nodes)}
    new_edge_list = [(node_map[u], node_map[v]) for u, v in edge_list]
    num_nodes = len(nodes)
    edge_index = torch.tensor(new_edge_list, dtype=torch.long).t().contiguous()
    x = torch.randn(num_nodes, 16)
    data = Data(x=x, edge_index=edge_index)
    return data

##########################################
# 模型定义
##########################################

class SELAREncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SELAREncoder, self).__init__()
        self.branch1 = GCNConv(in_channels, hidden_channels)
        self.branch2_1 = GCNConv(in_channels, hidden_channels)
        self.branch2_2 = GCNConv(hidden_channels, hidden_channels)
        self.branch3_1 = GCNConv(in_channels, hidden_channels)
        self.branch3_2 = GCNConv(hidden_channels, hidden_channels)
        self.branch3_3 = GCNConv(hidden_channels, hidden_channels)
        self.attention = nn.Linear(hidden_channels, 1)
        self.fc = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        h1 = F.relu(self.branch1(x, edge_index))
        h2 = F.relu(self.branch2_1(x, edge_index))
        h2 = F.relu(self.branch2_2(h2, edge_index))
        h3 = F.relu(self.branch3_1(x, edge_index))
        h3 = F.relu(self.branch3_2(h3, edge_index))
        h3 = F.relu(self.branch3_3(h3, edge_index))
        H = torch.stack([h1, h2, h3], dim=1)
        attn_scores = self.attention(H)
        attn_weights = torch.softmax(attn_scores, dim=1)
        h_meta = torch.sum(attn_weights * H, dim=1)
        h_out = h_meta + h1
        out = self.fc(h_out)
        return out

class MAGNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MAGNNEncoder, self).__init__()
        self.branch1 = GCNConv(in_channels, hidden_channels)
        self.branch2_1 = GCNConv(in_channels, hidden_channels)
        self.branch2_2 = GCNConv(hidden_channels, hidden_channels)
        self.branch3_1 = GCNConv(in_channels, hidden_channels)
        self.branch3_2 = GCNConv(hidden_channels, hidden_channels)
        self.branch3_3 = GCNConv(hidden_channels, hidden_channels)
        self.attention = nn.Linear(hidden_channels, 1)
        self.fc = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        h1 = F.relu(self.branch1(x, edge_index))
        h2 = F.relu(self.branch2_1(x, edge_index))
        h2 = F.relu(self.branch2_2(h2, edge_index))
        h3 = F.relu(self.branch3_1(x, edge_index))
        h3 = F.relu(self.branch3_2(h3, edge_index))
        h3 = F.relu(self.branch3_3(h3, edge_index))
        H = torch.stack([h1, h2, h3], dim=1)
        attn_scores = self.attention(H)
        attn_weights = torch.softmax(attn_scores, dim=1)
        h_meta = torch.sum(attn_weights * H, dim=1)
        out = self.fc(h_meta)
        return out

class MAHGAEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MAHGAEncoder, self).__init__()
        self.struct_conv1 = GCNConv(in_channels, hidden_channels)
        self.struct_conv2 = GCNConv(hidden_channels, hidden_channels)
        self.meta_branch1 = GCNConv(in_channels, hidden_channels)
        self.meta_branch2_1 = GCNConv(in_channels, hidden_channels)
        self.meta_branch2_2 = GCNConv(hidden_channels, hidden_channels)
        self.attention = nn.Linear(hidden_channels, 1)
        self.fc = nn.Linear(hidden_channels * 2, out_channels)
        
    def forward(self, x, edge_index):
        h_struct = F.relu(self.struct_conv1(x, edge_index))
        h_struct = F.relu(self.struct_conv2(h_struct, edge_index))
        h1 = F.relu(self.meta_branch1(x, edge_index))
        h2 = F.relu(self.meta_branch2_1(x, edge_index))
        h2 = F.relu(self.meta_branch2_2(h2, edge_index))
        H = torch.stack([h1, h2], dim=1)
        attn_scores = self.attention(H)
        attn_weights = torch.softmax(attn_scores, dim=1)
        h_meta = torch.sum(attn_weights * H, dim=1)
        h_comb = torch.cat([h_struct, h_meta], dim=1)
        out = self.fc(h_comb)
        return out

class LinkPredictor(nn.Module):
    def __init__(self, in_channels):
        super(LinkPredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 1)
        )
        
    def forward(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=1)
        return torch.sigmoid(self.mlp(x))

##########################################
# 评估指标计算
##########################################
def evaluate(model, predictor, data):
    model.eval()
    predictor.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
    pos_edge_index = data.pos_edge_label_index
    neg_edge_index = data.neg_edge_label_index
    pos_pred = predictor(z[pos_edge_index[0]], z[pos_edge_index[1]]).view(-1)
    neg_pred = predictor(z[neg_edge_index[0]], z[neg_edge_index[1]]).view(-1)
    preds = torch.cat([pos_pred, neg_pred]).detach().cpu().numpy()
    labels = np.hstack([np.ones(pos_pred.size(0)), np.zeros(neg_pred.size(0))])
    pred_labels = (preds >= 0.5).astype(int)
    acc = accuracy_score(labels, pred_labels)
    auc = roc_auc_score(labels, preds)
    precision = precision_score(labels, pred_labels)
    recall = recall_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels)
    return acc, auc, precision, recall, f1

##########################################
# 主函数与训练流程
##########################################
def main():
    parser = argparse.ArgumentParser(description='Link Prediction with Heterogeneous Meta-Path Methods')
    parser.add_argument('--model', type=int, default=1, choices=[1,2,3],
                        help='选择模型: 1-SELAR, 2-MAGNN, 3-MAHGA')
    parser.add_argument('--data', type=str, default='data/data.txt', help='USAir 数据文件路径')
    parser.add_argument('--epochs', type=int, default=1000, help='训练轮数')
    args = parser.parse_args()
    
    data = load_usair_data(args.data)
    transform = RandomLinkSplit(is_undirected=True, split_labels=True,
                                neg_sampling_ratio=1.0, add_negative_train_samples=True)
    train_data, val_data, test_data = transform(data)
    
    in_channels = data.num_features
    hidden_channels = 32
    out_channels = 16
    
    if args.model == 1:
        print("Using SELAR model...")
        encoder = SELAREncoder(in_channels, hidden_channels, out_channels)
        model_name = "SELAR"
    elif args.model == 2:
        print("Using MAGNN model...")
        encoder = MAGNNEncoder(in_channels, hidden_channels, out_channels)
        model_name = "MAGNN"
    elif args.model == 3:
        print("Using MAHGA model...")
        encoder = MAHGAEncoder(in_channels, hidden_channels, out_channels)
        model_name = "MAHGA"
    else:
        print("Invalid model choice!")
        sys.exit(1)
        
    predictor = LinkPredictor(out_channels)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=0.005)
    
    best_val_agg = -float('inf')
    best_epoch = 0
    best_val_metrics = None
    best_state = None

    print("Training...")
    for epoch in range(args.epochs):
        encoder.train()
        predictor.train()
        optimizer.zero_grad()
        z = encoder(train_data.x, train_data.edge_index)
        pos_edge_index = train_data.pos_edge_label_index
        neg_edge_index = train_data.neg_edge_label_index
        pos_pred = predictor(z[pos_edge_index[0]], z[pos_edge_index[1]]).view(-1)
        neg_pred = predictor(z[neg_edge_index[0]], z[neg_edge_index[1]]).view(-1)
        pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
        neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch+1:03d}, Loss: {loss.item():.4f}")
            
        if (epoch + 1) % 100 == 0:
            val_metrics = evaluate(encoder, predictor, val_data)
            agg = sum(val_metrics)
            print(f"Epoch: {epoch+1:03d}, Validation Metrics: {val_metrics}, Aggregate: {agg:.4f}")
            if agg > best_val_agg:
                best_val_agg = agg
                best_epoch = epoch + 1
                best_val_metrics = val_metrics
                best_state = encoder.state_dict()
    val_result_str = "\nBest Validation Metrics (at epoch {}):\nAccuracy: {:.4f}, AUC: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}".format(
        best_epoch, *best_val_metrics)
    print(val_result_str)
    
    encoder.load_state_dict(best_state)
    test_metrics = evaluate(encoder, predictor, test_data)
    test_result_str = "\nTest Metrics (using best model from validation):\nAccuracy: {:.4f}, AUC: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}".format(
        *test_metrics)
    print(test_result_str)
    
    result_file = f"result/{model_name}.txt"
    with open(result_file, "a", encoding="utf-8") as f:
        f.write(val_result_str + "\n")
        f.write(test_result_str + "\n")
        f.write("\n" + "-"*50 + "\n\n")

if __name__ == '__main__':
    main()