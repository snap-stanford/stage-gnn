import argparse
import os.path as osp
import time
import random
import numpy as np
import torch

# Set seeds and configure deterministic behavior
seed = 32
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Now import other modules
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GATv2Conv as GATConv
from torch_geometric.nn.conv import PNAConv as PNAConv
from torch_geometric.nn.pool import global_add_pool
from feature_method import encode_input_features
from models import MPNN

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, num_layers, method, dropout_rate=0.6, model="GINConv", edge_embed_dim=None, edge_num_layers=1, edge_dim=1):
        super().__init__()
        self.model = model
        self.method = method
        self.dropout = dropout_rate
        self.out_channels = out_channels
        if method == "stage":
            self.edgegraph_model = MPNN(input_dim=1, hidden_dim=edge_embed_dim, num_layers=edge_num_layers, edge_model="GINEConv", edge_dim=edge_dim)
            self.edge_embed_norm_layer = nn.LayerNorm(edge_embed_dim)
        elif method == "normalized":
            # MLP 
            self.phi = torch.nn.Sequential(
                torch.nn.Linear(1, in_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(in_channels, in_channels),
            )
            self.rho = torch.nn.Sequential(
                torch.nn.Linear(in_channels, in_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(in_channels, in_channels),
            )
        elif method == "raw":
            self.raw_linear = torch.nn.Linear(1, in_channels)
        elif method == "llm":
            self.llm_layer = nn.Linear(384, in_channels)

        # use custom MPNN model
        self.layers = torch.nn.ModuleList()
        self.layers.append(MPNN(input_dim=in_channels, hidden_dim=hidden_channels*heads, num_layers=num_layers, edge_model=model, edge_dim=edge_embed_dim))
        for _ in range(num_layers-1):
            self.layers.append(MPNN(input_dim=hidden_channels*heads, hidden_dim=hidden_channels*heads, num_layers=num_layers, edge_model=model, edge_dim=edge_embed_dim))
        self.layers.append(nn.Linear(hidden_channels * heads, out_channels))
        
    def forward(self, data):
        if self.method == "stage":
            if data.edgegraph_edge_attr.dim() == 1:
                data.edgegraph_edge_attr = data.edgegraph_edge_attr.unsqueeze(-1)
            if self.model == "GCNConv":
                data.edgegraph_edge_attr = data.edgegraph_edge_attr[:, 0:1]
            h = self.edgegraph_model(
                data.edgegraph_x, data.edgegraph_edge_index, data.edgegraph_edge_attr
            )
            edge_embeddings = global_add_pool(h, data.edgegraph2ppedge)
            edge_embeddings = self.edge_embed_norm_layer(edge_embeddings)
            data.edge_embeddings = edge_embeddings
        elif self.method == "normalized":
            x = self.phi(data.x_normalized.unsqueeze(-1))
            x = x.mean(dim=1)
            data.x = self.rho(x)
        elif self.method == "raw":
            xs = []
            for i in range(data.x_raw.size(0)):
                x = self.raw_linear(data.x_raw[i, :].unsqueeze(-1))
                xs.append(x)
            data.x = torch.stack(xs,dim=1).sum(dim=1)
        elif self.method == "llm":
            data.x = self.llm_layer(data.llm_x)

        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv in self.layers[:-1]:
            if self.method == "stage":
                x = conv(x, edge_index, edge_attr=edge_embeddings)
            else:    
                x = conv(x, edge_index)

            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.layers[-1](x)
        
        return x


def train(data, model, optimizer, penalty_weight=0.1):
    model.train()
    optimizer.zero_grad()

    out = model(data)
    if model.out_channels == 1:
        loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask].float())
    else:
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(data, mask, model):
    model.eval()
    h = model(data)
    if model.out_channels == 1:
        pred = h.squeeze()
        mse_loss = F.mse_loss(pred[mask], data.y[mask].float()) / len(data.y[mask])
        return mse_loss.item()
    else:
        pred = h.argmax(dim=-1)
        return int((pred[mask] == data.y[mask]).sum()) / sum(mask)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_dataset", type=str)
    parser.add_argument("--test_dataset", type=str) 
    parser.add_argument("--input_dim", type=int, default=64)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--feature_method", type=str, default="stage")
    parser.add_argument("--edge_embed_dim", type=int, default=32, help="the edge_dim (for edge_attr) of GNN model")
    parser.add_argument("--edge_embed_num_layers", type=int, default=2, help="the number of layers for edge embedding model")
    parser.add_argument("--task", type=str, help="specify age to predict age regression")
    parser.add_argument("--model_name", type=str, default="GINEConv", help="choice of GNN model: GINEConv, GCNConv_edge")
    parser.add_argument("--wandb", action="store_true", help="Track experiment")
    args = parser.parse_args()

    device = torch.device(args.device)

    init_wandb(
        name=f"GINE-{args.train_dataset}-{args.test_dataset}",
        input_dim=args.input_dim,
        hidden_channels=args.hidden_channels,
        heads=args.heads,
        num_layers=args.num_layers,
        lr=args.lr,
        dropout=args.dropout,
        epochs=args.epochs,
        feature_method=args.feature_method,
        edge_embed_dim=args.edge_embed_dim,
        edge_embed_num_layers=args.edge_embed_num_layers,
    )

    train_data = torch.load(osp.join(osp.dirname(osp.realpath(__file__)), f"{args.train_dataset}.pt"))
    test_data = torch.load(osp.join(osp.dirname(osp.realpath(__file__)), f"{args.test_dataset}.pt"))
    
    if args.task == "age":
        # Extract the age column as the regression label
        train_data.y = train_data.x[:, 0]
        test_data.y = test_data.x[:, -1]

        # Remove the age column from the features
        train_data.x = train_data.x[:, 1:]
        test_data.x = test_data.x[:, :-1]

    if args.feature_method != "age_only":
        train_data = encode_input_features(train_data, args.feature_method, args.input_dim).to(device)
        test_data = encode_input_features(test_data, args.feature_method, args.input_dim).to(device)
    else:
        args.input_dim = 1
        train_data.to(device)
        test_data.to(device)

    if args.feature_method == "stage":
        train_data.x = torch.ones(train_data.x.size(0), args.input_dim).to(device)
        test_data.x = torch.ones(test_data.x.size(0), args.input_dim).to(device)
        model = GNN(
            args.input_dim,
            args.hidden_channels,
            1 if args.task == "age" else 2,
            args.heads,
            args.num_layers,
            args.feature_method,
            edge_embed_dim=args.edge_embed_dim, 
            edge_num_layers=args.edge_embed_num_layers,
            model=args.model_name,
            edge_dim=1,
            dropout_rate=args.dropout
        ).to(device)
    else:
        model = GNN(
            args.input_dim,
            args.hidden_channels,
            1 if args.task == "age" else 2,
            args.heads,
            args.num_layers,
            args.feature_method,
            args.dropout
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    times = []
    train_times = []
    inference_times = []
    best_val_acc = 10**6 if args.task == "age" else 0
    best_train_acc = 10**6 if args.task == "age" else 0
    best_train_epoch = 0
    best_test_acc = 10**6 if args.task == "age" else 0
    best_test_epoch = 0
    train_at_best_test = 0
    val_at_best_test = 0
    train_acc = 0
    val_acc = 0
    test_acc = 10**6 if args.task == "age" else 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        loss = train(train_data, model, optimizer)
        train_acc = test(train_data, train_data.train_mask, model)

        if args.task == "age":
            if train_acc < best_train_acc:
                best_train_acc = train_acc
                best_train_epoch = epoch
        else:
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                best_train_epoch = epoch
        
        val_acc = test(train_data, train_data.valid_mask, model)
        train_times.append(time.time() - start)
        inference_start = time.time()
        tmp_test_acc = test(test_data, test_data.test_mask, model)

        if args.task == "age":
            if tmp_test_acc < best_test_acc:
                best_test_acc = tmp_test_acc
                best_test_epoch = epoch
                train_at_best_test = train_acc
                val_at_best_test = val_acc
        else:
            if tmp_test_acc > best_test_acc:
                best_test_acc = tmp_test_acc
                best_test_epoch = epoch
                train_at_best_test = train_acc
                val_at_best_test = val_acc
        
        if args.task == "age":
            if val_acc < best_val_acc + 0.02:
                if val_acc < best_val_acc:
                    best_val_acc = val_acc
                if tmp_test_acc < test_acc:
                    test_acc = tmp_test_acc
        else:
            if val_acc > best_val_acc - 0.02:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                if tmp_test_acc > test_acc:
                    test_acc = tmp_test_acc
        inference_times.append(time.time() - inference_start)
        log(Epoch=epoch, 
            Loss=f"{loss:.3f}", 
            Train=f"{train_acc:.3f}", 
            Val=f"{val_acc:.3f}", 
            Test=f"{test_acc:.3f}")
        times.append(time.time() - start)
    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
    print(f"Average train time per epoch from 20th: {torch.tensor(train_times[20:]).mean():.4f}s")
    print(f"Average train time per epoch from 20th to 40th: {torch.tensor(train_times[20:40]).mean():.4f}s")
    print(f"Average inference time per epoch from 20th: {torch.tensor(inference_times[20:]).mean():.4f}s")
    print(f"Average inference time per epoch from 20th to 40th: {torch.tensor(inference_times[20:40]).mean():.4f}s")
    print(f"Best train accuracy: {best_train_acc:.4f}")
    print(f"Best train epoch: {best_train_epoch}")
    print(f"Best test accuracy: {best_test_acc:.4f}")
    print(f"Best test epoch: {best_test_epoch}")
    print(f"Train accuracy at best test: {train_at_best_test:.4f}")
    print(f"Val accuracy at best test: {val_at_best_test:.4f}")


if __name__ == "__main__":
    main()
