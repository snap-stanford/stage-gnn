import torch
from torch_geometric.data import Data
import pandas as pd
from torch_geometric.utils import (
    to_scipy_sparse_matrix,
    to_torch_sparse_tensor,
    subgraph,
)
import numpy as np
import scipy.sparse as sp
import ast

from causallearn.utils.cit import CIT

def get_feature_map(csv_file_path, unwanted_columns=[]):
    df = pd.read_csv(csv_file_path)
    df = df.drop(columns=unwanted_columns[1:])
    column_names = df.columns.tolist()
    data_types = {}
    for column in df.columns:
        if df[column].iloc[0] == True or df[column].iloc[0] == False:
            data_types[column] = 'discrete'
        else:
            try:
                float(df[column].iloc[0])
                data_types[column] = 'continuous'
            except ValueError:
                data_types[column] = 'discrete'

    feature_map = {}
    for _, row in df.iterrows():
        id_column = unwanted_columns[0]
        id = row[id_column]
        if id_column == "product_id":
            for product_id in ast.literal_eval(id):
                feature_map[product_id] = row.drop(id_column)
        else:
            feature_map[id] = row.drop(id_column)
    data_types.pop(unwanted_columns[0])
    return feature_map, data_types


def get_user_product_graph(csv_file_path, start_row, end_row, category, feature_maps, feature_keys, category_column, product_id_column, user_id_column, event_type_column=None, event_type_mapping=None):
    df_chunks = pd.read_csv(
        csv_file_path, chunksize=100000
    )  # Adjust chunksize based on your memory capacity

    user2node = {}  # from user id to node id
    product2node = {}  # from product id to node id
    edge_index = []
    edge_attr = []

    if len(feature_maps) == 2:
        user_feature_map, product_feature_map = feature_maps
        user_feature_keys, product_feature_keys = feature_keys
    else:
        product_feature_map = feature_maps[0]
        product_feature_keys = feature_keys[0]

    # initialize each key in feature_keys according to its value
    x_map = {}
    x = {}

    all_feature_keys = product_feature_keys.copy()
    if len(feature_maps) == 2:
        all_feature_keys.update(user_feature_keys)
    for feature_name in all_feature_keys:
        if all_feature_keys[feature_name] == "discrete":
            x_map[feature_name] = {}
        x[feature_name] = []
    
    # Process each chunk
    count = 0
    break_flag = False
    for df in df_chunks:
        if break_flag:
            break
        for _, row in df.iterrows():
            if event_type_column is not None:
                if row[event_type_column] not in event_type_mapping:
                    continue
            if row[product_id_column] not in product_feature_map:
                continue
            if len(feature_maps) == 2:
                if row[user_id_column] not in user_feature_map:
                    continue
            if category != None:
                if category_column not in row:
                    product_category = product_feature_map[row[product_id_column]][category_column]
                else:
                    product_category = row[category_column]
                if product_category != category:
                    continue

            count += 1
            if count < start_row:
                continue
            if count > end_row:
                break_flag = True
                break
            user_id = row[user_id_column]
            product_id = row[product_id_column]

            for feature_name in product_feature_keys:
                # they are all products' features
                if user_id not in user2node:
                    x[feature_name].append(float("inf")) # for user node
                if product_id not in product2node:
                    if product_feature_keys[feature_name] == "continuous":
                        x[feature_name].append(product_feature_map[product_id][feature_name])
                    else:
                        feature_value = product_feature_map[product_id][feature_name]
                        if feature_value not in x_map[feature_name]:
                            x_map[feature_name][feature_value] = len(x_map[feature_name])
                        x[feature_name].append(x_map[feature_name][feature_value])
            
            
            if len(feature_maps) == 2:
                for feature_name in user_feature_keys:
                    # they are all users' features
                    if user_id not in user2node:
                        if user_feature_keys[feature_name] == "continuous":
                            x[feature_name].append(user_feature_map[user_id][feature_name])
                        else:
                            feature_value = user_feature_map[user_id][feature_name]
                            if feature_value not in x_map[feature_name]:
                                x_map[feature_name][feature_value] = len(x_map[feature_name])
                            x[feature_name].append(x_map[feature_name][feature_value])
                    if product_id not in product2node:
                        x[feature_name].append(float("inf"))

            if user_id not in user2node:
                user2node[user_id] = len(product2node) + len(user2node)
            if product_id not in product2node:
                product2node[product_id] = len(product2node) + len(user2node)

            user_node = user2node[user_id]
            product_node = product2node[product_id]

            # only from user to product, the reverse will be added later
            edge_index.append([user_node, product_node])
            if event_type_column is not None:
                edge_attr.append(event_type_mapping[row[event_type_column]])
            else:
                edge_attr.append(0)

    # Convert to tensors
    edge_index_tensor = torch.tensor(edge_index).t()
    edge_attr_tensor = torch.tensor(edge_attr)
    x = {feature_name: torch.tensor(x[feature_name], dtype=torch.float) for feature_name in x}

    data = Data(
        x=x,
        edge_index=edge_index_tensor,
        edge_attr=edge_attr_tensor,
        num_nodes=len(user2node) + len(product2node),
    )

    return data, user2node, product2node, all_feature_keys, x_map


def load_max_connected(data, user2node, product2node):
    assert data.edge_index is not None

    # Convert to scipy sparse matrix
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)

    # Find connected components
    num_components, component_labels = sp.csgraph.connected_components(
        adj, connection="weak"
    )

    if num_components <= 1:
        return (
            data,
            user2node,
            product2node,
        )  # Return original mappings if only one component

    # Find the largest component
    _, counts = np.unique(component_labels, return_counts=True)
    largest_component_label = np.argmax(counts)
    subset_np = component_labels == largest_component_label
    subset = torch.from_numpy(subset_np).to(data.edge_index.device, dtype=torch.bool)

    # Create a subgraph with only the largest connected component
    sub_data = data.subgraph(subset)

    # Subset the node features
    if data.x is not None:
        sub_data.x = {k: v[subset] for k, v in data.x.items()}


    # Update user and product node mappings
    node_idx_mapping = {
        old_idx.item(): i for i, old_idx in enumerate(torch.where(subset)[0])
    }
    new_user2node = {
        user: node_idx_mapping[node_id]
        for user, node_id in user2node.items()
        if node_id in node_idx_mapping
    }
    new_product2node = {
        product: node_idx_mapping[node_id]
        for product, node_id in product2node.items()
        if node_id in node_idx_mapping
    }
    return sub_data, new_user2node, new_product2node


# Build a graph with product-product edges to compute product-product conditional probabilities
def build_product_product_graph(data, up_product2node):

    adj = to_torch_sparse_tensor(data.edge_index, size=(data.num_nodes, data.num_nodes))
    twohopsadj = adj.T @ adj

    edge_index = []

    for p1 in up_product2node.keys():
        for p2 in up_product2node.keys():

            if p1 == p2:
                continue
            for r in range(
                twohopsadj[up_product2node[p1], up_product2node[p2]].int().item()
            ):
                edge_index.append([up_product2node[p1], up_product2node[p2]])

    return Data(
        x=data.x,
        edge_index=torch.tensor(edge_index).T,
        num_nodes=data.num_nodes,
    )


def compute_probabilities(data, features_keys):

    feature_values = dict.fromkeys(
        features_keys.keys(), None
    )  # Store distinct values for each feature
    marginal_probabilities = {}  # P(F_A = a)
    conditional_probabilities = {}  # P(F_{A|N} = a | F_{B|C} = b)
    internal_conditional_probabilities = {}  # P(F_{A|C} = a | F_{B|C} = b)

    for feature in features_keys:
        feature_values[feature] = set(data.x[feature].tolist())

    total_feature_nodes = 0

    # Compute marginal probabilities
    for feature in feature_values:
        mask = ~torch.isinf(data.x[feature])
        total_feature_nodes = torch.sum(mask).item()

        marginal_probabilities[feature] = {}
        if (
            features_keys[feature] == "continuous"
        ):
            for i, value in enumerate(feature_values[feature]):
                count = (data.x[feature] <= value).sum().item()
                marginal_probabilities[feature][value] = (
                    count / total_feature_nodes
                )
        else:
            for value in feature_values[feature]:
                count = (data.x[feature] == value).sum().item()
                marginal_probabilities[feature][value] = (
                    count / total_feature_nodes
                )

    for i in range(data.edge_index.size(-1)):
        u, v = data.edge_index[0, i], data.edge_index[1, i]
        for feature1 in features_keys:
            for feature2 in features_keys:
                if torch.isinf(data.x[feature1][u]) or torch.isinf(data.x[feature2][v]):
                    continue

                key = (
                    feature1,
                    data.x[feature1][u].item(),
                    feature2,
                    data.x[feature2][v].item(),
                )
                conditional_probabilities[key] = (
                    conditional_probabilities.get(key, 0) + 1
                )

    for i in range(data.num_nodes):
        for feature1 in features_keys:
            for feature2 in features_keys:
                if torch.isinf(data.x[feature1][i]) or torch.isinf(data.x[feature2][i]):
                    continue

                key = (
                    feature1,
                    data.x[feature1][i].item(),
                    feature2,
                    data.x[feature2][i].item(),
                )
                internal_conditional_probabilities[key] = (
                    internal_conditional_probabilities.get(key, 0) + 1
                )

    # Normalize conditional probabilities
    for key in conditional_probabilities:
        conditional_probabilities[key] /= data.edge_index.size(-1)

    for key in internal_conditional_probabilities:
        internal_conditional_probabilities[
            key
        ] /= total_feature_nodes  # assume all featured nodes have non-inf values in all features

    pvalues = {}

    data_obj = torch.stack(list(data.x.values()), dim=-1)
    mask = ~(torch.isinf(data_obj).sum(-1).bool())
    data_obj = data_obj[mask].numpy()

    kci_obj = CIT(data_obj, "kci")
    for i, feature1 in enumerate(features_keys):
        for j, feature2 in enumerate(features_keys):
            if np.std(data_obj[:, i]) == 0 or np.std(data_obj[:, j]) == 0:
                # Assign a p-value of 1 to reflect independence assumption
                pvalues[(feature1, feature2)] = 1
            else:
                try:
                    pvalues[(feature1, feature2)] = kci_obj(i, j)
                    if np.isnan(pvalues[(feature1, feature2)]):
                        pvalues[(feature1, feature2)] = 1
                except ValueError:
                    pvalues[(feature1, feature2)] = 0
    return (
        marginal_probabilities,
        conditional_probabilities,
        internal_conditional_probabilities,
        pvalues,
    )


def build_edge_graph(
    data, u, v, marginal_prob, conditional_prob, internal_conditional_prob, pvalues
):
    x = []
    edge_index = []
    edge_attr = []
    node2tuple = {}
    for i in [u, v]:
        for feature in data.x.keys():
            value = data.x[feature][i].item()
            tuple = (feature, value, i)
            prob = marginal_prob[feature].get(value, 0)
            x.append(prob)
            node2tuple[len(node2tuple)] = tuple

    for node1 in node2tuple:
        feature1, value1, product1 = node2tuple[node1]
        for node2 in node2tuple:
            feature2, value2, product2 = node2tuple[node2]
            prob = 0
            if product1 == product2:
                prob = internal_conditional_prob.get(
                    (feature1, value1, feature2, value2), 0
                )
            else:
                prob = conditional_prob.get(
                    (feature1, value1, feature2, value2), 0
                )
            edge_index.append([node1, node2])
            if pvalues is not None:
                edge_attr.append([prob, pvalues[feature1, feature2]])
            else:
                edge_attr.append([prob])

    return Data(
        x=torch.tensor(x).view(-1, 1),
        edge_index=torch.tensor(edge_index).T,
        edge_attr=torch.tensor(edge_attr).float(),
    )
