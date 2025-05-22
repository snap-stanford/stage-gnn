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
import torch.nn.functional as F

from tqdm import tqdm
from causallearn.utils.cit import CIT
import networkx as nx
import copy

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
            if category != None:
                if row[category_column] != category:
                    continue
            if row[product_id_column] not in product_feature_map:
                continue
            if len(feature_maps) == 2:
                if row[user_id_column] not in user_feature_map:
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

# This function is optional.
# Perform feature grouping to obtain a disjoint set of features, where each set contains feature that are mutually correlated.
# We then extract one "representative" features from each disjoint set to be used for downstream processing.
# Since most pairs of features are highly correlated, this should significantly reduce the size of downsteam edgegrpahs.
# This function performs the following main tasks:
#   1. Delete features who has 0 std (because they are constant features which are useless)
#   2. Perform pairwise independence test to obtain a square matrix of p-values
#   3. Build a correlation graph, where each node is a feature, and 2 nodes are connected iff their p-value is less than 
#      a pre-defined significance leve threshold
#   4. Iteratively extract maximal cliques from this correlation graph. Each clique will be a disjoint set of features.
#   5. Decide a representative feature from each disjoint set based on some criteria (e.g. the feature with the highest std)
#      and modify the data object accordingly.
def feature_grouping(data, alpha=0.05):
    feat_dim = data.x.shape[1]
    feature_keys = dict(zip(list(range(feat_dim)), ["dummy"] * feat_dim))
    n_original_features = len(feature_keys)
    feature_keys_list = list(feature_keys.keys())

    # Preprocessing - replace inf values with 0
    # convert data.x to numpy array data_obj
    data_obj = copy.deepcopy(data.x)
    data_obj[torch.isinf(data_obj) | torch.isnan(data_obj)] = 0
    data_obj = data_obj.numpy()

    # Step 1: Delete features with 0 std
    # feature_stds = [np.std(data_obj[:, i]) for i in range(data_obj.shape[1])]
    feature_stds = {feature: np.std(data_obj[:, i]) for i, feature in enumerate(feature_keys_list)}
    useless_features = [feature for feature in feature_keys_list if feature_stds[feature] == 0]
    print(f"Useless features: {useless_features}")
    print(f"Number of useless features: {len(useless_features)}")

    # Remove these useless features columns from the data.x
    data.x = data.x[:, [i for i in range(data.x.shape[1]) if i not in useless_features]]
    return data, None
    for feature in useless_features:
        del feature_keys[feature]
    data_obj = np.delete(data_obj, [feature_keys_list.index(f) for f in useless_features], axis=1)

    # Step 2: Perform pairwise independence test
    n_good_features = len(feature_keys)
    idx2feature = {i: feature for i, feature in enumerate(feature_keys.keys())}
    p_values = np.zeros((len(feature_keys), len(feature_keys)))
    n_iterations = 1  # Number of iterations to compute p-values
    for _ in tqdm(range(n_iterations)):
        # Randomly sample 100 entries from each feature
        indices = np.random.choice(data_obj.shape[0], 100, replace=False)
        kci_obj = CIT(data_obj[indices], "kci")
        for i, feature1 in enumerate(feature_keys):
            for j, feature2 in enumerate(feature_keys):
                if i != j:
                    # Calculate the p-value for the combined sample
                    p_value = kci_obj(i, j)
                    if np.isnan(p_value):
                        # print("NAN encountered:", feature1, feature2)
                        p_values[i, j] += 1
                    p_values[i, j] += p_value
                else:
                    p_values[i, j] += 1
    # average the p-values
    p_values /= n_iterations
    print(f"Average of p-values: {np.mean(p_values)}")
    print(f"Max of p-values: {np.max(p_values)}")
    print(f"Min of p-values: {np.min(p_values)}")
    print(f"Std of p-values: {np.std(p_values)}")
    print(f"Median of p-values: {np.median(p_values)}")

    # Step 3: Build correlation graph
    G = nx.Graph()
    G.add_nodes_from(range(n_good_features))
    for i in tqdm(range(n_good_features)):
        for j in range(i + 1, n_good_features):
            if p_values[i, j] < alpha:
                G.add_edge(i, j)

    # Step 4: Iteratively, Find the maximal cliques, delete these nodes from the graph, and repeat until the graph is empty
    disjoint_sets = []
    while G.number_of_nodes() > 0:
        next_clique = next(nx.find_cliques(G))
        disjoint_sets.append(next_clique)
        G.remove_nodes_from(next_clique)
        # print(f"Found a disjoint set of size {len(next_clique)}, {G.number_of_nodes()} nodes left")

    # Step 5: Decide a representative feature from each disjoint set
    repr_features, repr_feature_ids = [], []
    for disjoint_set in disjoint_sets:
        stds = [feature_stds[idx2feature[i]] for i in disjoint_set]
        max_std_idx = np.argmax(stds)
        feature_id = disjoint_set[max_std_idx]
        repr_feature_ids.append(feature_id)
        repr_features.append(idx2feature[feature_id])
    print(f"Number of original features: {n_original_features} is reduced to {len(repr_features)} after feature grouping")
    # Modify the data by picking only the representative features columns
    data.x = data.x[:, repr_features]

    # Get the reduced p-value matrix for the representative features. Also convert to the same format for downstream use
    reduced_p_values = {}
    for i, feature1 in tqdm(enumerate(repr_features)):
        for j, feature2 in enumerate(repr_features):
            reduced_p_values[(feature1, feature2)] = p_values[repr_feature_ids[i], repr_feature_ids[j]]
    return data, reduced_p_values


def compute_probabilities(data):
    feat_dim = data.x.shape[1]
    features_keys = dict(zip(list(range(feat_dim)), ["continuous"] * feat_dim))
    vect_marginal_probabilities = {}

    for feature in features_keys:
        uniques, counts = data.x[:, feature].unique(return_counts=True)        
        probs = counts.float() / data.x.shape[0]        
        vect_marginal_probabilities[feature] = dict(zip(uniques.tolist(), probs.tolist()))

    conditional_probabilities = {}
    internal_conditional_probabilities = {}
    for i in range(data.edge_index.size(-1)):
        u, v = data.edge_index[0, i], data.edge_index[1, i]
        for feature1 in features_keys:
            for feature2 in features_keys:
                if torch.isinf(data.x[u][feature1]) or torch.isinf(data.x[v][feature2]):
                    continue

                key = (
                    feature1,
                    data.x[u][feature1].item(),
                    feature2,
                    data.x[v][feature2].item(),
                )
                conditional_probabilities[key] = (
                    conditional_probabilities.get(key, 0) + 1
                )

    for i in range(data.num_nodes):
        for feature1 in features_keys:
            for feature2 in features_keys:
                if torch.isinf(data.x[i][feature1]) or torch.isinf(data.x[i][feature2]):
                    continue

                key = (
                    feature1,
                    data.x[i][feature1].item(),
                    feature2,
                    data.x[i][feature2].item(),
                )
                internal_conditional_probabilities[key] = (
                    internal_conditional_probabilities.get(key, 0) + 1
                )

    # Normalize conditional probabilities
    for key in conditional_probabilities:
        conditional_probabilities[key] /= data.edge_index.size(-1)

    total_feature_nodes = data.x.shape[0]
    for key in internal_conditional_probabilities:
        internal_conditional_probabilities[
            key
        ] /= total_feature_nodes 

    return (
        vect_marginal_probabilities,
        conditional_probabilities,
        internal_conditional_probabilities,
        None,
    )


def build_edge_graph(
    data, u, v, marginal_prob, conditional_prob, internal_conditional_prob, pvalues
):
    X = data.x
    X = torch.tensor(X, dtype=torch.float)
    num_nodes, dim = X.shape[0], X.shape[1]

    # Extract marginal probabilities for each feature from marginal_prob for both nodes u and v
    x = torch.tensor([
       marginal_prob[feat_dim][feat_val.item()] for node in [u, v] for feat_dim, feat_val in enumerate(X[node])
    ])

    # Initialize lists for edge indices and edge attributes
    edge_index = []
    edge_attr = []

    # Iterate through all pairs of features between node u and node v (cross-node features)
    for feat_u, val_u in enumerate(X[u]):
        for feat_v, val_v in enumerate(X[v]):
            # Get conditional probability between feature u and feature v
            prob = conditional_prob.get((feat_u, val_u.item(), feat_v, val_v.item()), 0)
            edge_index.append([feat_u, feat_v + dim])  # Cross-node edge
            if pvalues is not None:
                edge_attr.append([prob, pvalues.get((feat_u, feat_v), 0)])
            else:
                edge_attr.append([prob])

    # Iterate through all pairs of features within node u (within-node features for u)
    for feat1, val1 in enumerate(X[u]):
        for feat2, val2 in enumerate(X[u]):
            if feat1 != feat2:  # Ensure it's not a diagonal feature (i.e., not self-loops)
                # Get internal conditional probability
                prob = internal_conditional_prob.get((feat1, val1.item(), feat2, val2.item()), 0)
                edge_index.append([feat1, feat2])  # Within-node edge for u
                if pvalues is not None:
                    edge_attr.append([prob, pvalues.get((feat1, feat2), 0)])
                else:
                    edge_attr.append([prob])

    # Iterate through all pairs of features within node v (within-node features for v)
    for feat1, val1 in enumerate(X[v]):
        for feat2, val2 in enumerate(X[v]):
            if feat1 != feat2:  # Ensure it's not a diagonal feature (i.e., not self-loops)
                # Get internal conditional probability
                prob = internal_conditional_prob.get((feat1, val1.item(), feat2, val2.item()), 0)
                edge_index.append([feat1 + dim, feat2 + dim])  # Within-node edge for v
                if pvalues is not None:
                    edge_attr.append([prob, pvalues.get((feat1, feat2), 0)])
                else:
                    edge_attr.append([prob])

    # Convert edge_index and edge_attr to tensors
    edge_index = torch.tensor(edge_index).T
    edge_attr = torch.tensor(edge_attr).float()

    # Return the data object
    return Data(
        x=x.view(-1, 1),  # Marginal probabilities
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
