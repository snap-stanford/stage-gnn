from typing import List, Dict, Any

import torch
from torch_geometric.data import Batch
from torch_geometric.data import Data


from data_utils import (
    build_edge_graph,
    compute_probabilities,
    feature_grouping,
)

from tqdm import tqdm

def encode_input_features(data: Data,
                          feature_method: str,
                          input_dim: int,
                          ) -> Data:
    """
    Helper function to get the feature encoding method based on the feature_method
    """
    if feature_method == "stage":
        return EdgeGraphFeatureEncoder(data, input_dim)()
    elif feature_method == "structural":
        return StructuralFeatureEncoder(data, input_dim)()
    elif feature_method == "normalized":
        return NormalizedFeatureEncoder(data)()
    elif feature_method == "llm":
        return LLMFeatureEncoder(data)()
    elif feature_method == "raw":
        return RawFeatureEncoder(data)()
    elif feature_method == "gaussian":
        return GaussianFeatureEncoder(data, input_dim)()
    else:
        raise NotImplementedError("Feature encoding method not implemented yet")


class FeatureEncoder(object):
    """
    Base class for feature encoding methods
    """
    def __init__(self, data: Data, input_dim: int):
        super(FeatureEncoder, self).__init__()
        self.data = data
        self.input_dim = input_dim

    # empty encode function to be implemented by subclasses
    def encode(self):
        raise NotImplementedError("Feature encoding method not implemented yet")
    
    def __call__(self):
        return self.encode()
    

# STAGE's feature encoding method!
class EdgeGraphFeatureEncoder(FeatureEncoder):
    def __init__(self,
                 data: Data,
                 input_dim: int):

        super(EdgeGraphFeatureEncoder, self).__init__(data, input_dim)

    def get_edgegraph_data(self, 
                           data: Data,):
        
        """
        Get the edge graph data for the given dataset.
        Final """
        data, reduced_p_values = feature_grouping(data)
        marginal_prob, conditional_prob, internal_conditional_prob, pvalues = compute_probabilities(data)
        
        ## Construct edge graphs
        edge_graphs = []
        for idx in tqdm(range(data.edge_index.size(-1)), desc="Processing Edge Graphs", bar_format="{l_bar}{bar} [Time left: {remaining}]"):
            src, dst = data.edge_index[:, idx]
            src = src.item()
            dst = dst.item()
            edge_graphs.append(
                build_edge_graph(
                    data,
                    src,
                    dst,
                    marginal_prob,
                    conditional_prob,
                    internal_conditional_prob,
                    pvalues,
                )
            )
        return edge_graphs


    # Compute the edge graph data for each data in the data_dict
    def encode(self) -> Data:
        
        edge_graphs = self.get_edgegraph_data(self.data)

        # Store the edge graphs as disconnected components
        batch = Batch.from_data_list(edge_graphs)

        self.data.edgegraph2ppedge = batch.batch 
        self.data.edgegraph_x = batch.x
        self.data.edgegraph_edge_index = batch.edge_index
        self.data.edgegraph_edge_attr = batch.edge_attr
        self.data.x = torch.ones(self.data.x.size(0), self.input_dim)
        
        return self.data


class StructuralFeatureEncoder(FeatureEncoder):
    def __init__(self, 
                 data: Data,
                 input_dim: int):

        super(StructuralFeatureEncoder, self).__init__(data, input_dim)

    def encode(self) -> Data:
        """
        All ones feature encoding
        """

        self.data.x = torch.ones(self.data.x.size(0), self.input_dim)
        return self.data
    

class NormalizedFeatureEncoder(FeatureEncoder):
    def __init__(self, 
                 data: Data):

        self.data = data

    def encode(self) -> Data:
        """
        Normalized feature encoding based on identifying continuous features in .x
        """

        num_features = self.data.x.size(1)  # Number of features
        x_normalized = []

        for i in range(num_features):
            x = self.data.x[:, i]

            # Simple heuristic to identify continuous features:
            unique_values = torch.unique(x)
            if len(unique_values) > 10 and torch.is_floating_point(x):
                # Treat as continuous if there are more than 10 unique values and it's a floating-point tensor
                mask = torch.isinf(x) | torch.isnan(x) | (x == float('-inf'))
                valid_x = x[~mask]
                if valid_x.numel() > 1:  # Ensure there are at least two non-inf elements
                    mean = valid_x.mean()
                    std = valid_x.std()
                    if std > 0:
                        x = (x - mean) / std
                    else:
                        x = x - mean 
                else:
                    x = torch.zeros_like(x)  
                x[mask] = 0
            else:
                # Non-continuous feature, keep as is
                print(f"Feature {i} is not continuous, keeping original values.")
                
            x_normalized.append(x)

        if len(x_normalized) > 0:
            x_normalized = torch.stack(x_normalized, dim=1)  # Stack along the feature dimension
        else:
            print("Warning: No features found to normalize")
            x_normalized = torch.ones(self.data.x.size(0), 1)

        # Replace the original x with the normalized features
        self.data.x_normalized = x_normalized

        return self.data
    
class RawFeatureEncoder(FeatureEncoder):
    def __init__(self,
                 data: Data):
        
        self.data = data
            
    def encode(self) -> Data:
        """
        Raw feature encoding, but picking the extreme values for inf, -inf, and nan
        """
        num_features = self.data.x.size(1)
        x_raw = []

        for i in range(num_features):
            x = self.data.x[:, i]
            mask_inf = x == float('inf')
            mask_ninf = x == float('-inf')
            mask_nan = torch.isnan(x)
            mask = mask_inf | mask_ninf | mask_nan

            valid_x = x[~mask]
            if valid_x.numel() > 0:
                x[mask] = valid_x.max() + 1
            else:
                x[mask] = 1
            x_raw.append(x)
        
        x_raw = torch.stack(x_raw, dim=0)

        self.data.x_raw = x_raw
        return self.data

class GaussianFeatureEncoder(FeatureEncoder):
    def __init__(self, 
                 data,
                 input_dim: int):

        super(GaussianFeatureEncoder, self).__init__(data, input_dim)

    def encode(self) -> Data:
        """
        Gaussian feature encoding
        """
        
        self.data.x = torch.randn(self.data.x.size(0), self.input_dim)
        return self.data

class LLMFeatureEncoder(FeatureEncoder):
    def __init__(self, 
                 data: Data,):
        
        self.data = data
        self.model_name = 'all-MiniLM-L6-v2'

    def encode(self) -> Dict[str, Data]:
        """
        LLM feature encoding
        """

        x_as_a_string_list = []
        for i in range(self.data.x.size(0)):
            x_as_a_string = ""
            for j in range(self.data.x.size(1)):
                x_as_a_string += str(self.data.x[i, j].item()) + " "
            x_as_a_string_list.append(x_as_a_string)

        from sentence_transformers import SentenceTransformer

        text_model = SentenceTransformer(self.model_name)

        with torch.no_grad():
            emb = text_model.encode(x_as_a_string_list, show_progress_bar=True,
                               convert_to_tensor=True).cpu()
        
        emb = torch.tensor(emb)
        self.data.llm_x = emb

        return self.data