from typing import List, Dict, Any

import torch
from torch_geometric.data import Batch
from torch_geometric.data import Data


from .data_utils import (
    build_edge_graph,
    compute_probabilities,
)



def encode_input_features(data_dict: Dict[str, Data],
                          feature_method: str,
                          feat_metadata: Dict[str, Any],
                          input_dim: int,
                          ) -> Dict[str, Data]:
    """
    Helper function to get the feature encoding method based on the feature_method
    """
    if feature_method == "stage":
        return EdgeGraphFeatureEncoder(data_dict, feat_metadata)()
    elif feature_method == "structural":
        return StructuralFeatureEncoder(data_dict)()
    elif feature_method == "normalized":
        return NormalizedFeatureEncoder(data_dict, feat_metadata)()
    elif feature_method == "llm":
        return LLMFeatureEncoder(data_dict, feat_metadata)()
    elif feature_method == "raw":
        return RawFeatureEncoder(data_dict, feat_metadata)()
    elif feature_method == "gaussian":
        return GaussianFeatureEncoder(data_dict, input_dim)()
    elif feature_method == "price":
        return PriceFeatureEncoder(data_dict, feat_metadata)()
    else:
        raise NotImplementedError("Feature encoding method not implemented yet")


class FeatureEncoder(object):
    """
    Base class for feature encoding methods
    """
    def __init__(self, data_dict: Dict[str, Data], **kwargs):
        super(FeatureEncoder, self).__init__()
        self.data_dict = data_dict

    # empty encode function to be implemented by subclasses
    def encode(self):
        raise NotImplementedError("Feature encoding method not implemented yet")
    
    def __call__(self):
        return self.encode()
    

# STAGE's feature encoding method!
class EdgeGraphFeatureEncoder(FeatureEncoder):
    def __init__(self,
                 data_dict: Dict[str, Data],
                 feat_metadata: Dict[str, Any]):

        super(EdgeGraphFeatureEncoder, self).__init__(data_dict)
        self.data_dict = data_dict
        self.data_for_computing_edge_graphs = feat_metadata["data_for_computing_edge_graphs"]
        self.feature_keys = feat_metadata["feature_keys"]
        self.pvalue = feat_metadata["p_value"]


    def get_edgegraph_data(self, 
                           data: Data, 
                           feature_keys: List[str], 
                           pvalue: bool) -> List[Data]:
        
        """
        Get the edge graph data for the given dataset.
        Final """
        marginal_prob, conditional_prob, internal_conditional_prob, pvalues = compute_probabilities(data, feature_keys)
        
        if not pvalue:
            pvalues = None
        
        ## Construct edge graphs
        edge_graphs = []
        for idx in range(data.edge_index.size(-1)):
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
    def encode(self) -> Dict[str, Data]:
        
        edge_graphs = self.get_edgegraph_data(self.data_for_computing_edge_graphs, self.feature_keys, self.pvalue)

        data_dict = self.data_dict.copy()

        # Store the edge graphs as disconnected components
        batch = Batch.from_data_list(edge_graphs)

        # Add the edge graph data to all the Data in the data_dict
        for key in data_dict.keys():
            data_dict[key].edgegraph2ppedge = batch.batch
            data_dict[key].edgegraph_x = batch.x
            data_dict[key].edgegraph_edge_index = batch.edge_index
            data_dict[key].edgegraph_edge_attr = batch.edge_attr

            data_dict[key].x = None # Remove the original node features

        return data_dict


class StructuralFeatureEncoder(FeatureEncoder):
    def __init__(self, 
                 data_dict: Dict[str, Data]):

        super(StructuralFeatureEncoder, self).__init__(data_dict)

    def encode(self) -> Dict[str, Data]:
        """
        All ones feature encoding
        """

        data_dict = self.data_dict.copy()
        for key in data_dict.keys():
            data_dict[key].x = torch.ones(data_dict[key].num_nodes)
        return data_dict
    
class PriceFeatureEncoder(FeatureEncoder):
    def __init__(self,
                 data_dict: Dict[str, Data],
                 feat_metadata: Dict[str, Any]):
        self.data_dict = data_dict
        self.up_data = feat_metadata["up_data"]
        
        super(PriceFeatureEncoder, self).__init__(data_dict)
    
    def encode(self) -> Dict[str, Data]:
        """
        only use price feature for encoding
        """
        
        price = self.up_data.x["price"]

        data_dict = self.data_dict.copy()
        for key in data_dict.keys():
            data_dict[key].x_price = price


        return data_dict
    
class NormalizedFeatureEncoder(FeatureEncoder):
    def __init__(self, 
                 data_dict: Dict[str, Data],
                 feat_metadata: Dict[str, Any]):
        
        self.data_dict = data_dict
        self.up_data = feat_metadata["up_data"]
        self.feature_keys = feat_metadata["feature_keys"]

        super(NormalizedFeatureEncoder, self).__init__(data_dict)

    def encode(self) -> Dict[str, Data]:
        """
        Normalized feature encoding
        """

        continous_features = [key for key, type in self.feature_keys.items() if type == "continuous"]

        x_normalized = []
        for key in continous_features:
            x = self.up_data.x[key]

            mask = torch.isinf(x) | torch.isnan(x) | (x == float('-inf'))
            valid_x = x[~mask]
            if valid_x.numel() > 1:  # Ensure there are at least two non-inf elements
                mean = valid_x.mean()
                std = valid_x.std()
                if std > 0:
                    x = (x - mean) / std
                else:
                    x = x - mean  # or assign zeros or another placeholder value if std is zero
            else:
                x = torch.zeros_like(x)  # or use a different placeholder value if there are no valid entries
            x[mask] = 0

            x_normalized.append(x)

        if len(x_normalized) > 0:
            x_normalized = torch.stack(x_normalized, dim=0)
        else:
            print("Warning: No continuous features found")
            x_normalized = torch.ones(1, self.up_data.num_nodes)

        data_dict = self.data_dict.copy()
        for key in data_dict.keys():
            data_dict[key].x_normalized = x_normalized


        return data_dict
    
class RawFeatureEncoder(FeatureEncoder):
    def __init__(self,
                 data_dict: Dict[str, Data],
                 feature_metadata: Dict[str, Any]):
        
        self.data_dict = data_dict
        self.up_data = feature_metadata["up_data"]
        
        super(RawFeatureEncoder, self).__init__(data_dict)
    
    def encode(self) -> Dict[str, Data]:
        """
        Raw feature encoding
        """
        x_raw = []
        for key in self.up_data.x.keys():
            x = self.up_data.x[key]
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

        data_dict = self.data_dict.copy()
        for key in data_dict.keys():
            data_dict[key].x_raw = x_raw


        return data_dict

class GaussianFeatureEncoder(FeatureEncoder):
    def __init__(self, 
                 data_dict: Dict[str, Data],
                 input_dim: int):

        self.data_dict = data_dict
        self.input_dim = input_dim

        super(GaussianFeatureEncoder, self).__init__(data_dict)

    def encode(self) -> Dict[str, Data]:
        """
        Gaussian feature encoding
        """

        data_dict = self.data_dict.copy()
        for key in data_dict.keys():
            data_dict[key].x_gaussian = torch.randn(data_dict[key].num_nodes, self.input_dim)
        return data_dict

class LLMFeatureEncoder(FeatureEncoder):
    def __init__(self, 
                 data_dict: Dict[str, Data],
                 feat_metadata: Dict[str, Any]):
        
        self.data_dict = data_dict
        self.up_data = feat_metadata["up_data"]
        self.feature_keys = feat_metadata["feature_keys"]


        self.cat2name = {}
        for key in feat_metadata["x_map"].keys():
            feat_dict = feat_metadata["x_map"][key]

            self.cat2name[key] = {v: k for k, v in feat_dict.items()}

        self.model_name = 'all-MiniLM-L6-v2'

        super(LLMFeatureEncoder, self).__init__(data_dict)

    def encode(self) -> Dict[str, Data]:
        """
        LLM feature encoding
        """

        x_as_a_string_list = []
        for i in range(self.up_data.num_nodes):
            x_as_a_string = ""
            for feat_name, feat_type in self.feature_keys.items():
                if feat_type == 'continuous':
                    val =  str(self.up_data.x[feat_name][i].item())
                    if val == "inf":
                        val = "NONE"
                else:
                    cat_id = self.up_data.x[feat_name][i].item()
                    if cat_id not in self.cat2name[feat_name]:
                        val = "NONE"
                    else:
                        val = str(self.cat2name[feat_name][cat_id])

                x_as_a_string += str(feat_name) + ": " + val + ". "
            x_as_a_string_list.append(x_as_a_string)

        from sentence_transformers import SentenceTransformer
        
        text_model = SentenceTransformer(self.model_name)

        with torch.no_grad():
            emb = text_model.encode(x_as_a_string_list, show_progress_bar=True,
                               convert_to_tensor=True).cpu()
        
        emb = torch.tensor(emb)

        data_dict = self.data_dict.copy()
        for key in data_dict.keys():
            data_dict[key].x_emb = emb

        return data_dict