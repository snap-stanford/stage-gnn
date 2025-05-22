# STAGE: Zero-Shot Generalization of GNNs over Distinct Attribute Domains #

## Overview ##

STAGE (Statistical Transfer for Attributed Graph Embeddings) is a novel Graph Neural Network framework that addresses the challenge of zero-shot generalization across graphs with different node attribute domains. Unlike traditional GNNs that struggle with new node attributes, STAGE encodes statistical dependencies between attributes rather than their specific values, allowing it to generalize to unseen attribute spaces.

![STAGE](/asset/STAGE.png)

This codebase is based on PyG re-implementation of NBFNet. The entire repository contains the implementation of STAGE along with experiments and benchmarks to facilitate further research in cross-domain graph learning.

## Installation ##

Here is the instruction to install the dependencies via pip. Generally, NBFNet works with Python >= 3.7 and PyTorch >= 1.8.0.

```bash
conda create -n stage python=3.9.19
conda activate stage
pip install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter==2.1.2+pt22cu121 torch-geometric==2.5.2 -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
pip install ninja easydict pyyaml pandas causal-learn sentence-transformers==2.7.0 transformers==4.41.0
```

## Reproduction ##

To reproduce the results of STAGE, follow the instruction below.

### Link Prediction ###

1. Download Dataset CSV (ecommerce transaction)

- Download "2019-Nov.csv" from https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store and place it under "link_prediction/data/ecommerce/"
- Download "transaction_train.csv" from https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data and place it under "link_prediction/data/hm/"

2. Modify Config

- To change the training domain and the test domain: modify `train_categories` and `test_categories`
- To change the feature processing method to STAGE or other baselines: modify `feature_method`

3. Run Experiment

```bash
cd link_prediction
python run.py -c config.yaml
```

### Node Classification ###

1. Choose Dataset

- To use all processed features in Friendster and Pokec social network dataset: use "friendster.pt" and "pokec.pt"
- To only use the "age" feature: use "friendster_age_only.pt" and "pokec_age_only.pt"

2. Run Experiment

For example, to use all processed features:

```bash
cd node_classification
python gnn.py --train_dataset friendster --test_dataset pokec
```

## Citation ##

If you find this code useful in your research, please cite our paper:

```bibtex
@inproceedings{shen2025zeroshotgnn,
  title={Zero-Shot Generalization of GNNs over Distinct Attribute Domains},
  author={Shen, Yangyi and Zhou, Jincheng and Bevilacqua, Beatrice and Robinson, Joshua and Kanatsoulis, Charilaos and Leskovec, Jure and Ribeiro, Bruno},
  booktitle={Forty-Second International Conference on Machine Learning},
  year={2025},
  month = jun
}
```
