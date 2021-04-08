# DGL Implementation of GRACE
This DGL example implements the model proposed in the paper [Deep Graph Contrastive Representation Learning](https://arxiv.org/abs/2006.04131).

Author's code: https://github.com/CRIPAC-DIG/GRACE

## Example Implementor

This example was implemented by [Hengrui Zhang](https://github.com/hengruizhang98) when he was an applied scientist intern at AWS Shanghai AI Lab.

## Dependencies

- Python 3.7
- PyTorch 1.7.1
- dgl 0.6.0

## Datasets

##### Unsupervised Node Classification Datasets:

'Cora', 'Citeseer' and 'Pubmed'

| Dataset  | # Nodes | # Edges | # Classes |
| -------- | ------- | ------- | --------- |
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |


## Arguments

##### 	Node Classification:

```
--dataname         str     The graph dataset name.                Default is 'cora'.
--gpu              int     GPU index.                             Default is 0.
--split            int     Dataset spliting method.               Default is 'random'.
```

## How to run examples

###### Node Classification

For semi-supervised node classification on 'Cora', 'Citeseer' and 'Pubmed', we provide two implementations:

1. full-graph training, see 'main.py', where we contrast the local and global representations of the whole graph.
2. subgraph training, see 'main_sample.py', where we contrast the local and global representations of a sampled subgraph with fixed number of nodes.

```python
# Cora with random split
python main.py --dataname cora

# Cora with public split
python main.py --dataname cora --split public
```

replace 'cora' with 'citeseer' or 'pubmed' if you would like to run this example on other datasets.

## 	Performance

We use the same hyper-parameter settings as provided by the author, you can check config.yaml for detailed hyper-parameters for each dataset.

##### Node classification :

Random split

|      Dataset      | Cora | Citeseer | Pubmed |
| :---------------: | :--: | :------: | :----: |
| Accuracy Reported | 83.3 |   72.1   |  86.7  |
|   Author's Code   | 83.1 |   71.0   |  86.3  |
|        DGL        | 83.4 |   71.4   |  86.1  |

Public split

|    Dataset    | Cora | Citeseer | Pubmed |
| :-----------: | :--: | :------: | :----: |
| Author's Code | 79.9 |   68.6   |  81.3  |
|      DGL      | 80.1 |   68.9   |  81.2  |

* We use the same hyper-parameters as the random split setting, so you might get better performance with this model by fine-tuning the hyper-parameters.

