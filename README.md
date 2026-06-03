# Graph Structure Classification

This project investigates how different machine learning methods perform in classifying graph structures. The goal was to compare a traditional feature-based approach (Random Forest) with Graph Neural Networks (GCN and GAT).

The models were trained to classify graphs generated using:

- Watts-Strogatz
- Barabasi-Albert
- Random Regular

Graph sizes were varied and additional edge noise was introduced to create a more challenging classification task.

## Methods

The following models were implemented and evaluated:

- Random Forest
- Graph Convolutional Network (GCN)
- Graph Attention Network (GAT)

Random Forest was trained on graph-level statistical features, while GCN and GAT operated directly on graph-structured data using node features and graph connectivity.

## Requirements

- Python 3.x
- PyTorch
- PyTorch Geometric
- NetworkX
- Scikit-learn
- NumPy
- Matplotlib

## Repository Structure

```text
.
├── notebooks/
├── figures/
├── README.md
└── requirements.txt
