
import random
import networkx as nx
import torch
from torch_geometric.data import Data, Dataset

def generate_watts_strogatz(n_nodes, with_label=False):

    k = 6

    p = random.uniform(0.05, 0.3)

    G = nx.watts_strogatz_graph(
        n=n_nodes,
        k=k,
        p=p
    )

    # Add edge noise
    for _ in range(10):

        u = random.randint(0, n_nodes - 1)
        v = random.randint(0, n_nodes - 1)

        if u != v:

            if G.has_edge(u, v):

                G.remove_edge(u, v)

            else:

                G.add_edge(u, v)

    if with_label:
        return G, p

    return G


def generate_barabasi_albert(n_nodes):

    m = 3

    G = nx.barabasi_albert_graph(
        n=n_nodes,
        m=m
    )

    # Add edge noise
    for _ in range(10):

        u = random.randint(0, n_nodes - 1)
        v = random.randint(0, n_nodes - 1)

        if u != v:

            if G.has_edge(u, v):

                G.remove_edge(u, v)

            else:

                G.add_edge(u, v)

    return G


def generate_random_regular(n_nodes):

    d = 6

    G = nx.random_regular_graph(
        d=d,
        n=n_nodes
    )

    # Add edge noise
    for _ in range(10):

        u = random.randint(0, n_nodes - 1)
        v = random.randint(0, n_nodes - 1)

        if u != v:

            if G.has_edge(u, v):

                G.remove_edge(u, v)

            else:

                G.add_edge(u, v)

    return G

class WattsStrogatzDataset(Dataset):
    """
    Each item is a Watts-Strogatz small-world graph with randomly sampled
    parameters (k, p). The label `y` is the rewiring probability p in [0, 1].
    """
    def __init__(self, dataset_size, n_nodes):
        self.dataset_size = dataset_size
        self.n_nodes = n_nodes

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        G, p, k = generate_watts_strogatz(self.n_nodes, with_label=True)

        # get the adjacency matrix from the nx graph
        A = nx.adjacency_matrix(G).todense()
        A = torch.from_numpy(A) * 1.0

        # get the edge indices from the adjacency matrix
        edge_index = A.nonzero(as_tuple=False).T

        # Create the graph data object
        graph = Data(
            edge_index=edge_index, 
            y=[p],
        )

        return graph

ws_dataset = WattsStrogatzDataset(
    dataset_size=200,
    n_nodes=100,
)






import matplotlib.pyplot as plt
from pygsp import graphs

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

num_nodes = 25

G = generate_watts_strogatz(num_nodes)
pos = graphs.Ring(N = num_nodes).coords
nx.draw(G, pos, ax=axs[0], node_size=60,
            edge_color="gray", with_labels=False)
axs[0].set_title("Watts-Strogatz")

G = generate_barabasi_albert(num_nodes)
pos = nx.spring_layout(G)
nx.draw(G, pos, ax=axs[1], node_size=60,
            edge_color="gray", with_labels=False)
axs[1].set_title("Barabasi-Albert")

G = generate_random_regular(num_nodes)
pos = nx.spring_layout(G)
nx.draw(G, pos, ax=axs[2], node_size=60,
            edge_color="gray", with_labels=False)
axs[2].set_title("Random Regular")

plt.tight_layout()
plt.show()
