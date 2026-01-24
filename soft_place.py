# soft_place - a simple solver for differentiable k-median problem on graphs

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# graph utilities
def make_random_geometric_graph(n=50, radius=0.25, seed=1):
    G = nx.random_geometric_graph(n, radius, seed=seed)
    # ensure connectivity; if not, connect components
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for a, b in zip(comps[:-1], comps[1:]):
            ua = next(iter(a)); vb = next(iter(b))
            G.add_edge(ua, vb)
    # assign Euclidean distances as edge weight
    pos = nx.get_node_attributes(G, 'pos')
    for u, v in G.edges():
        x1, y1 = pos[u]; x2, y2 = pos[v]
        d = ((x1-x2)**2 + (y1-y2)**2)**0.5
        G[u][v]['weight'] = d
    return G

def compute_shortest_path_distance_matrix(G):
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    index = {node:i for i,node in enumerate(nodes)}
    D = np.zeros((n,n), dtype=float)
    for u in nodes:
        lengths = nx.single_source_dijkstra_path_length(G, u, weight='weight')
        for v, L in lengths.items():
            D[index[u], index[v]] = L
    return D, nodes

# differentiable model
class SoftPlaceFacilityModel(nn.Module):
    def __init__(self, n_nodes, k, beta=5.0, opening_cost=0.1):
        super().__init__()
        self.n = n_nodes
        self.k = k
        self.beta = beta
        self.opening_cost = opening_cost
        # logits controlling facility strengths per node
        self.s_logits = nn.Parameter(torch.zeros(n_nodes))  # initialize neutral
    def forward(self, D, demand):
        # D: (n,n) tensor of distances
        # demand: (n,) tensor
        # facility strengths s in [0,inf) but normalized to sum ~k
        s_softmax = torch.softmax(self.s_logits, dim=0)  # sums to 1
        s = s_softmax * self.k  # sum(s) = k (soft)
        # compute attraction a_ij = s_j * exp(-beta * D_ij)
        # D shape (n_customers, n_facilities) — here same n
        A = s.unsqueeze(0) * torch.exp(-self.beta * D)  # (n,n)
        P = A / (A.sum(dim=1, keepdim=True) + 1e-9)      # assignment probabilities
        # expected distance per customer i = sum_j P_ij * D_ij
        exp_dist = (P * D).sum(dim=1)  # (n,)
        cost = (demand * exp_dist).sum() + self.opening_cost * s.sum()
        return cost, s.detach().cpu().numpy(), P.detach().cpu().numpy()

# demo / training loop
def run_demo(n=50, k=3, iters=1000, lr=0.5, seed=1):
    G = make_random_geometric_graph(n=n, seed=seed)
    D_np, nodes = compute_shortest_path_distance_matrix(G)
    # synthetic demand per node (random)
    rng = np.random.RandomState(seed)
    demand = rng.rand(n).astype(float)
    demand = demand / demand.sum() * n  # scale so total demand ~ n
    # prepare tensors
    D = torch.tensor(D_np, dtype=torch.float32)
    demand_t = torch.tensor(demand, dtype=torch.float32)
    model = SoftPlaceFacilityModel(n, k, beta=5.0, opening_cost=0.05)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(iters):
        opt.zero_grad()
        cost, s_vals, _ = model(D, demand_t)
        cost.backward()
        opt.step()
        if it % (iters//5) == 0 or it == iters-1:
            print(f"iter {it:4d} cost {cost.item():.4f} top_s: {np.argsort(-s_vals)[:k].tolist()}")
    # final
    cost, s_vals, P = model(D, demand_t)
    topk_nodes_idx = list(np.argsort(-s_vals)[:k])
    topk_nodes = [nodes[i] for i in topk_nodes_idx]
    print("Final top-k node indices:", topk_nodes_idx)
    plot_solution(G, nodes, s_vals, topk_nodes_idx)
    return G, nodes, s_vals, topk_nodes_idx

def plot_solution(G, nodes, s_vals, topk_idx):
    pos = nx.get_node_attributes(G, 'pos')
    fig, ax = plt.subplots(figsize=(6,6))
    # draw graph
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4)
    sizes = (s_vals - s_vals.min() + 1e-6)
    sizes = 200 * sizes / sizes.max()
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color='C0', ax=ax)
    # highlight chosen facilities
    facility_nodes = [nodes[i] for i in topk_idx]
    nx.draw_networkx_nodes(G, pos, nodelist=facility_nodes, node_size=300, node_color='C1', ax=ax)
    ax.set_title("Learned facility strengths (size) and chosen facilities (orange)")
    plt.axis('off')
    plt.show()

# Run the demo
if __name__ == "__main__":
    run_demo(n=80, k=4, iters=800, lr=0.3, seed=42)
