""""
soft_place

a simple solver for a differentiable k-median / facility-location
problem on graphs


"""

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def make_random_geometric_graph(n=50, radius=0.25, seed=1):
    """
    Creates geometric graph with node positions and Euclidean edge weights.
    """
    G = nx.random_geometric_graph(n, radius, seed=seed)
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for a, b in zip(comps[:-1], comps[1:]):
            ua = next(iter(a)); vb = next(iter(b))
            G.add_edge(ua, vb)
    pos = nx.get_node_attributes(G, 'pos')
    for u, v in G.edges():
        x1, y1 = pos[u]; x2, y2 = pos[v]
        d = ((x1-x2)**2 + (y1-y2)**2)**0.5
        G[u][v]['weight'] = d
    return G


def compute_shortest_path_distance_matrix(G):
    """
    Computes dense all-pairs shortest-path distances (D[i,j] = dist from node i to j).
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    D = np.zeros((n, n), dtype=float)
    for u in nodes:
        lengths = nx.single_source_dijkstra_path_length(G, u, weight='weight')
        for v, L in lengths.items():
            D[index[u], index[v]] = L
    return D, nodes


class SoftPlaceFacilityModel(nn.Module):
    """
    Neural soft-place facility location solver through continuous relaxation.

    We have one learnable scalar per node (s_logits).
    Convert logits -> strengths -> soft facility selection.
    """
    def __init__(self, n_nodes, k, beta=5.0, opening_cost=0.1):
        super().__init__()
        self.n = n_nodes
        self.k = k
        self.beta = beta
        self.opening_cost = opening_cost

        # learnable logits: higher logit -> node becomes a stronger facility after softmax
        self.s_logits = nn.Parameter(torch.zeros(n_nodes))
        self.eps = 1e-9

    def forward(self, D, demand):
        # D: (n,n) distances
        # demand: (n,) nonnegative weights

        # normalize logits to a probability vector, then scale so strengths sum to k
        # reason: soft budget enforcement that remains differentiable
        s_softmax = torch.softmax(self.s_logits, dim=0)  # sums to 1
        s = s_softmax * self.k                            # sum(s) ≈ k

        # attraction: facility j attracts customer i proportional to s_j * exp(-beta * dist_ij).
        # strong nearby facilities score high
        A = s.unsqueeze(0) * torch.exp(-self.beta * D)   # shape (n_customers, n_facilities)

        # soft assignment: normalize attraction to obtain per-customer assignment probabilities
        # a differentiable twin of "assign each customer to nearest open facility"
        P = A / (A.sum(dim=1, keepdim=True) + self.eps)  # shape (n, n)

        # expected distance per customer under soft assignment
        exp_dist = (P * D).sum(dim=1)  # (n,)

        # final loss: demand-weighted expected distance + linear opening penalty
        cost = (demand * exp_dist).sum() + self.opening_cost * s.sum()

        # return cost (for optimization) and detached diagnostics
        return cost, s.detach().cpu().numpy(), P.detach().cpu().numpy()


def run_demo(n=50, k=3, iters=1000, lr=0.5, seed=1):
    G = make_random_geometric_graph(n=n, seed=seed)
    D_np, nodes = compute_shortest_path_distance_matrix(G)

    # generating synthetic demand per node (normalized so total demand ~= n)
    rng = np.random.RandomState(seed)
    demand = rng.rand(n).astype(float)
    demand = demand / demand.sum() * n

    # tensors used in optimization
    D = torch.tensor(D_np, dtype=torch.float32)
    demand_t = torch.tensor(demand, dtype=torch.float32)

    # model and optimizer
    model = SoftPlaceFacilityModel(n, k, beta=5.0, opening_cost=0.05)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # optimize logits to minimize differentiable objective
    for it in range(iters):
        opt.zero_grad()
        # cost depends differentiably on s_logits
        cost, s_vals, _ = model(D, demand_t)
        # gradients flow back into s_logits
        cost.backward()
        opt.step()
        if it % (max(1, iters // 5)) == 0 or it == iters - 1:
            # show which nodes currently have largest learned strength
            print(f"iter {it:4d} cost {cost.item():.4f} top_s: {np.argsort(-s_vals)[:k].tolist()}")

    # select top-k by learned strength and plot
    cost, s_vals, P = model(D, demand_t)
    topk_nodes_idx = list(np.argsort(-s_vals)[:k])
    print("Final top-k node indices:", topk_nodes_idx)
    plot_solution(G, nodes, s_vals, topk_nodes_idx)
    return G, nodes, s_vals, topk_nodes_idx


def plot_solution(G, nodes, s_vals, topk_idx):
    """
    Visualizes node strengths (by size) and highlights selected facilities.
    """
    pos = nx.get_node_attributes(G, 'pos')
    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4)
    sizes = (s_vals - s_vals.min() + 1e-6)
    sizes = 200 * sizes / sizes.max()
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color='C0', ax=ax)
    facility_nodes = [nodes[i] for i in topk_idx]
    nx.draw_networkx_nodes(G, pos, nodelist=facility_nodes, node_size=300, node_color='C1', ax=ax)
    ax.set_title("Learned strengths (size); chosen facilities (orange)")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    run_demo(n=80, k=4, iters=800, lr=0.3, seed=42)
