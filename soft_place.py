"""
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


def run_demo(n, k, iters=1000, lr=0.3, seed=42):
    """
    Run the optimization with the neural facility location.

    :param n: The number of nodes.
    :param k: The number of facilities to select.
    :param iters: The number of iterations for training.
    :param lr: The learning rate.
    :param seed: Random seed.
    """
    G = make_random_geometric_graph(n=n, seed=seed)
    D_np, nodes = compute_shortest_path_distance_matrix(G)

    # generating synthetic demand per node (normalized so total demand ~= n)
    rng = np.random.RandomState(seed)
    demand = rng.rand(n).astype(float)
    demand = demand / demand.sum() * n

    # tensors used in optimization
    D = torch.tensor(D_np, dtype=torch.float32)
    demand_t = torch.tensor(demand, dtype=torch.float32)

    # beta
    # controls how fast attraction falls off with distance: A_ij = s_j * exp(-beta * D_ij).
    # small beta - smooth, weak distance effect - per-customer proximity is weaker, closer to global
    # distance optimization
    # large beta - highly local attraction, focused on proximity of local demand
    beta = 10.0

    # model and optimizer
    model = SoftPlaceFacilityModel(n, k, beta, opening_cost=0.05)
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
    plot_solution(G, nodes, s_vals, topk_nodes_idx, demand)
    return G, nodes, s_vals, topk_nodes_idx

def plot_solution(G, nodes, s_vals, topk_idx, demand, max_plot_size=600):
    """
    Two-panel plot:
      - left: absolute sizes with numeric labels = learned strengths (top-k highlighted)
      - right: pure nodes, labels show demand only
    """
    # summary stats for strengths (left)
    s_sum = s_vals.sum()
    s_max = s_vals.max()
    s_min = s_vals.min()
    topk_strengths = np.sort(s_vals)[-len(topk_idx):][::-1]
    print(f"sum(s)={s_sum:.4f}, max(s)={s_max:.4f}, min(s)={s_min:.4f}")
    print("top-k strengths:", np.round(topk_strengths, 4))

    # positions
    pos = nx.get_node_attributes(G, 'pos')

    # prepare demand array: use provided argument, or G.nodes[n]['demand'], else zeros
    if demand is None:
        try:
            demand = np.array([G.nodes[n].get('demand', 0.0) for n in nodes], dtype=float)
        except Exception:
            demand = np.zeros(len(nodes), dtype=float)
    else:
        if hasattr(demand, 'detach'):
            demand = demand.detach().cpu().numpy()
        else:
            demand = np.array(demand, dtype=float)

    # absolute mapping: max -> max_plot_size (left)
    sizes_abs = (s_vals / (s_max if s_max > 0 else 1.0)) * max_plot_size

    # relative mapping
    sizes_rel = (s_vals - s_min + 1e-9)
    sizes_rel = 200 * sizes_rel / (sizes_rel.max() if sizes_rel.max() > 0 else 1.0)

    topk_nodes = [nodes[i] for i in topk_idx]
    topk_sizes_abs = [sizes_abs[i] for i in topk_idx]

    # strength labels (for left)
    strength_labels = {nodes[i]: f"{s_vals[i]:.3f}" for i in range(len(nodes))}
    topk_set = set(topk_nodes)

    # demand labels (for right)
    demand_labels = {nodes[i]: f"{demand[i]:.2f}" for i in range(len(nodes))}

    # label positions: slight offsets (left: up, right: down)
    label_pos_left = {n: (pos[n][0], pos[n][1] + 0.01) for n in pos}
    label_pos_right = {n: (pos[n][0], pos[n][1] - 0.015) for n in pos}

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))

    # absolute sizes + numeric strength labels (top-k emphasized)
    nx.draw_networkx_edges(G, pos, ax=ax0, alpha=0.25)
    nx.draw_networkx_nodes(G, pos, node_size=sizes_abs, node_color='C0', ax=ax0)
    nx.draw_networkx_nodes(G, pos, nodelist=topk_nodes, node_size=topk_sizes_abs,
                           node_color='C1', edgecolors='k', linewidths=1.2, ax=ax0)

    # non-top-k strength labels
    non_topk_strength_labels = {n: lbl for n, lbl in strength_labels.items() if n not in topk_set}
    if non_topk_strength_labels:
        nx.draw_networkx_labels(G, label_pos_left, non_topk_strength_labels, font_size=8, ax=ax0)

    # top-k strength labels in bold dark red
    topk_strength_labels = {n: strength_labels[n] for n in topk_nodes}
    if topk_strength_labels:
        nx.draw_networkx_labels(G, label_pos_left, topk_strength_labels, font_size=9,
                                font_color='darkred', font_weight='bold', ax=ax0)

    ax0.set_title("Absolute weights learned")
    ax0.axis('off')

    nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.25)

    # demand labels
    nx.draw_networkx_labels(G, label_pos_right, demand_labels, font_size=8, ax=ax1)

    ax1.set_title("Demand per node")
    ax1.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_demo(n=100, k=5)
