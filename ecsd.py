import math
from collections import Counter
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import networkx as nx

from math import lcm
from functools import reduce

def ecsd(array: list[list[int]]) -> nx.DiGraph:
    """Build the ECSD directed graph. array = [[d1,a1],[d2,a2],...], each entry maps n → d*n + a."""
    D = nx.DiGraph()

    maxdet = max(abs(x) for item in array for x in item)
    D.add_nodes_from(range(-maxdet, maxdet + 1))

    for n in range(-maxdet, maxdet + 1):
        for item in array:
            dest = item[0] * n + item[1]
            if -maxdet <= dest <= maxdet:
                D.add_edge(n, dest, label=str(item))

    return D

def is_exact_covering(array: list[list[int]]) -> bool:
    """Return True if array forms a valid exact covering system.
    Checks that every integer n in range(lcm of moduli) is covered exactly once."""
    mods    = [abs(item[0]) for item in array]
    offsets = [item[1] % abs(item[0]) for item in array]
    L = reduce(lcm, mods)
    return all(
        sum(1 for m, a in zip(mods, offsets) if n % m == a) == 1
        for n in range(L)
    )

def ecsdstuff(array: list[list[int]]) -> tuple:
    """Return (graph, node_count, cycles, sorted_cyclic_vertices). Use this to access raw data."""
    D = ecsd(array)
    order = D.number_of_nodes()
    cycles = list(nx.simple_cycles(D))
    vertices_in_cycles = sorted({v for cycle in cycles for v in cycle})
    return D, order, cycles, vertices_in_cycles


def cyclesizes(cycles: list[list[int]]) -> list[int]:
    """Return the length of each cycle. Pass the cycles list from ecsdstuff."""
    return [len(cycle) for cycle in cycles]


def compress_sizes(sizes: list[int]) -> str:
    """Format a size list as multiplicity strings, e.g. [1,1,2,6] → '2x1, 1x2, 1x6'."""
    counts = Counter(sizes)
    return ", ".join(f"{count}x{size}" for size, count in sorted(counts.items()))


def digital_repr(array: list[list[int]], n: int) -> list[int]:
    """Return the digital representation of n in the ECSD (Theorem 14, Neidmann 2026).

    Walks predecessors from n until hitting a cyclic vertex, collecting the
    offset (digit) used at each step. Returns [b0, b1, ..., bk] (LSB first)
    satisfying  n = b0*D^0 + b1*D^1 + ... + bk*D^k  (plus D^(k+1)*c if the
    cyclic ancestor c != 0). Also prints the equation.

    Theorem 14 guarantees a complete representation for all integers when the
    ECSD has a single component with 0 cyclic; warns otherwise."""
    _, _, cycles, verts = ecsdstuff(array)
    cycle_set = set(verts)
    D = array[0][0]          # common base — all |d_i| should be equal

    digits: list[int] = []
    current = n
    for _ in range(1000):    # safety cap; depth is O(log|n|) in practice
        if current in cycle_set:
            break
        for item in array:
            d, a = item[0], item[1]
            if (current - a) % d == 0:
                digits.append(a)
                current = (current - a) // d
                break

    cyclic_ancestor = current
    if cyclic_ancestor != 0:
        print(f"Warning: {n} belongs to a different component than 0 "
              f"(reached cyclic vertex {cyclic_ancestor}). "
              f"Theorem 14 requires 0 to be cyclic for a complete representation.")

    if not digits:
        print(f"{n} is a cyclic vertex — no tree representation.")
    else:
        base = f"({D})" if D < 0 else str(D)
        terms = [str(digits[0])] + [f"{b}*{base}^{j}" for j, b in enumerate(digits[1:], 1)]
        if cyclic_ancestor != 0:
            terms.append(f"{cyclic_ancestor}*{base}^{len(digits)}")
        print(f"{n} = {' + '.join(terms)}")

    return digits


def _rule_label(item: list[int]) -> str:
    d, a = item[0], item[1]
    if a == 0:
        return f'n → {d}n'
    sign = '+' if a > 0 else '−'
    return f'n → {d}n {sign} {abs(a)}'


def _expand_graph(array: list, cycle_set: set, levels: int) -> nx.DiGraph:
    """BFS from cycle nodes outward for `levels` steps to build the plotting graph."""
    D = nx.DiGraph()
    D.add_nodes_from(cycle_set)
    visited = set(cycle_set)
    frontier = set(cycle_set)

    for _ in range(levels):
        new_frontier = set()
        for n in frontier:
            for item in array:
                dest = item[0] * n + item[1]
                if dest not in visited:
                    D.add_node(dest)
                    visited.add(dest)
                    new_frontier.add(dest)
                D.add_edge(n, dest, label=str(item))
        frontier = new_frontier

    return D


def _layout_single(sub: nx.DiGraph, cycle_list: list, cycle_set: set) -> dict:
    """Radial sector layout for one component: cycle at center, tree fans outward."""
    pos = {}
    n_c = len(cycle_list)

    if n_c == 1:
        pos[cycle_list[0]] = (0.0, 0.0)
        queue = [(cycle_list[0], 0.0, 2 * math.pi, 1.5)]
    else:
        r_inner = 0.5
        sector = 2 * math.pi / n_c
        queue = []
        for i, node in enumerate(cycle_list):
            angle = 2 * math.pi * i / n_c
            pos[node] = (r_inner * math.cos(angle), r_inner * math.sin(angle))
            queue.append((node, angle - sector / 2, angle + sector / 2, r_inner + 1.2))

    placed = set(pos)
    while queue:
        parent, s, e, r = queue.pop(0)
        children = sorted(v for v in sub.successors(parent) if v not in placed and v not in cycle_set)
        if not children:
            continue
        placed.update(children)
        nc = len(children)
        for i, child in enumerate(children):
            a = s + (e - s) * (i + 0.5) / nc
            pos[child] = (r * math.cos(a), r * math.sin(a))
            queue.append((child, s + (e - s) * i / nc, s + (e - s) * (i + 1) / nc, r + 1.2))

    return pos


def _layout_expanded(D_plot: nx.DiGraph, cycle_set: set, cycles: list) -> dict:
    """Tile per-component radial layouts in a grid with uniform spacing."""
    components = sorted(nx.weakly_connected_components(D_plot), key=len, reverse=True)
    cols = math.ceil(math.sqrt(len(components)))

    # Pass 1: compute every per-component layout
    sub_pos_list = []
    for comp in components:
        sub = D_plot.subgraph(comp)
        cycle_list = next((c for c in cycles if set(c) <= comp),
                          sorted(n for n in comp if n in cycle_set))
        sub_pos_list.append(_layout_single(sub, list(cycle_list), cycle_set))

    # Pass 2: uniform grid step based on the widest component
    global_max_r = max(
        (max(math.hypot(x, y) for x, y in sp.values()) for sp in sub_pos_list if sp),
        default=1.0
    ) + 1.0

    all_pos = {}
    for idx, (comp, sub_pos) in enumerate(zip(components, sub_pos_list)):
        row, col = divmod(idx, cols)
        dx = col * global_max_r * 2.2
        dy = -row * global_max_r * 2.2
        for node, (x, y) in sub_pos.items():
            all_pos[node] = (x + dx, y + dy)

    return all_pos


def plot_ecsd(array: list[list[int]], levels: int = 5) -> None:
    """Plot the ECSD graph expanded `levels` tree layers from cycle nodes.
    Edges colored by rule (thick=cycle edge, thin=tree edge). Nodes: salmon=cyclic, lightblue=tree."""
    if (not is_exact_covering(array)):
        print("NOT ECSD")
        return
    D_small, _, cycles, vertices_in_cycles = ecsdstuff(array)
    cycle_set = set(vertices_in_cycles)

    D_plot = _expand_graph(array, cycle_set, levels)
    n_nodes = D_plot.number_of_nodes()

    rule_colors = [plt.cm.tab10(i % 10 / 10) for i in range(len(array))]

    if n_nodes <= 40:
        node_size, font_size, show_labels = 500, 9, True
    elif n_nodes <= 150:
        node_size, font_size, show_labels = 250, 7, True
    elif n_nodes <= 500:
        node_size, font_size, show_labels = 80, 5, True
    else:
        node_size, font_size, show_labels = 30, 4, False

    n_comp = len(list(nx.weakly_connected_components(D_plot)))
    cols = math.ceil(math.sqrt(n_comp))
    rows = math.ceil(n_comp / cols)
    pos = _layout_expanded(D_plot, cycle_set, cycles)

    plt.figure(figsize=(max(12, cols * 6), max(10, rows * 5)))
    node_colours = ['salmon' if n in cycle_set else 'lightblue' for n in D_plot.nodes()]
    nx.draw_networkx_nodes(D_plot, pos, node_color=node_colours, node_size=node_size)
    if show_labels:
        nx.draw_networkx_labels(D_plot, pos, font_size=font_size, font_weight='bold')

    kw = dict(arrows=True, connectionstyle='arc3,rad=0.05')
    for item, color in zip(array, rule_colors):
        rule_edges = [(u, v) for u, v, d in D_plot.edges(data=True) if d.get('label') == str(item)]
        cycle_e = [(u, v) for u, v in rule_edges if u in cycle_set and v in cycle_set]
        tree_e  = [(u, v) for u, v in rule_edges if (u, v) not in set(cycle_e)]
        if tree_e:
            nx.draw_networkx_edges(D_plot, pos, edgelist=tree_e,
                                   edge_color=[color]*len(tree_e), width=1.0, arrowsize=10, **kw)
        if cycle_e:
            nx.draw_networkx_edges(D_plot, pos, edgelist=cycle_e,
                                   edge_color=[color]*len(cycle_e), width=2.5, arrowsize=16, **kw)

    handles = [Line2D([0], [0], color=col, linewidth=2, label=_rule_label(item))
               for item, col in zip(array, rule_colors)]
    handles += [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='salmon',    markersize=10, label='cyclic vertex'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='tree vertex'),
    ]
    plt.legend(handles=handles, loc='upper right', framealpha=0.85)
    plt.title(f"ECSD {array}  (levels={levels})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def cycleinfo(array: list[list[int]]) -> None:

    """Print full cycle report: cyclic vertices, component count, cycle sizes, 0-cycle size, all cycles."""
    if (not is_exact_covering(array)):
        print("NOT ECSD")
        return
    D, order, cycles, vertices_in_cycles = ecsdstuff(array)
    print("vertices in cycles:", vertices_in_cycles)
    print("number of components:", len(cycles))
    print("number of cyclic vertices:", len(vertices_in_cycles))
    print("cycle sizes:", sorted(cyclesizes(cycles)))
    zerocycle = next((cycle for cycle in cycles if 0 in cycle), [])
    print("size of 0 cycle:", len(zerocycle))
    print("cycles:", cycles)


def cycleinfocompressed(array: list[list[int]]) -> None:
    
    """Print short report with cycle sizes in multiplicity form, e.g. '3x2, 1x6'."""
    if (not is_exact_covering(array)):
        print("NOT ECSD")
        return
    D, order, cycles, vertices_in_cycles = ecsdstuff(array)
    print("number of components:", len(cycles))
    print("number of cyclic vertices:", len(vertices_in_cycles))
    print("cycle sizes:", compress_sizes(sorted(cyclesizes(cycles))))


def cycleinfoshort(array: list[list[int]]) -> None:
    """Print short report: component count, cyclic vertex count, sorted cycle sizes."""
    if (not is_exact_covering(array)):
        print("NOT ECSD")
        return
    D, order, cycles, vertices_in_cycles = ecsdstuff(array)
    print("number of components:", len(cycles))
    print("number of cyclic vertices:", len(vertices_in_cycles))
    print("cycle sizes:", sorted(cyclesizes(cycles)))



if __name__ == "__main__":
    cycleinfo([[3,0], [3,1], [3,-7]])
    plot_ecsd([[3,0], [3,1], [3,-7]])
    print()
    cycleinfo([[3,0], [3,7], [3,-1]])
    plot_ecsd([[3,0], [3,7], [3,-1]])
