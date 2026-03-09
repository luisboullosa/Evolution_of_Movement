"""Helpers to draw neural network diagram and side-info for the visualizer.

This module provides two rendering modes:
- a simple diagram (no dependency on networkx)
- a networkx-based layout when networkx is available

It also exposes a small info renderer used by the main script.
"""

import numpy as np
from matplotlib.patches import Circle

try:
    import networkx as nx  # optional, used only for nicer layouts
except Exception:
    nx = None


def setup_nn_axis(fig, gs):
    ax_nn = fig.add_subplot(gs[1, 0])
    ax_nn.axis("off")

    return ax_nn


def draw_nn_diagram(ax_nn, use_nn, output_val=None) -> None:
    ax_nn.clear()
    ax_nn.axis("off")

    if use_nn is None:
        ax_nn.text(0.5, 0.5, "NN not found", ha="center", va="center", fontsize=10)
        return

    weights = use_nn.get("weights", [])
    biases = use_nn.get("biases", [])
    # optional second layer weights/biases (list of rows)
    layer2 = use_nn.get("layer2_weights", None)
    nodes2: int = len(layer2) if layer2 is not None else 0
    # detect multi-input: weights is list of lists (nodes x inputs)
    multi = False
    inputs_count = 1
    if len(weights) > 0 and hasattr(weights[0], "__iter__"):
        multi = True
        n: int = len(weights)
        inputs_count: int = len(weights[0]) if len(weights[0]) > 0 else 1
    else:
        n: int = max(len(weights), len(biases))

    xin, xnode, xout = 0.15, 0.5, 0.85
    yin = 0.5
    ys: list[float] = [(i + 1) / (n + 1) for i: int in range(n)] if n > 0 else [0.5]

    # draw input nodes: single or multiple
    input_names: list[str] = ["in"]
    if multi:
        # default names for spatial + occupancy + internal resources
        # Order must match input vector used elsewhere: left, center, right, occupancy, resources
        base_names: list[str] = ["L", "C", "R", "occ", "RES"]
        input_names: list[str] = [base_names[i] if i < len(base_names) else f"in{i}" for i: int in range(inputs_count)]
        # visual symbols for inputs: left arrow, center dot, right arrow, X for occupancy,
        # and 'R' for the individual's internal resources (RES)
        symbol_map: dict[str, str] = {
            "L": "\u2190",
            "C": "\u2022",
            "R": "\u2192",
            "occ": "\u2716",
            "RES": "R",
        }
        # place them vertically
        xin = 0.08
        ys_in: list[float] = [(i + 1) / (inputs_count + 1) for i: int in range(inputs_count)]
        for j, name in enumerate(input_names):
            yj: float = ys_in[j]
            ax_nn.add_patch(Circle((xin, yj), 0.04, facecolor="lightgray", edgecolor="k"))
            sym: str = symbol_map.get(name, name)
            # use a slightly larger fontsize for symbols to improve visibility
            ax_nn.text(xin, yj, sym, ha="center", va="center", fontsize=12, fontweight="bold")
    else:
        ax_nn.add_patch(Circle((xin, yin), 0.045, facecolor="lightgray", edgecolor="k"))
        ax_nn.text(
            xin,
            yin,
            "in",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            zorder=6,
        )
    ax_nn.add_patch(Circle((xout, yin), 0.045, facecolor="lightgray", edgecolor="k"))
    ax_nn.text(
        xout,
        yin,
        "out",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        zorder=6,
    )

    # draw hidden nodes and edges from inputs
    for i, y in enumerate(ys):
        ax_nn.add_patch(Circle((xnode, y), 0.04, facecolor="white", edgecolor="k"))
        ax_nn.text(xnode, y, f"n{i}", ha="center", va="center", fontsize=9, zorder=6)
        if multi:
            # connect each input j to node i
            ys_in: list[float] = [(j + 1) / (inputs_count + 1) for j: int in range(inputs_count)]
            for j: int in range(inputs_count):
                xin_j = 0.08
                yj: float = ys_in[j]
                w = 0.0
                try:
                    w = weights[i][j]
                except Exception:
                    w = 0.0
                color: str = "0.0" if w >= 0 else "0.2"
                lw: float = max(0.4, min(3.0, abs(w) * 3.0))
                ax_nn.plot([xin_j, xnode], [yj, y], color=color, linewidth=lw, alpha=0.9)
                mx: float = (xin_j + xnode) / 2.0
                my: float = (yj + y) / 2.0
                ax_nn.text(mx, my, f"{w:.2f}", fontsize=7, color=color, ha="center", va="center")
        else:
            w = weights[i] if i < len(weights) else 0.0
            color: str = "0.0" if w >= 0 else "0.2"
            lw: float = max(0.5, min(4.0, abs(w) * 4.0))
            ax_nn.plot([xin, xnode], [yin, y], color=color, linewidth=lw, alpha=0.9)
            mx: float = (xin + xnode) / 2.0
            my: float = (yin + y) / 2.0
            ax_nn.text(
                mx,
                my,
                f"{w:.2f}",
                fontsize=8,
                color=color,
                ha="center",
                va="center",
                zorder=6,
            )
        # if there's no second layer, connect hidden node directly to output
        if nodes2 == 0:
            ax_nn.plot([xnode, xout], [y, yin], color="0.5", linewidth=0.8, alpha=0.8)

    # if second layer present, draw a middle column of nodes and connect layers
    if nodes2 > 0:
        # place second layer in its own column between first hidden and output
        xnode2 = 0.75
        ys2: list[float] = [(i + 1) / (nodes2 + 1) for i: int in range(nodes2)]
        for i2, y2 in enumerate(ys2):
            ax_nn.add_patch(Circle((xnode2, y2), 0.04, facecolor="white", edgecolor="k"))
            ax_nn.text(xnode2, y2, f"m{i2}", ha="center", va="center", fontsize=9, zorder=6)
            # connect first-layer nodes to second-layer node
            for i, y in enumerate(ys):
                w = 0.0
                try:
                    w = layer2[i2][i]
                except Exception:
                    w = 0.0
                color: str = "0.0" if w >= 0 else "0.2"
                lw: float = max(0.4, min(3.0, abs(w) * 3.0))
                ax_nn.plot([xnode, xnode2], [y, y2], color=color, linewidth=lw, alpha=0.9)
                mx: float = (xnode + xnode2) / 2.0
                my: float = (y + y2) / 2.0
                ax_nn.text(mx, my, f"{w:.2f}", fontsize=7, color=color, ha="center", va="center")
            # connect second-layer node to output
            ax_nn.plot([xnode2, xout], [y2, yin], color="0.5", linewidth=0.8, alpha=0.8)

    if output_val is not None:
        try:
            sign: float = 1.0 if float(output_val) >= 0.0 else -1.0
        except Exception:
            sign = 1.0

        dx: float = 0.12 * sign
        ax_nn.annotate(
            "",
            xy=(xout + dx, yin),
            xytext=(xout, yin),
            arrowprops=dict(
                arrowstyle="->",
                color="0.0",
                linewidth=max(1.0, min(4.0, abs(float(output_val)) * 2.0)),
            ),
        )
        ax_nn.text(
            xout + dx + 0.02 * sign,
            yin,
            f"{float(output_val):.3f}",
            ha=("left" if sign > 0 else "right"),
            va="center",
            fontsize=8,
        )

    ax_nn.set_xlim(0.0, 1.0)
    ax_nn.set_ylim(0.0, 1.0)


def draw_nn_graph(ax_nn, use_nn, output_val=None):
    """Prefer a networkx layout if available, otherwise draw the simple diagram."""
    if nx is None:
        draw_nn_diagram(ax_nn, use_nn, output_val=output_val)
        return

    ax_nn.clear()
    ax_nn.axis("off")

    weights = use_nn.get("weights", [])
    biases = use_nn.get("biases", [])
    layer2 = use_nn.get("layer2_weights", None)
    nodes2: int = len(layer2) if layer2 is not None else 0
    multi = False
    inputs_count = 1
    if len(weights) > 0 and hasattr(weights[0], "__iter__"):
        multi = True
        n: int = len(weights)
        inputs_count: int = len(weights[0])
    else:
        n: int = max(len(weights), len(biases))

    G = nx.DiGraph()
    # add input nodes
    input_names: list[str] = ["in"]
    if multi:
        input_names: list[str] = [f"in{j}" for j: int in range(inputs_count)]
        for name: str in input_names:
            G.add_node(name)
    else:
        G.add_node("in")
    for i: int in range(n):
        G.add_node(f"n{i}")
    # second-layer nodes
    if nodes2 > 0:
        for j: int in range(nodes2):
            G.add_node(f"m{j}")
    G.add_node("out")

    if multi:
        for i: int in range(n):
            for j: int in range(inputs_count):
                w = 0.0
                try:
                    w = weights[i][j]
                except Exception:
                    w = 0.0
                G.add_edge(f"in{j}", f"n{i}", weight=w)
    else:
        for i, w in enumerate(weights):
            G.add_edge("in", f"n{i}", weight=w)
    if nodes2 > 0:
        for i: int in range(n):
            for j: int in range(nodes2):
                w = 0.0
                try:
                    w = layer2[j][i]
                except Exception:
                    w = 0.0
                G.add_edge(f"n{i}", f"m{j}", weight=w)
        for j: int in range(nodes2):
            G.add_edge(f"m{j}", "out", weight=1.0)
    else:
        for i: int in range(n):
            G.add_edge(f"n{i}", "out", weight=1.0)

    pos: dict[str, tuple[float, float]] = {"out": (0.95, 0.5)}
    if multi:
        # place inputs on the left stacked vertically
        ys_in: list[float] = [(j + 1) / (inputs_count + 1) for j: int in range(inputs_count)]
        for j, name in enumerate(input_names):
            pos[name] = (0.05, ys_in[j])
    else:
        pos["in"] = (0.05, 0.5)
    # position hidden nodes: first hidden layer at x=0.5, optional second layer at x=0.75
    x_first = 0.5
    x_second = 0.75
    # first-layer nodes
    first_nodes: list[str] = [f"n{i}" for i: int in range(n)]
    if len(first_nodes) > 0:
        ys_first: list[float] = [(i + 1) / (len(first_nodes) + 1) for i: int in range(len(first_nodes))]
        for name, y in zip(first_nodes, ys_first):
            pos[name] = (x_first, y)
    # second-layer nodes (if any)
    if nodes2 > 0:
        second_nodes: list[str] = [f"m{j}" for j: int in range(nodes2)]
        ys_second: list[float] = [(i + 1) / (len(second_nodes) + 1) for i: int in range(len(second_nodes))]
        for name, y in zip(second_nodes, ys_second):
            pos[name] = (x_second, y)

    nx.draw_networkx_nodes(G, pos, ax=ax_nn, node_size=900, node_color="lightgray")
    # draw node labels; for input nodes use symbols similar to the simple diagram
    for nname, npos in pos.items():
        label: str = nname
        if multi and nname.startswith("in"):
            # map in0..in3 -> L, C, R, occ symbol
            try:
                idx = int(nname[2:])
            except Exception:
                idx = None
            if idx is not None:
                # Map input indices to symbols: 0=L,1=C,2=right neighbor arrow,3=occupancy X,4=internal resources 'R'
                mapping: dict[int, str] = {0: "\u2190", 1: "\u2022", 2: "\u2192", 3: "\u2716", 4: "R"}
                label: str = mapping.get(idx, nname)
        elif not multi and nname == "in":
            label = "\u2022"
        ax_nn.text(
            npos[0],
            npos[1],
            label,
            fontsize=9 if len(str(label)) > 1 else 12,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
        )

    edges = G.edges(data=True)
    mx = my = None
    for u, v, data in edges:
        w = data.get("weight", 0.0)
        color: str = "blue" if w >= 0 else "red"
        lw: float = max(0.5, min(4.0, abs(w) * 4.0))
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax_nn, edge_color=color, width=lw, alpha=0.8)
        mx: float = (pos[u][0] + pos[v][0]) / 2.0
        my: float = (pos[u][1] + pos[v][1]) / 2.0

    if mx is not None and my is not None:
        ax_nn.text(
            mx,
            my,
            f"{data.get('weight', 0.0):.2f}",
            fontsize=7,
            ha="center",
            va="center",
            zorder=5,
        )

    if output_val is not None:
        try:
            sign: float = 1.0 if float(output_val) >= 0.0 else -1.0
        except Exception:
            sign = 1.0

        dx: float = 0.12 * sign
        out_pos: tuple[float, float] = pos.get("out", (0.95, 0.5))
        ax_nn.annotate(
            "",
            xy=(out_pos[0] + dx, out_pos[1]),
            xytext=(out_pos[0], out_pos[1]),
            arrowprops=dict(
                arrowstyle="->",
                color="0.0",
                linewidth=max(1.0, min(4.0, abs(float(output_val)) * 2.0)),
            ),
        )
        ax_nn.text(
            out_pos[0] + dx + 0.02 * sign,
            out_pos[1],
            f"{float(output_val):.3f}",
            ha=("left" if sign > 0 else "right"),
            va="center",
            fontsize=8,
        )


def draw_info(
    ax_nn,
    ax_info,
    use_nn=None,
    spend=None,
    delta=None,
    resources=None,
    output_vec=None,
    fitness=None,
    fitness_rank_pct=None,
    existence=None,
    current_time=None,
    times=None,
    pos=None,
    pos_percentile=None,
    pop=None,
    local_input=None,
):
    """
    Render the small info box containing local resource, NN output vector,
    and fitness and timing info.
    """
    if use_nn is None:
        return
    # prepare lines
    info_lines = []
    if resources is not None:
        try:
            info_lines.append(f"resource: {float(resources):.3f}")
        except Exception:
            pass
    if output_vec is not None:
        try:
            out_str: str = ", ".join([f"{v:.3f}" for v in np.asarray(output_vec).tolist()])
            info_lines.append(f"NN output: [{out_str}]")
        except Exception:
            pass
    # show relative fitness (label as 'rel fitness') when available
    # append the rank-percentile in parentheses if available
    if fitness is not None:
        try:
            if fitness_rank_pct is not None:
                info_lines.append(f"rel fitness: {float(fitness):.3f} ({fitness_rank_pct:.1f}%)")
            else:
                info_lines.append(f"rel fitness: {float(fitness):.3f}")
        except Exception:
            pass
    # show position and percentile if available
    if pos is not None:
        try:
            if pos_percentile is not None:
                info_lines.append(f"pos: {int(pos)} ({pos_percentile:.1f}%)")
            else:
                info_lines.append(f"pos: {int(pos)}")
        except Exception:
            pass
    # show population size when available
    if pop is not None:
        try:
            info_lines.append(f"pop: {int(pop)}")
        except Exception:
            pass
    # show the local NN input value (resource at local site)
    if local_input is not None:
        try:
            info_lines.append(f"local input: {float(local_input):.3f}")
        except Exception:
            pass
    if existence is not None and times is not None:
        try:
            first, last = existence
            info_lines.append(f"exists: t={times[first]}..t={times[last]}")
        except Exception:
            pass
    if current_time is not None:
        info_lines.append(f"current t: {current_time}")
    footer: str = ""
    if spend is not None:
        footer += f"spend={spend:.3f} "
    if delta is not None:
        footer += f"delta={delta}"

    try:
        if ax_info is not None:
            ax_info.clear()
            ax_info.axis("off")
            if len(info_lines) > 0:
                text: str = "\n".join(info_lines)
                if footer:
                    text += "\n" + footer
                ax_info.text(
                    0.0,
                    1.0,
                    text,
                    transform=ax_info.transAxes,
                    fontsize=9,
                    family="monospace",
                    va="top",
                )
        else:
            # fallback to ax_nn
            if len(info_lines) > 0:
                text: str = "\n".join(info_lines)
                if footer:
                    text += "\n" + footer
                ax_nn.text(
                    0.02,
                    0.02,
                    text,
                    transform=ax_nn.transAxes,
                    fontsize=9,
                    family="monospace",
                    va="bottom",
                )
    except Exception:
        pass


def compute_nn_response_for(x, use_nn, input_builder=None):
    """Compute per-input response array for a network-like dict.

    x may be a scalar or numpy array; use_nn is expected to be a dict with
    'weights' and 'biases' lists. If `input_builder` is provided it will be
    called for each scalar x value to produce the full input vector (for
    multi-input nets). The function preserves fast vectorized computation for
    legacy single-input networks and falls back to per-element evaluation for
    multi-input networks or when an input_builder is given.
    """
    # use_nn is a dict {"weights": [...], "biases": [...]} or None
    if use_nn is None:
        return np.tanh(x)

    weights = use_nn.get("weights", [])
    biases = use_nn.get("biases", [])

    # If an input_builder is provided, evaluate element-wise (safe for multi-input)
    if input_builder is not None:
        # handle array-like x
        if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
            xs = list(x)
            # build input matrix (rows=len(xs), cols=inputs_count)
            # determine inputs_count from weights if possible
            inputs_count: int | None = len(weights[0]) if (len(weights) > 0 and hasattr(weights[0], "__iter__")) else None
            input_rows = []
            for xi in xs:
                vec = input_builder(float(xi))
                # ensure numeric sequence
                try:
                    row: list[float] = [float(v) for v in vec]
                except Exception:
                    row: list[float] = [float(vec)]
                input_rows.append(row)
            # pad/truncate rows to a common length
            max_cols: int = max(len(r) for r in input_rows) if input_rows else 0
            mat: np.ndarray[tuple[int, int], np.dtype[np.Any]] = np.zeros((len(input_rows), max_cols), dtype=float)
            for i, r in enumerate(input_rows):
                for j, v in enumerate(r[:max_cols]):
                    mat[i, j] = v
            # Now compute outputs via matrix multiplication if weights are nested
            if len(weights) > 0 and hasattr(weights[0], "__iter__"):
                W: np.ndarray[tuple[Any, ...], np.dtype[np.Any]] = np.array([[float(wij) for wij in wn] + [0.0] * (mat.shape[1] - len(wn)) for wn in weights])
                # W shape: (n_nodes, inputs); mat shape: (n_x, inputs)
                dot = mat.dot(W.T)  # shape (n_x, n_nodes)
                b: np.ndarray[tuple[Any, ...], np.dtype[np.Any]] = np.array([float(bb) for bb in biases] + [0.0] * (dot.shape[1] - len(biases)))[: dot.shape[1]]
                z = np.tanh(dot + b.reshape((1, -1)))
                s = np.sum(z, axis=1)
                return s
            # fallback: compute per-row via compute_output_vector
            out: list[float] = [float(compute_output_vector(use_nn, row)[0]) for row in mat]
            return np.array(out)
        # scalar
        return float(compute_output_vector(use_nn, input_builder(float(x)))[0])

    # No input_builder: attempt vectorized computation for single-input nets
    if len(weights) > 0 and not hasattr(weights[0], "__iter__"):
        # weights is a list of scalars: vectorized across x
        s = np.zeros_like(x)
        for w, b in zip(weights, biases):
            s = s + np.tanh(float(w) * x + float(b))
        return s

    # Multi-input weights present but no input_builder: attempt vectorized evaluation
    if len(weights) > 0 and hasattr(weights[0], "__iter__"):
        # If x is iterable, build input matrix assuming scalar center-only input (first column)
        if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
            xs: np.ndarray[tuple[Any, ...], np.dtype[np.Any]] = np.asarray(list(x), dtype=float)
            n_x = xs.shape[0]
            inputs_count: int = len(weights[0])
            mat: np.ndarray[tuple[int, int], np.dtype[np.Any]] = np.zeros((n_x, inputs_count), dtype=float)
            mat[:, 0] = xs
            W: np.ndarray[tuple[Any, ...], np.dtype[np.Any]] = np.array([[float(wij) for wij in wn] for wn in weights])
            dot = mat.dot(W.T)
            b: np.ndarray[tuple[Any, ...], np.dtype[np.Any]] = np.array([float(bb) for bb in biases] + [0.0] * (dot.shape[1] - len(biases)))[: dot.shape[1]]
            z = np.tanh(dot + b.reshape((1, -1)))
            s = np.sum(z, axis=1)
            return s
        # scalar
        return float(compute_output_vector(use_nn, float(x))[0])


def compute_output_vector(use_nn, input_val) -> np.ndarray[tuple[Any, ...], np.dtype[np.Any]]:
    """Return the final output vector for the network given an input.

    Supports two layouts of `use_nn`:
    - weights/biases as lists of scalars -> single output
    - weights as list of lists (one sublist per output) -> multi-output
    Returns a numpy array of outputs.
    """
    # Support scalar or vector input_val. Return a 1-element array [total]
    if use_nn is None:
        # if input_val is vector-like, reduce to first element or apply tanh to scalar
        try:
            if hasattr(input_val, "__iter__"):
                in0 = float(input_val[0])
                return np.array([np.tanh(in0)])
        except Exception:
            pass
        return np.array([np.tanh(float(input_val))])

    weights = use_nn.get("weights", [])
    biases = use_nn.get("biases", [])
    # optional second-layer
    layer2 = use_nn.get("layer2_weights", None)
    layer2_biases = use_nn.get("layer2_biases", None)

    # Determine if weights are nested (nodes x inputs)
    if layer2 is not None and layer2_biases is not None:
        # compute first layer (support nested or flat weights)
        # ensure input_val is a sequence
        if not hasattr(input_val, "__iter__"):
            input_vec: list[float] = [float(input_val)]
        else:
            input_vec = list(input_val)
        # first-layer activations
        layer1 = []
        if len(weights) > 0 and hasattr(weights[0], "__iter__"):
            for i, w_node in enumerate(weights):
                dot = 0.0
                for j, w in enumerate(w_node):
                    if j < len(input_vec):
                        dot += float(w) * float(input_vec[j])
                b = biases[i] if i < len(biases) else 0.0
                layer1.append(np.tanh(dot + float(b)))
        else:
            in0: float = float(input_vec[0]) if len(input_vec) > 0 else 0.0
            for w, b in zip(weights, biases):
                layer1.append(np.tanh(float(w) * in0 + float(b)))
        # compute second layer outputs
        total = 0.0
        for i, brow in enumerate(layer2):
            dot = 0.0
            for j, wj in enumerate(brow):
                if j < len(layer1):
                    dot += float(wj) * float(layer1[j])
            b2 = layer2_biases[i] if i < len(layer2_biases) else 0.0
            a2 = np.tanh(dot + float(b2))
            total += a2
        return np.array([total])

    if len(weights) > 0 and hasattr(weights[0], "__iter__"):
        # Ensure input_val is a sequence
        if not hasattr(input_val, "__iter__"):
            # scalar -> treat as center-only vector
            input_vec: list[float] = [float(input_val)]
        else:
            input_vec = list(input_val)
        total = 0.0
        for i, w_node in enumerate(weights):
            # compute dot product of w_node and input_vec (handle mismatched lengths)
            dot = 0.0
            for j, w in enumerate(w_node):
                if j < len(input_vec):
                    dot += float(w) * float(input_vec[j])
            b = biases[i] if i < len(biases) else 0.0
            a = np.tanh(dot + float(b))
            total += a
        return np.array([total])

    # legacy single-input case: weights is a list of scalars per node
    # accept scalar or vector input_val; use first element if vector
    in0: float = float(input_val[0]) if hasattr(input_val, "__iter__") else float(input_val)
    total = 0.0
    for w, b in zip(weights, biases):
        total += np.tanh(float(w) * in0 + float(b))
    return np.array([total])


def forward_single_with_nn(input_val, use_nn) -> tuple[Any, Any] | tuple[Any, float]:
    """Compute a scalar directional output and magnitude from a network.

    Returns (direction, magnitude) where direction is tanh(total) and
    magnitude is abs(total). Matches previous behavior in the main script.
    """
    # Use compute_output_vector to handle scalar or vector inputs uniformly
    if use_nn is None:
        try:
            in0: float = float(input_val[0]) if hasattr(input_val, "__iter__") else float(input_val)
        except Exception:
            in0 = float(input_val)
        s = np.tanh(in0)
        return np.tanh(s), abs(s)

    out: np.ndarray[tuple[Any, ...], np.dtype[np.Any]] = compute_output_vector(use_nn, input_val)
    try:
        total = float(np.asarray(out).flatten()[0])
    except Exception:
        total = 0.0
    direction = np.tanh(total)
    magnitude: float = abs(total)
    return direction, magnitude
