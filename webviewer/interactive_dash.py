"""
Interactive (Shiny-like) playground for evo-movement-sim viewers.

Features:
- Environment explorer: tweak Perlin-ish parameters and watch the 1D landscape evolve.
- NN response explorer: edit weights/biases and see movement vs gather decisions.
- Carrying capacity estimator: rough K calculator using the simulator's resource and cost rules.

Run:
    python webviewer/interactive_dash.py --port 8050

Dependencies: dash>=2, plotly, numpy.
"""

import argparse
import copy
import math
from typing import List, Tuple

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

try:
    from randomgen import ChaCha as RGChaCha
except ImportError:
    RGChaCha = None


# Defaults pulled from the Rust simulator (see rust/src/config.rs)
NN_INPUTS = 5
NN_NODES = 10
NN_NODES2 = 10
NN_LAYERS = 2
MUTATION_RATE = 1.0
MUTATION_SIGMA = 0.1
CHACHA_ROUNDS = 12  # Rust's StdRng uses ChaCha12


def chacha_rng(seed: int) -> np.random.Generator:
    """Match rust rand::rngs::StdRng (ChaCha12) using randomgen's ChaCha bit generator."""
    if RGChaCha is None:
        raise RuntimeError("randomgen.ChaCha not installed; pip install randomgen")
    # randomgen.ChaCha is runtime-compatible with numpy's BitGenerator, but static type
    # checkers may complain; cast to np.random.BitGenerator to satisfy typing.
    from typing import cast

    return np.random.Generator(cast(np.random.BitGenerator, RGChaCha(seed=seed, rounds=CHACHA_ROUNDS)))


def smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple box smoothing to mimic lower-frequency Perlin; window<=1 returns original."""
    if window <= 1:
        return arr
    w = int(window)
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")


def preset_strategies() -> List[Tuple[str, dict]]:
    """Deterministic preset NN states to explore distinct response shapes."""

    def empty_state():
        return {
            "w1": [[0.0] * NN_INPUTS for _ in range(NN_NODES)],
            "b1": [0.0] * NN_NODES,
            "w2": [[0.0] * NN_NODES for _ in range(NN_NODES2)],
            "b2": [0.0] * NN_NODES2,
        }

    def set_w(row, col, val, arr):
        arr[row][col] = val

    presets: List[Tuple[str, dict]] = []

    # 0) Neutral gatherer: stays near zero, typically gathers.
    presets.append(("neutral", empty_state()))

    # 1) Resource seeker: move right when resources high (positive response on resource).
    s1 = empty_state()
    set_w(0, 0, 1.5, s1["w1"])
    s1["b1"][0] = 0.0
    set_w(0, 0, 1.0, s1["w2"])
    s1["b2"][0] = 0.0
    presets.append(("seek high resource", s1))

    # 2) Resource avoider: move left when resources high (negative response).
    s2 = empty_state()
    set_w(0, 0, -1.5, s2["w1"])
    s2["b1"][0] = 0.0
    set_w(0, 0, 1.0, s2["w2"])
    s2["b2"][0] = 0.0
    presets.append(("avoid high resource", s2))

    # 3) Competition avoider: move when competition high (repelled by competition input).
    s3 = empty_state()
    set_w(1, 1, 1.5, s3["w1"])
    s3["b1"][1] = 0.0
    set_w(0, 1, 1.0, s3["w2"])
    s3["b2"][0] = 0.0
    presets.append(("competition avoider", s3))

    # 4) Edge seeker: prefers mid resources, slight oscillation via two opposing nodes.
    s4 = empty_state()
    set_w(0, 0, 1.2, s4["w1"])
    s4["b1"][0] = -0.4
    set_w(1, 0, -1.2, s4["w1"])
    s4["b1"][1] = 0.4
    set_w(0, 0, 0.9, s4["w2"])
    set_w(0, 1, 0.9, s4["w2"])
    s4["b2"][0] = 0.0
    presets.append(("mid-range seeker", s4))

    # 5) Bold mover: strong bias to move right regardless of input.
    s5 = empty_state()
    s5["b1"][0] = 1.2
    set_w(0, 0, 1.0, s5["w2"])
    s5["b2"][0] = 1.0
    presets.append(("always move right", s5))

    # 6) Cautious gatherer: negative bias keeps it gathering unless strong resource.
    s6 = empty_state()
    s6["b1"][0] = -1.0
    set_w(0, 0, 1.0, s6["w2"])
    s6["b2"][0] = -0.5
    presets.append(("cautious gatherer", s6))

    return presets


def cauchy_mutation(rng: np.random.Generator, sigma: float) -> float:
    """Sample a Cauchy(0, sigma) increment."""
    return float(rng.standard_cauchy() * sigma)


def perlin_base(width: int, seed: int, resource_scale: float) -> np.ndarray:
    """Match rust/perlin.rs::generate (StdRng seeded, values in [0,1]) and scale."""
    rng = chacha_rng(seed)
    base = rng.random(width)
    return base * resource_scale


def perlin_slice(width: int, seed: int, time_offset: float, time_scale: float) -> np.ndarray:
    """Match rust/perlin.rs::sample_time_slice: mix seed with time_offset, ignore scales."""
    tseed = int(time_offset * 1e6) & 0xFFFFFFFFFFFFFFFF
    tseed = (tseed + seed) % (2**63 - 1)
    rng = chacha_rng(tseed)
    return rng.random(width)


def env_frame(
    base: np.ndarray,
    seed: int,
    t: float,
    time_amplitude: float,
    time_scale: float,
    resource_scale: float,
) -> np.ndarray:
    # Mirror rust/environment.rs: resources = (perlin_base + slice * time_amplitude) * resource_scale
    slice_vals = perlin_slice(base.size, seed, t, time_scale)
    return (base + slice_vals * time_amplitude) * resource_scale


def forward_multi_layer(
    inputs: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Emulate rust NeuralNetwork forward_multi for 2-layer nets with flexible inputs."""
    x = np.array(inputs, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    # pad or truncate to NN_INPUTS to keep shape stable
    if x.shape[1] < NN_INPUTS:
        pad = np.zeros((x.shape[0], NN_INPUTS), dtype=float)
        pad[:, : x.shape[1]] = x
        x = pad
    elif x.shape[1] > NN_INPUTS:
        x = x[:, :NN_INPUTS]

    h = np.tanh(x @ w1.T + b1)
    out2 = np.tanh(h @ w2.T + b2)
    summed = np.sum(out2, axis=1)
    return summed, out2


def parse_csv_floats(text: str) -> List[float]:
    if not text.strip():
        return []
    vals = []
    for part in text.split(","):
        try:
            vals.append(float(part.strip()))
        except Exception:
            continue
    return vals


def carrying_capacity(
    width: int,
    resource_scale: float,
    time_amplitude: float,
    maintenance_cost_per_node: float,
    death_risk: float,
) -> Tuple[float, str]:
    mean_base = 0.5 * resource_scale
    mean_time = 0.5 * time_amplitude
    mean_res = (mean_base + mean_time) * resource_scale
    maint = (NN_NODES + NN_NODES2) * maintenance_cost_per_node
    if maint <= 0 or mean_res <= 0:
        return math.inf, "infinite"
    k = (mean_res * width) / maint
    if death_risk > 0:
        k *= max(0.1, 1.0 - min(death_risk * 0.5, 0.9))
    if k > 50_000:
        return k, "infinite (capped >50k)"
    return k, f"~{int(k):,}"


def build_app() -> Dash:
    app = Dash(__name__)
    presets = preset_strategies()

    app.layout = html.Div(
        [
            html.H2("evo-movement-sim interactive playground"),
            dcc.Store(
                id="nn-state",
                data=copy.deepcopy(presets[0][1]),
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("1) Environment explorer"),
                            html.P("Adjust Perlin parameters (mirrors rust/perlin.rs + environment.rs)."),
                            html.Label("Grid width"),
                            dcc.Slider(
                                id="width",
                                min=8,
                                max=400,
                                step=1,
                                value=64,
                                marks=None,
                                tooltip={"placement": "bottom"},
                            ),
                            html.Label("Seed"),
                            dcc.Input(id="seed", type="number", value=42, debounce=True),
                            html.Label("Resource scale"),
                            dcc.Slider(
                                id="resource-scale",
                                min=0.1,
                                max=5.0,
                                step=0.1,
                                value=1.0,
                                marks=None,
                                tooltip={"placement": "bottom"},
                            ),
                            html.Label("Smoothing window (bigger = smoother)"),
                            dcc.Slider(
                                id="smooth-window",
                                min=1,
                                max=25,
                                step=1,
                                value=1,
                                marks=None,
                                tooltip={"placement": "bottom"},
                            ),
                            html.Label("Time speed"),
                            dcc.Slider(
                                id="time-speed",
                                min=0.0,
                                max=2.0,
                                step=0.05,
                                value=0.2,
                                marks=None,
                                tooltip={"placement": "bottom"},
                            ),
                            html.Label("Time amplitude"),
                            dcc.Slider(
                                id="time-amplitude",
                                min=0.0,
                                max=3.0,
                                step=0.1,
                                value=1.0,
                                marks=None,
                                tooltip={"placement": "bottom"},
                            ),
                            html.Label("Time scale"),
                            dcc.Slider(
                                id="time-scale",
                                min=0.0,
                                max=1.0,
                                step=0.01,
                                value=0.1,
                                marks=None,
                                tooltip={"placement": "bottom"},
                            ),
                            html.Div(
                                [
                                    dcc.Checklist(
                                        id="auto-play",
                                        options=[{"label": "auto-play", "value": "run"}],
                                        value=[],
                                        inline=True,
                                    ),
                                    html.Label("Manual time"),
                                    dcc.Slider(
                                        id="manual-time",
                                        min=0.0,
                                        max=10.0,
                                        step=0.1,
                                        value=0.0,
                                        marks=None,
                                        tooltip={"placement": "bottom"},
                                    ),
                                ]
                            ),
                            dcc.Graph(
                                id="env-graph",
                                style={"height": "420px"},
                                config={"responsive": False},
                            ),
                            dcc.Store(id="env-base"),
                            dcc.Store(id="env-time", data={"t": 0.0}),
                            dcc.Interval(id="tick", interval=500, n_intervals=0),
                        ],
                        style={
                            "flex": "1 1 0",
                            "padding": "12px",
                            "border": "1px solid #ccc",
                            "overflow": "visible",
                        },
                    ),
                    html.Div(
                        [
                            html.H4("2) NN response explorer"),
                            html.P("Uses rust NN defaults (inputs=5, nodes=10, nodes2=10, layers=2)."),
                            html.Label("Strategy preset"),
                            dcc.Slider(
                                id="strategy-index",
                                min=0,
                                max=len(presets) - 1,
                                step=1,
                                value=0,
                                marks={
                                    i: {
                                        "label": short,
                                        "style": {"whiteSpace": "nowrap", "fontSize": "11px"},
                                    }
                                    for i, (short, _) in enumerate(
                                        [
                                            ("neutral", None),
                                            ("seek high", None),
                                            ("avoid high", None),
                                            ("avoid comp", None),
                                            ("mid seek", None),
                                            ("move right", None),
                                            ("cautious", None),
                                        ]
                                    )
                                },
                                tooltip={"placement": "bottom"},
                                updatemode="drag",
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Apply preset", id="apply-weights", n_clicks=0, style={"marginRight": "6px"}
                                    ),
                                    html.Button("Mutate", id="mutate", n_clicks=0),
                                ],
                                style={"marginBottom": "6px"},
                            ),
                            dcc.Graph(
                                id="nn-graph",
                                style={"height": "420px"},
                                config={"responsive": False},
                            ),
                            html.Div(id="movement-text", style={"marginTop": "8px", "fontFamily": "monospace"}),
                        ],
                        style={
                            "flex": "1 1 0",
                            "padding": "12px",
                            "border": "1px solid #ccc",
                            "overflow": "visible",
                        },
                    ),
                    html.Div(
                        [
                            html.H4("3) Carrying capacity (K) estimator"),
                            html.P("Uses default NN size from rust/config.rs (nodes=10, nodes2=10)."),
                            html.Label("Grid width"),
                            dcc.Slider(
                                id="k-width",
                                min=8,
                                max=400,
                                step=1,
                                value=64,
                                marks=None,
                                tooltip={"placement": "bottom"},
                            ),
                            html.Label("Resource scale"),
                            dcc.Slider(
                                id="k-resource-scale",
                                min=0.1,
                                max=5.0,
                                step=0.1,
                                value=1.0,
                                marks=None,
                                tooltip={"placement": "bottom"},
                            ),
                            html.Label("Time amplitude"),
                            dcc.Slider(
                                id="k-time-amplitude",
                                min=0.0,
                                max=3.0,
                                step=0.1,
                                value=1.0,
                                marks=None,
                                tooltip={"placement": "bottom"},
                            ),
                            html.Label("Maintenance cost per node"),
                            dcc.Slider(
                                id="k-maint",
                                min=0.0,
                                max=0.05,
                                step=0.001,
                                value=0.01,
                                marks=None,
                                tooltip={"placement": "bottom"},
                            ),
                            html.Label("Death risk"),
                            dcc.Slider(
                                id="k-death",
                                min=0.0,
                                max=0.1,
                                step=0.001,
                                value=0.0,
                                marks=None,
                                tooltip={"placement": "bottom"},
                            ),
                            html.Div(
                                id="k-output",
                                style={"marginTop": "10px", "fontFamily": "monospace", "fontSize": "18px"},
                            ),
                        ],
                        style={
                            "flex": "1 1 0",
                            "padding": "12px",
                            "border": "1px solid #ccc",
                            "overflow": "visible",
                        },
                    ),
                ],
                style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
            ),
        ],
        style={"fontFamily": "Segoe UI", "margin": "12px"},
    )

    # Base recomputation when structure params change
    @app.callback(
        Output("env-base", "data"),
        Input("width", "value"),
        Input("seed", "value"),
        Input("resource-scale", "value"),
        Input("smooth-window", "value"),
    )
    def recompute_base(width, seed, resource_scale, smooth_window):
        base = perlin_base(int(width), int(seed), float(resource_scale))
        base = smooth(base, int(smooth_window or 1))
        return base.tolist()

    # Time keeper
    @app.callback(
        Output("env-time", "data"),
        Input("tick", "n_intervals"),
        State("env-time", "data"),
        State("time-speed", "value"),
        State("auto-play", "value"),
        State("manual-time", "value"),
    )
    def advance_time(_n, data, time_speed, autoplay, manual_time):
        t = float(data.get("t", 0.0))
        if autoplay and "run" in autoplay:
            t += float(time_speed)
        else:
            t = float(manual_time or 0.0)
        return {"t": t}

    # Environment graph update
    @app.callback(
        Output("env-graph", "figure"),
        Input("env-base", "data"),
        Input("env-time", "data"),
        Input("time-amplitude", "value"),
        Input("time-scale", "value"),
        Input("resource-scale", "value"),
        Input("seed", "value"),
        Input("smooth-window", "value"),
    )
    def update_env_plot(base_data, time_data, time_amplitude, time_scale, resource_scale, seed, smooth_window):
        if base_data is None:
            return go.Figure()
        base = np.array(base_data, dtype=float)
        t = float(time_data.get("t", 0.0)) if time_data else 0.0
        slice_vals = perlin_slice(base.size, int(seed), t, float(time_scale))
        slice_vals = smooth(slice_vals, int(smooth_window or 1))
        frame = (base + slice_vals * float(time_amplitude)) * float(resource_scale)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=frame, mode="lines", name=f"t={t:.2f}"))
        ymax_expected = (float(resource_scale) + float(time_amplitude)) * float(resource_scale)
        ymin = 0.0
        ymax = max(ymax_expected, float(frame.max()) if frame.size else 1.0)
        pad = max(0.1, 0.05 * ymax)
        fig.update_layout(
            autosize=False,
            height=380,
            margin=dict(l=10, r=10, t=30, b=30),
            xaxis_title="position",
            yaxis_title="resources",
            xaxis=dict(range=[-1, base.size + 1]),
            yaxis=dict(range=[ymin - pad, ymax + pad]),
        )
        return fig

    # NN response graph update
    @app.callback(
        Output("nn-state", "data"),
        Input("apply-weights", "n_clicks"),
        Input("mutate", "n_clicks"),
        Input("strategy-index", "value"),
        State("nn-state", "data"),
        prevent_initial_call=True,
    )
    def sync_nn(apply_clicks, mutate_clicks, strategy_idx, state):
        ctx = dash.callback_context
        trigger = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
        new_state = copy.deepcopy(state) if state else copy.deepcopy(presets[0][1])

        rng = np.random.default_rng()

        if "strategy-index.value" in trigger or "apply-weights.n_clicks" in trigger:
            idx = int(strategy_idx or 0)
            idx = max(0, min(idx, len(presets) - 1))
            _, preset_state = presets[idx]
            new_state = copy.deepcopy(preset_state)

        if "mutate.n_clicks" in trigger:
            mutated = 0
            for arr in new_state["w1"]:
                for i in range(len(arr)):
                    if rng.random() < MUTATION_RATE:
                        arr[i] += cauchy_mutation(rng, MUTATION_SIGMA)
                        mutated += 1
            for i in range(len(new_state["b1"])):
                if rng.random() < MUTATION_RATE:
                    new_state["b1"][i] += cauchy_mutation(rng, MUTATION_SIGMA)
                    mutated += 1
            for arr in new_state["w2"]:
                for i in range(len(arr)):
                    if rng.random() < MUTATION_RATE:
                        arr[i] += cauchy_mutation(rng, MUTATION_SIGMA)
                        mutated += 1
            for i in range(len(new_state["b2"])):
                if rng.random() < MUTATION_RATE:
                    new_state["b2"][i] += cauchy_mutation(rng, MUTATION_SIGMA)
                    mutated += 1
            if mutated == 0:
                # ensure visible change: mutate one random parameter
                targets = (
                    [("w1", i, j) for i in range(NN_NODES) for j in range(NN_INPUTS)]
                    + [("b1", i, None) for i in range(NN_NODES)]
                    + [("w2", i, j) for i in range(NN_NODES2) for j in range(NN_NODES)]
                    + [("b2", i, None) for i in range(NN_NODES2)]
                )
                idx = int(rng.integers(0, len(targets)))
                t = targets[idx]
                if t[0] == "w1":
                    new_state["w1"][t[1]][t[2]] += cauchy_mutation(rng, MUTATION_SIGMA)
                elif t[0] == "b1":
                    new_state["b1"][t[1]] += cauchy_mutation(rng, MUTATION_SIGMA)
                elif t[0] == "w2":
                    new_state["w2"][t[1]][t[2]] += cauchy_mutation(rng, MUTATION_SIGMA)
                else:
                    new_state["b2"][t[1]] += cauchy_mutation(rng, MUTATION_SIGMA)

        return new_state

    @app.callback(
        Output("nn-graph", "figure"),
        Output("movement-text", "children"),
        Input("nn-state", "data"),
    )
    def update_nn_plot(state):
        w1 = np.array(state.get("w1", [[0.0] * NN_INPUTS for _ in range(NN_NODES)]), dtype=float)
        b1 = np.array(state.get("b1", [0.0] * NN_NODES), dtype=float)
        w2 = np.array(state.get("w2", [[0.0] * NN_NODES for _ in range(NN_NODES2)]), dtype=float)
        b2 = np.array(state.get("b2", [0.0] * NN_NODES2), dtype=float)

        xs = np.linspace(-5, 10, 200)
        fig = go.Figure()
        chosen_idx = len(xs) // 2
        chosen_x = xs[chosen_idx]
        summaries = []

        def make_inputs(
            center: np.ndarray,
            comp: float = 0.0,
            left: np.ndarray | float = 0.0,
            right: np.ndarray | float = 0.0,
            occ: float = 0.0,
        ) -> np.ndarray:
            center_arr = np.array(center, dtype=float)
            left_arr = np.broadcast_to(left, center_arr.shape)
            right_arr = np.broadcast_to(right, center_arr.shape)
            inputs = np.zeros((center_arr.size, NN_INPUTS), dtype=float)
            inputs[:, 0] = center_arr
            if NN_INPUTS > 1:
                inputs[:, 1] = comp
            if NN_INPUTS > 2:
                inputs[:, 2] = left_arr
            if NN_INPUTS > 3:
                inputs[:, 3] = right_arr
            if NN_INPUTS > 4:
                inputs[:, 4] = occ
            return inputs

        scenarios = [
            ("peak", make_inputs(xs, comp=0.0, left=0.0, right=0.0, occ=0.0), "#2c7be5", "solid"),
            ("slope→", make_inputs(xs, comp=0.0, left=xs * 0.5, right=xs, occ=0.0), "#7c4dff", "solid"),
            ("slope←", make_inputs(xs, comp=0.0, left=xs, right=xs * 0.5, occ=0.0), "#00b8a9", "solid"),
            ("sink", make_inputs(-xs, comp=0.0, left=0.0, right=0.0, occ=0.0), "#ef6c00", "solid"),
            ("flat", make_inputs(xs, comp=0.0, left=xs, right=xs, occ=0.0), "#607d8b", "solid"),
        ]
        # add a competition/high-density variant for peak only
        scenarios.append(("peak (comp=2)", make_inputs(xs, comp=2.0, left=0.0, right=0.0, occ=0.0), "#9c27b0", "dot"))

        fig.add_hrect(y0=-0.5, y1=0.5, fillcolor="#d9ecff", opacity=0.18, line_width=0)
        fig.add_hrect(y0=0.5, y1=3.0, fillcolor="#e6f7e6", opacity=0.12, line_width=0)
        fig.add_hrect(y0=-3.0, y1=-0.5, fillcolor="#fde2e2", opacity=0.12, line_width=0)

        for name, inp_mat, color, dash in scenarios:
            summed, _ = forward_multi_layer(inp_mat, w1, b1, w2, b2)
            y_chosen = float(summed[chosen_idx])
            fig.add_trace(go.Scatter(x=xs, y=summed, mode="lines", name=name, line=dict(color=color, dash=dash)))
            if "comp=2" not in name:
                fig.add_trace(
                    go.Scatter(
                        x=[chosen_x],
                        y=[y_chosen],
                        mode="markers",
                        name=f"{name} @ x={chosen_x:.1f}",
                        marker=dict(size=7),
                    )
                )
            magnitude = abs(y_chosen)
            will_move = magnitude >= 0.5
            direction = "right" if y_chosen > 0 else "left"
            summaries.append(f"{name}: resp={y_chosen:.3f} -> {'move '+direction if will_move else 'gather'}")

        fig.add_hline(y=0.5, line=dict(color="orange", dash="dot"))
        fig.add_hline(y=-0.5, line=dict(color="orange", dash="dot"))
        fig.add_vline(x=chosen_x, line=dict(color="gray", dash="dot"))
        fig.update_layout(
            autosize=False,
            height=420,
            margin=dict(l=60, r=40, t=40, b=50),
            xaxis_title="center resource input",
            yaxis_title="NN response (sum over layer2)",
            xaxis=dict(range=[-5, 10]),
            legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.9)", orientation="h"),
        )

        text = " | ".join(summaries)
        text += "\nArchitecture: inputs=5 (center, competition, left, right, occupancy), nodes=10, nodes2=10, layers=2"
        text += f" | mutation Cauchy(sigma={MUTATION_SIGMA}, rate={MUTATION_RATE})"
        text += " | scenarios shown: peak, slope→, slope←, sink, flat (plus peak with comp=2 dotted)"
        return fig, text

    # Carrying capacity calculator
    @app.callback(
        Output("k-output", "children"),
        Input("k-width", "value"),
        Input("k-resource-scale", "value"),
        Input("k-time-amplitude", "value"),
        Input("k-maint", "value"),
        Input("k-death", "value"),
    )
    def update_k(width, resource_scale, time_amplitude, maint, death_risk):
        k_val, label = carrying_capacity(
            int(width), float(resource_scale), float(time_amplitude), float(maint), float(death_risk)
        )
        mean_base = 0.5 * float(resource_scale)
        mean_time = 0.5 * float(time_amplitude)
        mean_res = (mean_base + mean_time) * float(resource_scale)
        return (
            f"Mean resource per site ≈ {mean_res:.3f}; maintenance per individual ≈ {(NN_NODES+NN_NODES2)*maint:.3f}. "
            f"Estimated K: {label} (raw {k_val:.1f})."
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="Interactive evo-movement-sim dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    app = build_app()
    # dash>=3 replaces run_server with run
    app.run(debug=False, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
