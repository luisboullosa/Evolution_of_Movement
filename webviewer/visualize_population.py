"""
Population visualizer: clickable map of individuals with reaction-norm panel.
Usage:
  python webviewer/visualize_population.py [webviewer] [--headless]

This reads the same files as `visualize_individual.py` (env.jsonl, individuals.jsonl,
nn_index.json) and shows all individuals on the environment as a scatter. Click an
individual to select it; the right-hand panel shows that individual's reaction
norm (using the same g_nn.compute_nn_response_for helper) and a small NN diagram.

Headless mode: pass --headless to save screenshot to <simdir>/population_screenshot.png
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import random

import graphs_env as g_env
import graphs_nn as g_nn
import graphs_resp_fitness as g_rf

# first arg: base folder (default 'webviewer')
arg = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "webviewer"
root_arg = arg
base = arg

# discover available sim folders (if arg is a directory)
def discover_sims(arg_path):
    if not os.path.isdir(arg_path):
        return []
    sims = sorted([d for d in os.listdir(arg_path) if os.path.isdir(os.path.join(arg_path, d)) and d.startswith("sim")])
    return sims

sims = []
if os.path.isdir(arg):
    sims = discover_sims(arg)
    if len(sims) == 1:
        base = os.path.join(arg, sims[0])
    elif len(sims) > 1:
        # default to latest sim
        base = os.path.join(arg, sims[-1])

# headless detection: if the script is run headless we prefer to load only
# the last (most recent) sim rather than scanning many subfolders.
explicit_headless = any(a in ("--headless", "--screenshot") for a in sys.argv)

# globals that will be (re)populated by load_sim
sim_list = []


def try_load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def sim_supports_display(sim_path):
    """Return True if the sim folder contains valid env.jsonl and individuals.jsonl
    with at least one parseable JSON line each. This filters out sims that are
    missing files or are corrupted.
    """
    env_jsonl = os.path.join(sim_path, "env.jsonl")
    ind_jsonl = os.path.join(sim_path, "individuals.jsonl")
    if not (os.path.exists(env_jsonl) and os.path.exists(ind_jsonl)):
        return False
    try:
        # try to parse first non-empty line from each file
        with open(env_jsonl, 'r', encoding='utf-8') as ef:
            for el in ef:
                el = el.strip()
                if not el:
                    continue
                json.loads(el)
                break
        with open(ind_jsonl, 'r', encoding='utf-8') as inf:
            for il in inf:
                il = il.strip()
                if not il:
                    continue
                json.loads(il)
                break
        return True
    except Exception:
        return False


def load_sim(sim_base):
    global env_list, ind_list, times, cur_frame, environment, individuals, n_frames, env_global_max, nn_index, ids, positions, i_resources, sim_params

    env_list = []
    ind_list = []
    times = []

    env_jsonl = os.path.join(sim_base, "env.jsonl")
    ind_jsonl = os.path.join(sim_base, "individuals.jsonl")
    nn_index_file = os.path.join(sim_base, "nn_index.json")

    if not (os.path.exists(env_jsonl) and os.path.exists(ind_jsonl)):
        raise FileNotFoundError(f"No env/individual snapshot files found in {sim_base}")

    # load all frames into memory for animation (streaming parse paired lines)
    with open(env_jsonl, "r", encoding='utf-8') as ef, open(ind_jsonl, "r", encoding='utf-8') as inf:
        for el, il in zip(ef, inf):
            try:
                eobj = json.loads(el)
                iobj = json.loads(il)
            except Exception:
                continue
            env_list.append(eobj.get("resources", []))
            ind_list.append(iobj.get("individuals", []))
            times.append(eobj.get("t", None))

    if len(env_list) == 0 or len(ind_list) == 0:
        raise RuntimeError("No valid frames found in sim folder")

    # start at frame 0
    cur_frame = 0
    environment = np.array(env_list[cur_frame])
    individuals = ind_list[cur_frame]
    n_frames = min(len(env_list), len(ind_list))

    # compute a global max for stable y-axis across animation
    env_global_max = 0.0
    for r in env_list:
        try:
            if len(r) > 0:
                env_global_max = max(env_global_max, float(np.max(r)))
        except Exception:
            pass
    if env_global_max <= 0.0:
        env_global_max = 1.0

    # load nn index if present
    nn_index = None
    if os.path.exists(nn_index_file):
        try:
            with open(nn_index_file, "r", encoding='utf-8') as f:
                nn_index = json.load(f)
        except Exception:
            nn_index = None

    # try to read any config/params JSONs in this sim folder
    sim_params = {}
    for fname in ["config.json", "sim_config.json", "scan_config.json", "run_config.json"]:
        p = os.path.join(sim_base, fname)
        if os.path.exists(p):
            j = try_load_json(p)
            if isinstance(j, dict):
                sim_params.update(j)
    # also try nn_index (it sometimes stores metadata)
    #if nn_index and isinstance(nn_index, dict):
    #    sim_params.setdefault('nn_index_keys', list(nn_index.keys())[:10])

    # build arrays for plotting
    ids = []
    positions = []
    i_resources = []
    for ind in individuals:
        try:
            ids.append(int(ind.get("id", 0)))
            positions.append(int(ind.get("pos", 0)))
            i_resources.append(float(ind.get("resources", 0.0)))
        except Exception:
            continue

    ids = np.array(ids)
    positions = np.array(positions)
    i_resources = np.array(i_resources)

    return {
        'env_list': env_list,
        'ind_list': ind_list,
        'times': times,
        'n_frames': n_frames,
        'env_global_max': env_global_max,
        'nn_index': nn_index,
        'sim_params': sim_params,
    }

# attempt initial load
def find_and_load_initial(base_path, root_path):
    """Try to load the provided base path; if it fails and root_path is a
    directory containing multiple sim subfolders, try each sim subfolder until
    one loads. Raise the last exception if none succeed.
    """
    last_exc = None
    try:
        return load_sim(base_path)
    except Exception as e:
        last_exc = e
    # If root_path is a directory, and we are *not* running headless, try sim
    # subfolders (prefer newest first). For headless runs we only attempt the
    # provided base (which should already point to the last sim) to avoid
    # scanning many folders.
    if not explicit_headless and os.path.isdir(root_path):
        sims = discover_sims(root_path)
        for s in reversed(sims):
            candidate = os.path.join(root_path, s)
            try:
                return load_sim(candidate)
            except Exception as e:
                last_exc = e
                continue
    # nothing worked
    raise last_exc if last_exc is not None else RuntimeError('No sim folders found')


if os.path.isdir(root_arg):
    # discover sim subfolders and filter to only those that support display
    raw_sims = discover_sims(root_arg)
    sim_list = [s for s in raw_sims if sim_supports_display(os.path.join(root_arg, s))]
    # if no displayable sims found but the base itself is a sim folder, try that
    if not sim_list and os.path.basename(base).startswith('sim'):
        if sim_supports_display(base):
            sim_list = [os.path.basename(base)]
    if sim_list:
        try:
            current_sim_index = sim_list.index(os.path.basename(base))
        except Exception:
            current_sim_index = max(0, len(sim_list)-1)
i_resources = np.array([])
sim_params = {}
# movement overlay controls
show_movement = False
movement_artists = []

try:
    _meta = find_and_load_initial(base, root_arg)
    env_list = _meta['env_list']
    ind_list = _meta['ind_list']
    times = _meta['times']
    n_frames = _meta['n_frames']
    env_global_max = _meta['env_global_max']
    nn_index = _meta['nn_index']
    sim_params = _meta['sim_params']
except Exception as e:
    # Friendly error: only complain if no sim subfolder contained valid snapshots
    print(f"Failed to find a sim with valid env/individual snapshots: {e}")
    sys.exit(1)



# setup figure with two panels: left env map, right reaction-norm + NN
fig = plt.figure(figsize=(12,6))

try:
    fig.canvas.manager.set_window_title("Environment and individuals (click a dot)")
except Exception:
    try:
        plt.get_current_fig_manager().set_window_title("Environment and individuals (click a dot)")
    except Exception:
        pass

ax_map = fig.add_subplot(1,2,1)
ax_info = fig.add_subplot(2,2,2)
ax_resp = fig.add_subplot(2,2,4)

# --- UI: sim chooser (dropdown fallback to prev/next) and params display ---
# `sim_list` is computed earlier and already filtered to displayable sims.
# Ensure it's available; don't repopulate with unfiltered sims here.
current_sim_index = globals().get('current_sim_index', 0)
if 'sim_list' not in globals() or not isinstance(sim_list, list):
    sim_list = []
    if os.path.isdir(root_arg):
        raw_sims = discover_sims(root_arg)
        sim_list = [s for s in raw_sims if sim_supports_display(os.path.join(root_arg, s))]
        if not sim_list and os.path.basename(base).startswith('sim'):
            if sim_supports_display(base):
                sim_list = [os.path.basename(base)]
if sim_list:
    try:
        current_sim_index = sim_list.index(os.path.basename(base))
    except Exception:
        current_sim_index = max(0, len(sim_list)-1)

# small dropdown-like selector (popup RadioButtons) + prev/next buttons at the top for sim selection
ax_simbox = fig.add_axes([0.01, 0.92, 0.20, 0.045])
from matplotlib.widgets import RadioButtons
sim_button = Button(ax_simbox, sim_list[current_sim_index] if sim_list else os.path.basename(base))
ax_prev = fig.add_axes([0.22, 0.92, 0.03, 0.045])
ax_next = fig.add_axes([0.26, 0.92, 0.03, 0.045])
btn_prev = Button(ax_prev, '<')
btn_next = Button(ax_next, '>')
ax_move_toggle = fig.add_axes([0.30, 0.92, 0.09, 0.045])
btn_move = Button(ax_move_toggle, 'Show move: Off')

# params display area
ax_params = fig.add_axes([0.72, 0.92, 0.27, 0.07])
ax_params.axis('off')
params_text_obj = ax_params.text(0, 1.0, '', va='top', family='monospace', wrap=True)
# population count display (small label on the map)
pop_text_obj = ax_map.text(0.02, 0.98, '', transform=ax_map.transAxes, va='top', ha='left', family='monospace', fontsize=9)

def render_sim_params():
    try:
        if not sim_params:
            params_text_obj.set_text('No simulation parameters found')
            return

        def fmt_val(v):
            # Try to format numbers compactly, otherwise fall back to str()
            try:
                if isinstance(v, float):
                    # 3 significant digits, scientific for very small/large
                    if abs(v) >= 0.001 and abs(v) < 1000:
                        return f"{v:.3g}"
                    else:
                        return f"{v:.3e}"
                if isinstance(v, int):
                    return str(v)
                # for lists/dicts, represent briefly
                if isinstance(v, (list, tuple)):
                    if len(v) > 6:
                        return str(v[:6]) + '...'
                    return str(v)
                if isinstance(v, dict):
                    # show keys only
                    return '{' + ','.join(list(v.keys())[:6]) + ('...' if len(v.keys())>6 else '') + '}'
                return str(v)
            except Exception:
                return '<unprintable>'

        parts = []
        for k in sorted(sim_params.keys()):
            try:
                parts.append(f"{k}: {fmt_val(sim_params[k])}")
            except Exception:
                parts.append(f"{k}: <unprintable>")
        params_text_obj.set_text('\n'.join(parts))
    except Exception:
        params_text_obj.set_text('Failed to render params')

def on_sim_change(sim_name):
    global base, env_list, ind_list, times, n_frames, env_global_max, nn_index, sim_params, cur_frame
    # compute new base path
    if os.path.isdir(root_arg):
        new_base = os.path.join(root_arg, sim_name)
    else:
        new_base = sim_name
    try:
        meta = load_sim(new_base)
    except Exception as e:
        print('Failed to load sim', new_base, e)
        return
    # update globals
    env_list = meta['env_list']
    ind_list = meta['ind_list']
    times = meta['times']
    n_frames = meta['n_frames']
    env_global_max = meta['env_global_max']
    nn_index = meta['nn_index']
    sim_params = meta['sim_params']
    base = new_base
    # reset frame and UI
    cur_frame = 0
    slider.valmax = max(0, n_frames-1)
    slider.ax.set_xlim(slider.valmin, slider.valmax)
    slider.set_val(0)
    update_scatter_data(0)
    plot_for_selection(None)
    render_sim_params()

def prev_sim(event):
    global current_sim_index
    if not sim_list:
        return
    current_sim_index = (current_sim_index - 1) % len(sim_list)
    try:
        btn_text = sim_list[current_sim_index]
        sim_button.label.set_text(btn_text)
    except Exception:
        pass
    on_sim_change(sim_list[current_sim_index])

def next_sim(event):
    global current_sim_index
    if not sim_list:
        return
    current_sim_index = (current_sim_index + 1) % len(sim_list)
    try:
        btn_text = sim_list[current_sim_index]
        sim_button.label.set_text(btn_text)
    except Exception:
        pass
    on_sim_change(sim_list[current_sim_index])

btn_prev.on_clicked(prev_sim)
btn_next.on_clicked(next_sim)
def toggle_movement(event):
    global show_movement
    show_movement = not show_movement
    try:
        btn_move.label.set_text('Show move: On' if show_movement else 'Show move: Off')
    except Exception:
        pass
    # refresh current frame overlays
    try:
        update_scatter_data(cur_frame)
    except Exception:
        pass

btn_move.on_clicked(toggle_movement)

def open_sim_dropdown(event):
    """Open a small popup with radio buttons to choose a sim from sim_list."""
    if not sim_list:
        return
    try:
        # small popup figure
        pop = plt.figure(figsize=(3, min(6, 0.2 + 0.3 * len(sim_list))))
        axr = pop.add_axes([0.05, 0.05, 0.9, 0.9])
        radios = RadioButtons(axr, sim_list, active=current_sim_index)

        def _on_pick(label):
            try:
                idx = sim_list.index(label)
            except Exception:
                idx = 0
            # update main UI button label and load sim
            try:
                sim_button.label.set_text(label)
            except Exception:
                pass
            # set the global index and change
            global current_sim_index
            current_sim_index = sim_list.index(label)
            on_sim_change(label)
            try:
                plt.close(pop)
            except Exception:
                pass

        radios.on_clicked(_on_pick)
    except Exception as ex:
        print('Failed to open sim selector popup', ex)

sim_button.on_clicked(open_sim_dropdown)


# draw environment line
env_idx = np.arange(len(environment))
ln_env = g_env.draw_env_line(ax_map, env_idx, environment)
# ensure environment line has a low zorder so individuals plot on top
try:
    if hasattr(ln_env, "set_zorder"):
        ln_env.set_zorder(2)
    else:
        for _ln in ln_env:
            try:
                _ln.set_zorder(2)
            except Exception:
                pass
except Exception:
    pass

# plot individuals exactly at the environment resource values for their positions
# ensure we use data coordinates (transData) and place markers above the line
sc = ax_map.scatter(positions, environment[positions], c=i_resources, cmap="viridis", s=40, alpha=0.01, zorder=5, transform=ax_map.transData)
ax_map.set_xlim(env_idx.min(), env_idx.max())
# Use global env max so the y-axis doesn't jump during animation
ax_map.set_ylim(0, max(env_global_max * 1.2, 1.0))

selected = {"id": None, "index": None}

# helper to build input vector: left, center, right, occupancy, resources
def build_input_for(idx, ind_rec):
    w = len(environment)
    pos = int(ind_rec.get("pos", 0))
    left_idx = (pos - 1) % w
    right_idx = (pos + 1) % w
    left_val = float(environment[left_idx]) if left_idx < len(environment) else 0.0
    center_val = float(environment[pos]) if pos is not None and pos < len(environment) else 0.0
    right_val = float(environment[right_idx]) if right_idx < len(environment) else 0.0
    try:
        occ = int(np.sum([1 for x in individuals if int(x.get("pos",0))==pos]))
    except Exception:
        occ = 0
    res_val = float(ind_rec.get("resources", 0.0))
    return [left_val, center_val, right_val, float(occ), float(res_val)]

# reaction norm plotting helper
xs = np.linspace(0, max(float(environment.max())*1.2 if environment.size>0 else 1.0, 1.0), 300)

def plot_for_selection(sel_idx):
    ax_info.clear()
    ax_resp.clear()
    if sel_idx is None:
        ax_info.text(0.1,0.5, "No selection", transform=ax_info.transAxes)
        fig.canvas.draw_idle()
        return
    ind_rec = individuals[sel_idx]
    sel_id = int(ind_rec.get("id",0))
    pos = int(ind_rec.get("pos",0))
    res = float(ind_rec.get("resources",0.0))
    ax_info.text(0.02,0.95, f"id: {sel_id}\npos: {pos}\nres: {res:.3f}", transform=ax_info.transAxes, va='top', family='monospace')

    # try to get nn
    use_nn = None
    if nn_index is not None:
        key = str(sel_id)
        if key in nn_index:
            use_nn = nn_index[key]

    if use_nn is None:
        ax_resp.text(0.5,0.5, "NN not found", ha='center')
        fig.canvas.draw_idle()
        return

    # safer: use graphs_nn.make_input via compute_nn_response_for's input_builder
    def input_builder(xval, l=None, r=None, o=None, res_val=None):
        # build consistent with visualize_individual.make_input
        left_const = float(environment[(pos-1)%len(environment)])
        right_const = float(environment[(pos+1)%len(environment)])
        occ_val = float(np.sum([1 for x in individuals if int(x.get("pos",0))==pos]))
        def mk(cval, l=left_const, r=right_const, o=occ_val, res_val=res):
            vec = [l, cval, r]
            # respect nn_use_occupancy if specified
            nn_use_occ = True
            try:
                if isinstance(use_nn, dict) and "nn_use_occupancy" in use_nn:
                    nn_use_occ = bool(use_nn.get("nn_use_occupancy", True))
            except Exception:
                nn_use_occ = True
            if nn_use_occ:
                vec.append(o)
            vec.append(res_val)
            return vec
        return mk(xval)

    ys = g_nn.compute_nn_response_for(xs, use_nn, input_builder=lambda xv: input_builder(xv))
    ax_resp.plot(xs, ys)
    ax_resp.set_title(f"Reaction norm for id {sel_id}")
    fig.canvas.draw_idle()


def find_index_by_id(ind_id):
    """Return the index of individual with given id in the current frame, or None."""
    try:
        lst = ind_list[cur_frame]
    except Exception:
        return None
    for i, rec in enumerate(lst):
        try:
            if int(rec.get('id', -1)) == int(ind_id):
                return i
        except Exception:
            continue
    return None

# click handler
def on_click(event):
    global cur_frame, individuals, environment
    if event.inaxes != ax_map:
        return
    # find nearest point
    if len(positions)==0:
        return
    x = int(round(event.xdata))
    # find individuals at that position in the current frame, pick first
    matches = [i for i, rec in enumerate(ind_list[cur_frame]) if int(rec.get("pos",0))==x]
    if not matches:
        return
    sel = matches[0]
    selected['id'] = int(ind_list[cur_frame][sel].get('id',0))
    selected['index'] = sel
    # update current frame's indivs and env_res before plotting
    individuals = ind_list[cur_frame]
    environment = np.array(env_list[cur_frame])
    # refresh global arrays used for scatter (positions, i_resources)
    update_scatter_data(cur_frame)
    plot_for_selection(sel)

fig.canvas.mpl_connect('button_press_event', on_click)

# add simple play/pause and slider controls
from matplotlib.widgets import Slider, Button
from matplotlib.patches import FancyArrowPatch

# axes for controls
ax_play = fig.add_axes([0.45, 0.01, 0.1, 0.04])
btn_play = Button(ax_play, 'Play')
ax_slider = fig.add_axes([0.1, 0.01, 0.3, 0.04])
slider = Slider(ax_slider, 'frame', 0, n_frames - 1, valinit=0, valstep=1)

playing = {'val': False}
timer = None

def update_scatter_data(frame_idx):
    global positions, i_resources, ids, sc, environment, individuals, ln_env, env_idx
    environment = np.array(env_list[frame_idx])
    individuals = ind_list[frame_idx]
    ids = []
    poses_list = []
    ress_list = []
    for ind in individuals:
        try:
            ids.append(int(ind.get("id", 0)))
            poses_list.append(int(ind.get("pos", 0)))
            ress_list.append(float(ind.get("resources", 0.0)))
        except Exception:
            continue
    if len(poses_list) == 0:
        positions = np.array([])
        i_resources = np.array([])
        # clear scatter
        sc.set_offsets(np.empty((0,2)))
        sc.set_array(np.array([]))
        return
    positions = np.array(poses_list)
    i_resources = np.array(ress_list)
    # redraw environment line for this frame so the resource curve matches frame
    try:
        # remove previous line(s) if present
        if 'ln_env' in globals() and ln_env is not None:
            try:
                if hasattr(ln_env, 'remove'):
                    ln_env.remove()
                else:
                    for _ln in ln_env:
                        try:
                            _ln.remove()
                        except Exception:
                            pass
            except Exception:
                pass
    except Exception:
        pass
    # compute new env index and draw
    try:
        env_idx = np.arange(len(environment))
        ln_env = g_env.draw_env_line(ax_map, env_idx, environment)
        try:
            if hasattr(ln_env, 'set_zorder'):
                ln_env.set_zorder(2)
            else:
                for _ln in ln_env:
                    try:
                        _ln.set_zorder(2)
                    except Exception:
                        pass
        except Exception:
            pass
    except Exception:
        pass

    # update scatter points - plot at resource curve y-coordinates
    offsets = np.column_stack((positions, environment[positions]))
    # offsets are in data coordinates (x=position, y=environment[position])
    sc.set_offsets(offsets)
    
    # keep y-limits stable across frames (use env_global_max computed at startup)
    try:
        ax_map.set_ylim(0, max(env_global_max * 1.2, 1.0))
    except Exception:
        pass

    # update population size label for this frame
    try:
        pop_count = len(individuals) if individuals is not None else 0
        pop_text_obj.set_text(f"Population: {pop_count}")
    except Exception:
        try:
            pop_text_obj.set_text('Population: ?')
        except Exception:
            pass

    # movement overlay: show arrows for individuals that moved from previous frame
    try:
        # clear previous overlay artists
        global movement_artists, show_movement
        for art in movement_artists:
            try:
                art.remove()
            except Exception:
                pass
        movement_artists = []

        if show_movement and frame_idx > 0:
            prev_idx = max(0, frame_idx - 1)
            prev_inds = ind_list[prev_idx]
            prev_env = np.array(env_list[prev_idx])
            # build id -> pos mapping for previous frame
            prev_pos = {}
            for rec in prev_inds:
                try:
                    prev_pos[int(rec.get('id', -1))] = int(rec.get('pos', 0))
                except Exception:
                    continue

            moved_pairs = []
            for rec in individuals:
                try:
                    cid = int(rec.get('id', -1))
                    cpos = int(rec.get('pos', 0))
                except Exception:
                    continue
                if cid in prev_pos:
                    ppos = prev_pos[cid]
                    if ppos != cpos:
                        moved_pairs.append((ppos, cpos))

            # limit number of arrows for performance
            max_arrows = 10000
            if len(moved_pairs) > max_arrows:
                # sample evenly
                step = max(1, len(moved_pairs) // max_arrows)
                moved_pairs = moved_pairs[::step]

            for (x0,x1) in moved_pairs:
                try:
                    arr = FancyArrowPatch((x0, 1.4), (x1, 1.4), arrowstyle='->', mutation_scale=8,
                                          color='red', alpha=0.35, linewidth=0.8, zorder=4)
                    ax_map.add_patch(arr)
                    movement_artists.append(arr)
                except Exception:
                    continue
    except Exception:
        # don't let overlay failures break the main UI
        pass

def on_slider(val):
    global cur_frame
    f = int(slider.val)
    cur_frame = f
    update_scatter_data(cur_frame)
    # try to preserve selection across frames by id
    if selected.get('id', None) is not None:
        idx = find_index_by_id(selected['id'])
        if idx is not None:
            selected['index'] = idx
            plot_for_selection(idx)
            return
        else:
            # selected id not present in this frame; clear displayed selection
            selected['index'] = None
            plot_for_selection(None)
            return
    # no prior selection
    plot_for_selection(None)

def play_toggle(event):
    playing['val'] = not playing['val']
    try:
        if playing['val']:
            btn_play.label.set_text('Pause')
            start_timer()
        else:
            btn_play.label.set_text('Play')
            stop_timer()
    except Exception:
        pass

def start_timer():
    global timer
    try:
        timer = fig.canvas.new_timer(interval=400)
        timer.add_callback(timer_step)
        timer.start()
    except Exception:
        timer = None

def stop_timer():
    global timer
    try:
        if timer is not None:
            timer.stop()
    except Exception:
        pass

def timer_step():
    global cur_frame
    if cur_frame < n_frames - 1:
        cur_frame += 1
        slider.set_val(cur_frame)
    else:
        # stop at last frame
        playing['val'] = False
        btn_play.label.set_text('Play')
        stop_timer()

btn_play.on_clicked(play_toggle)
slider.on_changed(on_slider)

fig.canvas.mpl_connect('button_press_event', on_click)

# initial empty plot
plot_for_selection(None)
# render params for the initially loaded sim
try:
    render_sim_params()
except Exception:
    pass

# headless support
explicit_headless = any(a in ("--headless","--screenshot") for a in sys.argv)
if explicit_headless:
    # Save the current view (frame 0 by default)
    out_path = os.path.join(base, "population_screenshot.png")
    try:
        fig.savefig(out_path, dpi=150)
        print("Saved screenshot to", out_path)
    except Exception as ex:
        print("Failed saving screenshot:", ex)
    # Also save a snapshot at a random frame during the simulation
    try:
        if n_frames > 1:
            rand_frame = random.randrange(0, n_frames)
        else:
            rand_frame = 0
        # update view to the random frame
        cur_frame = int(rand_frame)
        slider.set_val(cur_frame)
        update_scatter_data(cur_frame)
        plot_for_selection(None)
        rand_path = os.path.join(base, f"population_screenshot_random_t{cur_frame}.png")
        fig.savefig(rand_path, dpi=150)
        print("Saved random-frame screenshot to", rand_path)
    except Exception as ex:
        print("Failed saving random-frame screenshot:", ex)
else:
    plt.show()
