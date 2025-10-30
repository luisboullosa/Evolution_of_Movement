def setup_env_axis(fig, gs):
    """Create and return the environment axis from the provided gridspec.

    Keeps the caller logic simple and centralizes any future layout tweaks.
    """
    ax_env = fig.add_subplot(gs[0, 0])

    return ax_env


def draw_env_line(ax_env, env_idx, env_res):
    """Draw the environment resource line and return the Line2D artist."""
    (ln_env,) = ax_env.plot(env_idx, env_res, label="resource", color="0.3")
    ax_env.set_xlim(env_idx.min(), env_idx.max())

    return ln_env


def update_env(
    ax_env, ln_env, env_idx, env_res, pos, env_global_max, times_i, sel_id, resources
):
    """Update the environment line and axis titles for a new timepoint."""
    ln_env.set_ydata(env_res)
    ax_env.set_ylim(0, max(env_global_max * 1.2, 1.0))
    ax_env.set_title(
        f"Environment (t={times_i}) - ind {sel_id} pos={pos} res={resources:.2f}"
    )
    ax_env.set_xlabel("position")
    ax_env.set_ylabel("resource")
