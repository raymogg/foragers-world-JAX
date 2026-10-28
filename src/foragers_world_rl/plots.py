"""Figures and plot-ready data for an experiment's write-up.

    uv run --group plots foragers-plots results/band_sweep

Reads an experiment's output dir (summary.csv, episodes.csv, raw/, training/,
meta.json) and writes, under <dir>/figures/:

    <name>_light.png / _dark.png / _light.svg / _dark.svg
    data/<name>.csv      the exact numbers each figure plots

Every figure has a CSV next to it so the post can show a table, or redraw the
chart interactively, without re-deriving anything. Figures are faceted by band
(one panel each) rather than colouring bands, so colour only ever means policy.

Palette: the dataviz skill's validated categorical slots 1-3 (blue, orange,
aqua), stepped separately for light and dark. Aqua is under 3:1 contrast on the
light surface, so every policy is also direct-labelled or in a legend.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "ppo_greedy": "#2a78d6", "ppo_sampled": "#eb6834", "best_fixed": "#1baf7a",
        "in_band": "#2a78d6", "above": "#eb6834", "below": "#6250d6",
        "before": "#c3c2b7", "lost": "#52514e",
        "band_fill": "#898781", "band_alpha": 0.13,
        "seq": ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
        "seq_zero": "#eef3f9",   # a probability of 0: faint, but not the surface
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "ppo_greedy": "#3987e5", "ppo_sampled": "#d95926", "best_fixed": "#199e70",
        "in_band": "#3987e5", "above": "#d95926", "below": "#9085e9",
        "before": "#383835", "lost": "#898781",
        "band_fill": "#c3c2b7", "band_alpha": 0.12,
        "seq": ["#1a1a19", "#104281", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"],
        "seq_zero": "#262a30",
    },
}

POLICY_LABEL = {"ppo_greedy": "PPO (greedy)", "ppo_sampled": "PPO (sampled)",
                "best_fixed": "Best fixed action"}
POLICIES = ["ppo_greedy", "ppo_sampled", "best_fixed"]
STEPS = 500


# ---------------------------------------------------------------------- io


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


class Exp:
    def __init__(self, root: Path):
        self.root = root
        self.meta = json.loads((root / "meta.json").read_text())
        self.bands = {k: tuple(v) for k, v in self.meta["bands"].items()}
        self.best_fixed = self.meta["best_fixed"]
        self.costs = np.asarray(self.meta["env"]["move_costs"])
        self.initial_pop = self.meta["env"]["initial_population"]
        self.max_pop = self.meta["env"]["max_population"]
        self.episodes = read_csv(root / "episodes.csv")
        self._raw = {}

    def raw(self, band, policy):
        if (band, policy) not in self._raw:
            self._raw[band, policy] = dict(np.load(self.root / "raw" / f"{band}__{policy}.npz"))
        return self._raw[band, policy]

    def ep(self, band, policy, col):
        # csv writes booleans as "True"/"False"
        num = lambda v: {"True": 1.0, "False": 0.0}.get(v, None) if v in ("True", "False") \
            else float(v)
        return np.array([num(r[col]) for r in self.episodes
                         if r["band"] == band and r["policy"] == policy])

    def policy_label(self, band, policy):
        if policy == "best_fixed":
            a = int(self.best_fixed[band].split(":")[1])
            return f"Best fixed (cost {self.costs[a]:g})"
        return POLICY_LABEL[policy]

    def band_title(self, band):
        lo, hi = self.bands[band]
        return f"{band}  ·  band {lo}–{hi}"


# ---------------------------------------------------------------------- style


def style(theme: str):
    t = THEMES[theme]
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "text.color": t["ink"], "axes.labelcolor": t["ink2"],
        "xtick.color": t["muted"], "ytick.color": t["muted"],
        "axes.edgecolor": t["axis"], "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": t["grid"], "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlecolor": t["ink"],
        "axes.titlelocation": "left", "axes.titlepad": 8,
        "font.size": 9.5, "axes.labelsize": 9.5, "legend.fontsize": 9,
        "legend.frameon": False, "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round", "svg.fonttype": "none",
    })
    return t


def save(fig, out: Path, name: str, theme: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}_{theme}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}_{theme}.svg", bbox_inches="tight")
    plt.close(fig)


def suptitle(fig, title, subtitle, t):
    fig.suptitle(title, x=0.01, y=1.0, ha="left", va="bottom", fontsize=13,
                 fontweight="semibold", color=t["ink"])
    fig.text(0.01, 0.995, subtitle, ha="left", va="top", fontsize=9.5, color=t["ink2"])


def shade_band(ax, band, t, axis="y"):
    lo, hi = band
    span = ax.axhspan if axis == "y" else ax.axvspan
    span(lo, hi, color=t["band_fill"], alpha=t["band_alpha"], lw=0, zorder=0)


# ---------------------------------------------------------------------- figures
#
# Each figure function computes its data, writes the CSV once (light pass), and
# draws. Returning rows keeps the data next to the drawing that uses it.


def fig_returns(e: Exp, t, out, theme):
    """Mean return per band x policy, with the 10th-90th percentile range."""
    rows = []
    for band in e.bands:
        for p in POLICIES:
            r = e.ep(band, p, "return")
            rows.append({"band": band, "band_low": e.bands[band][0], "band_high": e.bands[band][1],
                         "policy": p, "policy_label": e.policy_label(band, p),
                         "mean_return": r.mean(), "p10": np.percentile(r, 10),
                         "median": np.median(r), "p90": np.percentile(r, 90),
                         "min": r.min(), "max": r.max(),
                         "extinct_frac": e.ep(band, p, "extinct").mean()})

    bands = list(e.bands)
    fig, ax = plt.subplots(figsize=(8, 0.95 * len(bands) + 1.2))
    offs = {"ppo_greedy": -0.22, "ppo_sampled": 0.0, "best_fixed": 0.22}
    for i, band in enumerate(bands):
        for p in POLICIES:
            r = next(x for x in rows if x["band"] == band and x["policy"] == p)
            y = i + offs[p]
            ax.plot([r["p10"], r["p90"]], [y, y], color=t[p], lw=2, alpha=0.45,
                    solid_capstyle="round")
            ax.plot(r["mean_return"], y, "o", ms=7, color=t[p],
                    markeredgecolor=t["surface"], markeredgewidth=1.5, zorder=3,
                    label=POLICY_LABEL[p] if i == 0 else None)
            pct = round(100 * r["extinct_frac"])
            ext = f"  · {pct}% extinct" if pct >= 1 else ""
            ax.text(max(r["p90"], r["mean_return"]) + 6, y, f"{r['mean_return']:.0f}{ext}",
                    va="center", fontsize=8.5, color=t["ink2"])
    ax.set_yticks(range(len(bands)), [e.band_title(b) for b in bands])
    ax.invert_yaxis()
    ax.set_xlim(0, 560)
    ax.set_xticks(range(0, 501, 100))
    ax.axvline(STEPS, color=t["axis"], lw=1, ls=(0, (2, 2)))
    ax.text(STEPS, -0.6, "max 500", ha="center", fontsize=8, color=t["muted"])
    ax.set_xlabel("Return (steps in band, of 500)  ·  dot = mean, line = 10th–90th percentile")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, labelcolor=t["ink"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    suptitle(fig, "PPO beats the best fixed action in every band",
             "Mean return over 256 evaluation worlds per band", t)
    save(fig, out, "01_returns", theme)
    return rows


def fig_time_allocation(e: Exp, t, out, theme):
    """Where the 500 steps went: before entry, in band, above, below, after extinction."""
    parts = [("before", "Before first entering band"), ("in_band", "In band"),
             ("above", "Above band"), ("below", "Below band"),
             ("lost", "After extinction")]
    rows = []
    for band in e.bands:
        for p in POLICIES:
            length = e.ep(band, p, "length")
            inb = e.ep(band, p, "steps_in_band")
            above = e.ep(band, p, "steps_above_after_entry")
            below = e.ep(band, p, "steps_below_after_entry")
            before = length - inb - above - below
            assert (before >= 0).all()
            rows.append({"band": band, "policy": p, "policy_label": e.policy_label(band, p),
                         "before": before.mean(), "in_band": inb.mean(), "above": above.mean(),
                         "below": below.mean(), "lost": (STEPS - length).mean()})

    bands = list(e.bands)
    fig, axes = plt.subplots(len(bands), 1, figsize=(8, 1.15 * len(bands) + 1.3), sharex=True)
    for ax, band in zip(axes, bands):
        sub = [r for r in rows if r["band"] == band]
        for j, r in enumerate(sub):
            left = 0.0
            for key, _ in parts:
                w = r[key]
                if w <= 0:
                    continue
                # 2px surface gap between segments
                ax.barh(j, w, left=left, height=0.62, color=t[key],
                        edgecolor=t["surface"], linewidth=1.5)
                if key == "in_band" and w > 40:
                    ax.text(left + 6, j, f"{w:.0f}", va="center", fontsize=8,
                            color="#ffffff")
                left += w
        ax.set_yticks(range(len(sub)), [r["policy_label"] for r in sub], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_title(e.band_title(band), fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0, labelcolor=t["ink2"])
        ax.set_xlim(0, STEPS)
    axes[-1].set_xlabel("Steps (of 500), mean over 256 worlds  ·  white number = steps in band")
    handles = [plt.Rectangle((0, 0), 1, 1, color=t[k]) for k, _ in parts]
    fig.legend(handles, [lbl for _, lbl in parts], loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.04))
    suptitle(fig, "How each episode's 500 steps were spent",
             "Above/below are counted after the population first enters the band", t)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save(fig, out, "02_time_allocation", theme)
    return rows


def fig_population_over_time(e: Exp, t, out, theme):
    """Median population and 10th-90th percentile across worlds, per step."""
    rows = []
    bands = list(e.bands)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True)
    for ax, band in zip(axes.flat, bands):
        shade_band(ax, e.bands[band], t)
        for p in ["best_fixed", "ppo_greedy"]:
            pops = e.raw(band, p)["pops"].astype(float)
            pops = np.concatenate([np.full((pops.shape[0], 1), e.initial_pop), pops], axis=1)
            q10, q50, q90 = np.percentile(pops, [10, 50, 90], axis=0)
            x = np.arange(pops.shape[1])
            ax.fill_between(x, q10, q90, color=t[p], alpha=0.16, lw=0)
            ax.plot(x, q50, color=t[p], lw=1.4, label=e.policy_label(band, p))
            for s in range(len(x)):
                rows.append({"band": band, "policy": p, "step": s,
                             "p10": q10[s], "median": q50[s], "p90": q90[s]})
        ax.set_title(e.band_title(band), fontsize=10)
        ax.set_ylim(0, 260)
        ax.set_xlim(0, STEPS)
        ax.legend(loc="upper right", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Step")
    for ax in axes[:, 0]:
        ax.set_ylabel("Population")
    suptitle(fig, "Population over time",
             "Line = median across 256 worlds, ribbon = 10th–90th percentile, grey = target band", t)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, out, "03_population_over_time", theme)
    return rows


def fig_example_episodes(e: Exp, t, out, theme):
    """One median PPO episode per band: population above, chosen move cost below."""
    picks = {r["band"]: r["world"] for r in e.meta["recorded"]
             if r["policy"] == "ppo_greedy" and r["which"] == "median"}
    bands = [b for b in e.bands if b in picks]
    rows = []
    fig, axes = plt.subplots(2, len(bands), figsize=(3.1 * len(bands), 4.8), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    for c, band in enumerate(bands):
        raw = e.raw(band, "ppo_greedy")
        w = picks[band]
        live = raw["live"][w]
        n = int(live.sum())
        pops = np.concatenate([[e.initial_pop], raw["pops"][w][:n]])
        costs = e.costs[raw["actions"][w][:n]]
        ax = axes[0, c]
        shade_band(ax, e.bands[band], t)
        ax.plot(np.arange(n + 1), pops, color=t["ppo_greedy"], lw=1.0)
        ax.set_title(f"{e.band_title(band)}\nworld {w}, return {int(raw['rewards'][w][:n].sum())}",
                     fontsize=9.5)
        ax.set_ylim(0, max(180, pops.max() * 1.05))
        ax2 = axes[1, c]
        ax2.step(np.arange(n), costs, where="post", color=t["ppo_greedy"], lw=0.8)
        ax2.set_ylim(-0.1, e.costs.max() + 0.1)
        ax2.set_xlim(0, STEPS)
        for s in range(n):
            rows.append({"band": band, "world": w, "step": s, "pop_before": int(pops[s]),
                         "action": int(raw["actions"][w][s]), "move_cost": float(costs[s]),
                         "pop_after": int(pops[s + 1]), "reward": int(raw["rewards"][w][s])})
        ax2.set_xlabel("Step")
    axes[0, 0].set_ylabel("Population")
    axes[1, 0].set_ylabel("Move cost")
    suptitle(fig, "What the controller does: one median PPO episode per band",
             "Top: population (grey = target band). Bottom: move cost chosen each step.", t)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, "04_example_episodes", theme)
    return rows


def fig_example_zoom(e: Exp, t, out, theme, start=200, length=60):
    """Close-up of the same median episodes, so individual switches are visible."""
    picks = {r["band"]: r["world"] for r in e.meta["recorded"]
             if r["policy"] == "ppo_greedy" and r["which"] == "median"}
    bands = [b for b in e.bands if b in picks]
    rows = []
    fig, axes = plt.subplots(2, len(bands), figsize=(3.1 * len(bands), 4.8), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    s0, s1 = start, start + length
    for c, band in enumerate(bands):
        raw = e.raw(band, "ppo_greedy")
        w = picks[band]
        n = int(raw["live"][w].sum())
        if n < s1:   # episode ended early: show its last `length` steps
            s0, s1 = max(0, n - length), n
        # population when acting at step s = population after step s - 1
        pops_before = np.concatenate([[e.initial_pop], raw["pops"][w]])[s0:s1]
        costs = e.costs[raw["actions"][w][s0:s1]]
        steps = np.arange(s0, s1)
        ax = axes[0, c]
        shade_band(ax, e.bands[band], t)
        ax.plot(steps, pops_before, color=t["ppo_greedy"], lw=1.2, marker="o", ms=3)
        ax.set_title(f"{e.band_title(band)}\nworld {w}, steps {s0}–{s1 - 1}", fontsize=9.5)
        lo, hi = e.bands[band]
        ax.set_ylim(max(0, min(lo, pops_before.min()) - 15), max(hi, pops_before.max()) + 15)
        ax2 = axes[1, c]
        ax2.step(steps, costs, where="post", color=t["ppo_greedy"], lw=1.2)
        ax2.set_ylim(-0.1, e.costs.max() + 0.1)
        ax2.set_xlabel("Step")
        for s, p_, cst in zip(steps, pops_before, costs):
            rows.append({"band": band, "world": w, "step": int(s), "pop_when_acting": int(p_),
                         "move_cost": float(cst)})
        s0, s1 = start, start + length
    axes[0, 0].set_ylabel("Population")
    axes[1, 0].set_ylabel("Move cost")
    suptitle(fig, f"Close-up: {length} steps of the same episodes",
             "Each point is the population the agent saw; the cost below is what it chose in "
             "response.", t)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, "04b_example_zoom", theme)
    return rows


def fig_deviation(e: Exp, t, out, theme):
    """Distribution of each episode's largest overshoot and undershoot."""
    rows = []
    bands = list(e.bands)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    pols = ["ppo_greedy", "best_fixed"]
    for ax, (col, title) in zip(axes, [("max_overshoot", "Largest overshoot above band"),
                                       ("max_undershoot", "Largest undershoot below band")]):
        for i, band in enumerate(bands):
            for k, p in enumerate(pols):
                v = e.ep(band, p, col)
                v = v[np.isfinite(v)]
                y = i + (k - 0.5) * 0.36
                q = np.percentile(v, [10, 25, 50, 75, 90]) if len(v) else [np.nan] * 5
                rows.append({"metric": col, "band": band, "policy": p,
                             "policy_label": e.policy_label(band, p), "n": len(v),
                             "mean": v.mean() if len(v) else np.nan,
                             "p10": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p90": q[4],
                             "max": v.max() if len(v) else np.nan})
                if col == "max_undershoot" and e.bands[band][0] == 0:
                    if k == 0:
                        ax.text(2, i, "n/a — band floor is 0; failure shows as extinction",
                                va="center", fontsize=8, color=t["muted"])
                    continue
                ax.plot([q[0], q[4]], [y, y], color=t[p], lw=1.2, alpha=0.6)
                ax.plot([q[1], q[3]], [y, y], color=t[p], lw=5, alpha=0.9,
                        solid_capstyle="butt")
                ax.plot(q[2], y, "|", color=t["surface"], ms=7, mew=1.5)
                ax.plot([], [], color=t[p], lw=5, label=POLICY_LABEL[p] if i == 0 else None)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Foragers outside the band (after first entry)")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(left=0)
    axes[0].set_yticks(range(len(bands)), [e.band_title(b) for b in bands])
    axes[0].tick_params(axis="y", length=0, labelcolor=t["ink"])
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], color=t[p], lw=5) for p in pols]
    fig.legend(handles, [POLICY_LABEL[p] for p in pols], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.06))
    suptitle(fig, "How far the population strays from the band",
             "Per-episode maximum. Thick bar = middle 50% of worlds, thin line = 10th–90th "
             "percentile, notch = median.", t)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    save(fig, out, "05_deviation", theme)
    return rows


def fig_training(e: Exp, t, out, theme):
    """PPO learning curve per band, against the best fixed action's return."""
    rows = []
    bands = list(e.bands)
    fig, axes = plt.subplots(1, len(bands), figsize=(3 * len(bands), 3.2), sharey=True)
    for ax, band in zip(axes, bands):
        hist = read_csv(e.root / "training" / f"{band}.csv")
        x = np.array([float(h["env_steps"]) for h in hist]) / 1e6
        y = np.array([float(h["mean_return"]) for h in hist])
        ok = np.isfinite(y)
        ax.plot(x[ok], y[ok], color=t["ppo_greedy"], lw=1.4, label="PPO (training episodes)")
        fixed = e.ep(band, "best_fixed", "return").mean()
        ax.axhline(fixed, color=t["best_fixed"], lw=1.2, ls=(0, (4, 3)),
                   label=e.policy_label(band, "best_fixed"))
        ax.text(x[-1], fixed + 8, f"best fixed {fixed:.0f}", ha="right", fontsize=8,
                color=t["ink2"])
        ax.set_title(e.band_title(band), fontsize=9.5)
        ax.set_xlabel("Env steps (millions)")
        ax.set_ylim(0, 510)
        for h, xx in zip(hist, x):
            rows.append({"band": band, "update": int(h["update"]), "env_steps": int(h["env_steps"]),
                         "mean_return": h["mean_return"], "episodes": h["episodes"],
                         "entropy": h["entropy"], "best_fixed_return": fixed})
    axes[0].set_ylabel("Mean episode return")
    suptitle(fig, "Learning curves",
             "Return of episodes finishing in each update (sampled actions, so below greedy "
             "eval). Dashed = best fixed action.", t)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, out, "06_training", theme)
    return rows


def fig_action_usage(e: Exp, t, out, theme):
    """Share of steps at each move cost, PPO greedy vs the best fixed action."""
    rows = []
    bands = list(e.bands)
    n_a = len(e.costs)
    mat = np.zeros((len(bands), n_a))
    for i, band in enumerate(bands):
        raw = e.raw(band, "ppo_greedy")
        acts = raw["actions"][raw["live"]]
        mat[i] = np.bincount(acts, minlength=n_a) / len(acts)
        for a in range(n_a):
            rows.append({"band": band, "action": a, "move_cost": e.costs[a],
                         "frac_steps_ppo_greedy": mat[i, a]})
    cmap = LinearSegmentedColormap.from_list("seq", t["seq"])
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(bands) + 1.6))
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=max(0.6, mat.max()), aspect="auto")
    for i, band in enumerate(bands):
        a_fix = int(e.best_fixed[band].split(":")[1])
        ax.add_patch(plt.Rectangle((a_fix - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=t["best_fixed"], lw=1.8))
        for a in range(n_a):
            if mat[i, a] >= 0.005:
                ax.text(a, i, f"{100 * mat[i, a]:.0f}%", ha="center", va="center", fontsize=8,
                        color="#ffffff" if mat[i, a] > 0.3 else t["ink"])
    ax.set_xticks(range(n_a), [f"{c:g}" for c in e.costs])
    ax.set_yticks(range(len(bands)), [e.band_title(b) for b in bands])
    ax.tick_params(length=0, labelcolor=t["ink2"])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Move cost  ·  outlined cell = that band's best fixed action")
    suptitle(fig, "Which move costs PPO uses",
             "Share of steps at each cost (greedy policy, all 256 worlds)", t)
    save(fig, out, "07_action_usage", theme)
    return rows


def fig_policy_map(e: Exp, t, out, theme):
    """How often each cost is chosen at each population: the learned control law."""
    rows = []
    bands = list(e.bands)
    edges = np.arange(0, 210, 5)
    # Start the ramp one step off the surface, so P = 0 (a cost never chosen at
    # that population) reads differently from a blank bin with too few steps.
    cmap = LinearSegmentedColormap.from_list("seq", [t["seq_zero"]] + t["seq"][1:])
    fig, axes = plt.subplots(1, len(bands), figsize=(3.2 * len(bands), 3.6), sharey=True)
    for ax, band in zip(axes, bands):
        raw = e.raw(band, "ppo_greedy")
        live = raw["live"]
        pop = np.rint(raw["obs"][..., 1] * e.max_pop)[live]   # population when acting
        act = raw["actions"][live]
        H, _, _ = np.histogram2d(pop, act, bins=[edges, np.arange(len(e.costs) + 1) - 0.5])
        counts = H.sum(axis=1, keepdims=True)
        P = np.where(counts >= 20, H / np.maximum(counts, 1), np.nan)  # P(cost | pop bin)
        ax.imshow(P.T, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1,
                  extent=(edges[0], edges[-1], -0.5, len(e.costs) - 0.5))
        lo, hi = e.bands[band]
        for b in (lo, hi):
            ax.axvline(b, color=t["ink2"], lw=1, ls=(0, (3, 2)))
        ax.set_title(e.band_title(band), fontsize=9.5)
        ax.set_xlabel("Population when acting")
        ax.grid(False)
        for i in range(len(edges) - 1):
            if counts[i, 0] < 20:
                continue
            for a in range(len(e.costs)):
                rows.append({"band": band, "pop_bin_low": edges[i], "pop_bin_high": edges[i + 1],
                             "action": a, "move_cost": e.costs[a], "n_steps_in_bin": int(counts[i, 0]),
                             "p_action_given_pop": P[i, a]})
    axes[0].set_yticks(range(len(e.costs))[::2], [f"{c:g}" for c in e.costs[::2]])
    axes[0].set_ylabel("Move cost chosen")
    suptitle(fig, "The learned control law",
             "Share of steps choosing each cost, given the population (greedy PPO). Palest = "
             "never; blank = fewer than 20 steps. Dashed = band edges.", t)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, out, "08_policy_map", theme)
    return rows


# (name, function). The name is used for both the image and its data CSV.
FIGURES = [
    ("01_returns", fig_returns),
    ("02_time_allocation", fig_time_allocation),
    ("03_population_over_time", fig_population_over_time),
    ("04_example_episodes", fig_example_episodes),
    ("04b_example_zoom", fig_example_zoom),
    ("05_deviation", fig_deviation),
    ("06_training", fig_training),
    ("07_action_usage", fig_action_usage),
    ("08_policy_map", fig_policy_map),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("experiment", type=Path, help="experiment output dir, e.g. results/band_sweep")
    a = ap.parse_args()
    e = Exp(a.experiment)
    out = a.experiment / "figures"
    for theme in THEMES:
        t = style(theme)
        for name, fig_fn in FIGURES:
            rows = fig_fn(e, t, out, theme)
            if theme == "light":
                write_csv(out / "data" / f"{name}.csv", rows)
    print(f"wrote {len(FIGURES)} figures x {len(THEMES)} themes to {out}/")


if __name__ == "__main__":
    main()
