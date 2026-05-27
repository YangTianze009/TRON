"""Number Line QA -- number line with marked points and intervals.

Fixes:
  - L0 and L9 previously produced pixel-identical images (only the question
    template changed). Now each level uses its own sub-RNG, different number
    of points, different label style, different tick style.
  - Removed the numeric label ("A (3)") next to each dot — that leaked the
    coordinate into text. Now labels are just "A", "B", ... and the learner
    must read the value from the tick grid.
  - Expanded question type pool (8 types), template variants per type.
  - Diverse tick style, marker shape, marker color per seed.
  - Fixed overlapping labels by staggering label heights and only plotting
    tick numbers at integer grid positions.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import RegularPolygon
from PIL import Image

from .standalone_base import StandaloneVisualEnv

# ----------------------------------------------------------------------
# Marker shape helpers
# ----------------------------------------------------------------------
def _draw_marker(ax, x, y, shape: str, color, size, edge="black", lw=1.0):
    if shape == "circle":
        m = plt.Circle((x, y), size, facecolor=color, edgecolor=edge,
                       linewidth=lw, zorder=5)
        ax.add_patch(m)
    elif shape == "square":
        m = mpatches.Rectangle((x - size, y - size), 2 * size, 2 * size,
                               facecolor=color, edgecolor=edge,
                               linewidth=lw, zorder=5)
        ax.add_patch(m)
    elif shape == "triangle":
        m = RegularPolygon((x, y), numVertices=3, radius=size * 1.35,
                           orientation=math.pi / 2,
                           facecolor=color, edgecolor=edge,
                           linewidth=lw, zorder=5)
        ax.add_patch(m)
    elif shape == "diamond":
        m = RegularPolygon((x, y), numVertices=4, radius=size * 1.25,
                           orientation=math.pi / 4,
                           facecolor=color, edgecolor=edge,
                           linewidth=lw, zorder=5)
        ax.add_patch(m)
    elif shape == "star":
        m = RegularPolygon((x, y), numVertices=5, radius=size * 1.3,
                           facecolor=color, edgecolor=edge,
                           linewidth=lw, zorder=5)
        ax.add_patch(m)
    else:
        m = plt.Circle((x, y), size, facecolor=color, edgecolor=edge,
                       linewidth=lw, zorder=5)
        ax.add_patch(m)

class NumberLineQA(StandaloneVisualEnv):
    ENV_NAME = "number_line"

    # ---- Question type pool ----
    _QUESTION_TEMPLATES = {
        "count_in_interval": [
            "How many labeled points lie in the interval [{a}, {b}]?",
            "Count the number of labeled points whose value is between {a} and {b} inclusive.",
            "How many of the labeled points fall within the closed interval [{a}, {b}]?",
        ],
        "which_between": [
            "How many labeled points lie strictly between {la} and {lb}?",
            "Count the labeled points that are strictly between {la} and {lb} on the number line.",
        ],
        "distance": [
            "What is the distance between {la} and {lb} on the number line?",
            "Compute |{la} - {lb}|, the distance between the two labeled points.",
        ],
        "midpoint": [
            "What is the midpoint of {la} and {lb}?",
            "Find the value of the midpoint between the points labeled {la} and {lb}.",
        ],
        "closest_to": [
            "Which labeled point is closest to {val}?",
            "Of the labeled points, which one has value nearest to {val}?",
        ],
        "absolute_value_distance": [
            "What is |{la} - {lb}| + |{lb} - {lc}|? (Sum of absolute distances.)",
            "Compute the path length {la} → {lb} → {lc} on the number line.",
        ],
        "weighted_midpoint": [
            "Compute the weighted midpoint of {la} and {lb} with weights {w1} and {w2}: "
            "({w1}*{la} + {w2}*{lb})/({w1}+{w2}). Round to 2 decimals.",
            "With weights {w1} and {w2} on {la} and {lb}, what is the weighted average "
            "({w1}*{la} + {w2}*{lb})/({w1}+{w2})? Round to 2 decimals.",
        ],
        "weighted_centroid_3": [
            "Compute the weighted centroid of {la}, {lb}, {lc} with weights {w1}, {w2}, {w3}: "
            "({w1}*{la} + {w2}*{lb} + {w3}*{lc})/({w1}+{w2}+{w3}). Round to 2 decimals.",
            "Using weights {w1}, {w2}, {w3} on the points {la}, {lb}, {lc} respectively, "
            "find ({w1}*{la} + {w2}*{lb} + {w3}*{lc})/({w1}+{w2}+{w3}). Round to 2 decimals.",
        ],
        "arithmetic_chain": [
            "Compute ({la} + {lb}) * {k} - {lc}. Give the answer as a decimal (round to 2 decimals if needed).",
            "Evaluate {k} * ({la} + {lb}) - {lc} using the labeled values on the number line. Round to 2 decimals.",
        ],
        "leftmost": [
            "Which labeled point has the smallest (leftmost) value?",
            "Which labeled point is the leftmost on the number line?",
        ],
        "rightmost": [
            "Which labeled point has the largest (rightmost) value?",
            "Which labeled point is the rightmost on the number line?",
        ],
        "range": [
            "What is the range (max - min) of the values of the labeled points?",
            "Compute the difference between the largest and smallest labeled values.",
        ],
    }

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        # L0-L1 : count_in_interval / leftmost / rightmost  (read-off, trivial)
        # L2-L3 : which_between, range
        # L4-L5 : distance / midpoint / closest_to
        # L6-L7 : absolute_value_distance / range / midpoint
        # L8-L9 : weighted_midpoint
        configs = {
            0: {"qtypes": ["count_in_interval", "leftmost", "rightmost"],
                "n_range": (4, 5), "use_frac": False,
                "val_range": (-6, 9)},
            1: {"qtypes": ["count_in_interval", "leftmost"],
                "n_range": (4, 6), "use_frac": False,
                "val_range": (-6, 9)},
            2: {"qtypes": ["which_between", "range"],
                "n_range": (5, 6), "use_frac": False,
                "val_range": (-8, 12)},
            3: {"qtypes": ["which_between", "range", "count_in_interval"],
                "n_range": (5, 7), "use_frac": False,
                "val_range": (-8, 12)},
            4: {"qtypes": ["distance", "midpoint", "closest_to"],
                "n_range": (4, 6), "use_frac": False,
                "val_range": (-8, 12)},
            5: {"qtypes": ["distance", "midpoint"],
                "n_range": (5, 7), "use_frac": False,
                "val_range": (-8, 14)},
            6: {"qtypes": ["absolute_value_distance", "range"],
                "n_range": (4, 6), "use_frac": False,
                "val_range": (-10, 14)},
            7: {"qtypes": ["absolute_value_distance", "midpoint"],
                "n_range": (5, 7), "use_frac": False,
                "val_range": (-10, 14)},
            8: {"qtypes": ["weighted_midpoint", "weighted_centroid_3"],
                "n_range": (5, 7), "use_frac": False,
                "val_range": (-12, 16)},
            9: {"qtypes": ["weighted_centroid_3", "arithmetic_chain"],
                "n_range": (7, 9), "use_frac": False,
                "val_range": (-15, 20)},
        }
        return configs[level]

    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        # Per-level sub-RNG so L0 and L9 are visually different even at seed 0
        sub = random.Random((self.seed or 0) * 1000 + level * 37 + 3083)
        style = self._random_style()

        n_pts = sub.randint(*cfg["n_range"])
        vlo, vhi = cfg["val_range"]
        use_frac = cfg["use_frac"]

        # Sample n distinct points
        if use_frac:
            pts_set: set = set()
            while len(pts_set) < n_pts:
                pts_set.add(round(sub.uniform(vlo, vhi), 1))
            pts = sorted(pts_set)
        else:
            pool = list(range(vlo, vhi + 1))
            if len(pool) < n_pts:
                return None
            pts = sorted(sub.sample(pool, n_pts))

        # Label style pool
        label_style = sub.choice(["A", "P", "lower", "greek"])
        if label_style == "A":
            labels = [chr(ord("A") + i) for i in range(n_pts)]
        elif label_style == "P":
            labels = [f"P{i+1}" for i in range(n_pts)]
        elif label_style == "lower":
            labels = [chr(ord("a") + i) for i in range(n_pts)]
        else:
            greek = ["α", "β", "γ", "δ", "ε", "ζ", "η", "θ"]
            labels = greek[:n_pts]

        # Randomize label order relative to positions
        if sub.random() < 0.4:
            sub.shuffle(labels)

        # ----- Rendering -----
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(9.5 * sc, 3.0 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])

        lo, hi = min(pts) - 1.5, max(pts) + 1.5
        # Axis line
        line_color = sub.choice([style["geo_line_color"], "#222", "#444"])
        ax.arrow(lo, 0, hi - lo, 0,
                 head_width=0.12, head_length=0.3,
                 fc=line_color, ec=line_color,
                 linewidth=max(1.4, style["line_width"]),
                 length_includes_head=True,
                 zorder=2)
        # Draw a head at the other end too if wanted
        if sub.random() < 0.6:
            ax.arrow(hi, 0, -(hi - lo), 0,
                     head_width=0.12, head_length=0.3,
                     fc=line_color, ec=line_color,
                     linewidth=0,
                     length_includes_head=True,
                     zorder=2)

        # Ticks: choose spacing and style
        tick_step = sub.choice([1, 1, 2])
        tick_length = sub.uniform(0.08, 0.15)
        tick_color = sub.choice(["gray", "#555", "#333"])
        tick_fs = style["font_size_base"] - 2
        t_lo = int(math.floor(lo))
        t_hi = int(math.ceil(hi))
        for tick in range(t_lo, t_hi + 1):
            if tick_step > 1 and tick % tick_step != 0:
                # minor ticks — shorter, no label
                ax.plot([tick, tick], [-tick_length * 0.5, tick_length * 0.5],
                        color=tick_color, linewidth=0.6, alpha=0.6, zorder=3)
                continue
            ax.plot([tick, tick], [-tick_length, tick_length],
                    color=tick_color, linewidth=1.0, zorder=3)
            ax.text(tick, -tick_length - 0.22, str(tick),
                    ha="center", va="top",
                    fontsize=tick_fs, color=tick_color,
                    fontfamily=style["font_family"])

        # Marker shape + colors
        marker_shape = sub.choice(["circle", "circle", "square", "diamond",
                                   "triangle", "star"])
        marker_size = sub.uniform(0.11, 0.18)
        palette = list(style["palette"])
        sub.shuffle(palette)

        # Alternate label heights so they never overlap
        label_y_hi = sub.uniform(0.32, 0.45)
        label_y_lo = sub.uniform(0.55, 0.70)

        for i, (v, lbl) in enumerate(zip(pts, labels)):
            color = palette[i % len(palette)]
            _draw_marker(ax, v, 0, marker_shape, color, marker_size,
                         edge="#222", lw=1.1)
            ly = label_y_hi if i % 2 == 0 else label_y_lo
            # Label text ONLY (no numeric coordinate leak)
            ax.text(v, ly, lbl,
                    ha="center", va="bottom",
                    fontsize=style["font_size_base"] + 1,
                    fontweight="bold",
                    color=color,
                    fontfamily=style["font_family"])

        ax.set_xlim(lo - 0.5, hi + 0.5)
        ax.set_ylim(-0.9, 1.15)
        ax.axis("off")

        titles = ["Number Line", "Labeled Points on the Number Line",
                  "Number Line Diagram", "Points on a Number Line"]
        ax.set_title(sub.choice(titles),
                     fontsize=style["font_size_base"] + 1,
                     fontweight="bold",
                     fontfamily=style["font_family"])

        img = self.fig_to_pil(fig, dpi=style["dpi"])

        # ----- Question routing -----
        qtype = sub.choice(cfg["qtypes"])
        templates = self._QUESTION_TEMPLATES[qtype]

        if qtype == "count_in_interval":
            # Pick a sensible [a, b] interval
            center = sub.uniform(min(pts) - 1, max(pts) + 1)
            half = sub.uniform(1.5, 4.5)
            a = int(round(center - half))
            b = int(round(center + half))
            if a > b:
                a, b = b, a
            cnt = sum(1 for v in pts if a <= v <= b)
            q = sub.choice(templates).format(a=a, b=b)
            return q, str(cnt), img

        if qtype == "which_between":
            i, j = sorted(sub.sample(range(len(pts)), 2))
            between = [labels[k] for k in range(len(pts))
                       if pts[i] < pts[k] < pts[j]]
            cnt = len(between)
            q = sub.choice(templates).format(la=labels[i], lb=labels[j])
            return q, str(cnt), img

        if qtype == "distance":
            i, j = sub.sample(range(len(pts)), 2)
            d = round(abs(pts[i] - pts[j]), 2)
            q = sub.choice(templates).format(la=labels[i], lb=labels[j])
            if d == int(d):
                return q, str(int(d)), img
            return q, str(d), img

        if qtype == "midpoint":
            i, j = sub.sample(range(len(pts)), 2)
            m = round((pts[i] + pts[j]) / 2, 2)
            q = sub.choice(templates).format(la=labels[i], lb=labels[j])
            if m == int(m):
                return q, str(int(m)), img
            return q, str(m), img

        if qtype == "closest_to":
            val = round(sub.uniform(min(pts), max(pts)), 1)
            closest_idx = min(range(len(pts)), key=lambda k: abs(pts[k] - val))
            q = sub.choice(templates).format(val=val)
            return q, labels[closest_idx], img

        if qtype == "absolute_value_distance":
            if len(pts) < 3:
                return None
            i, j, k = sub.sample(range(len(pts)), 3)
            total = round(abs(pts[i] - pts[j]) + abs(pts[j] - pts[k]), 2)
            q = sub.choice(templates).format(
                la=labels[i], lb=labels[j], lc=labels[k])
            if total == int(total):
                return q, str(int(total)), img
            return q, str(total), img

        if qtype == "weighted_midpoint":
            i, j = sub.sample(range(len(pts)), 2)
            w1 = sub.randint(1, 4)
            w2 = sub.randint(1, 4)
            while w1 == w2:
                w2 = sub.randint(1, 4)
            wm = round((w1 * pts[i] + w2 * pts[j]) / (w1 + w2), 2)
            q = sub.choice(templates).format(
                la=labels[i], lb=labels[j], w1=w1, w2=w2)
            return q, str(wm), img

        if qtype == "weighted_centroid_3":
            if len(pts) < 3:
                return None
            i, j, k = sub.sample(range(len(pts)), 3)
            w1 = sub.randint(1, 5)
            w2 = sub.randint(1, 5)
            w3 = sub.randint(1, 5)
            # Ensure not all equal (otherwise just arithmetic mean)
            while w1 == w2 == w3:
                w3 = sub.randint(1, 5)
            val = round((w1 * pts[i] + w2 * pts[j] + w3 * pts[k]) / (w1 + w2 + w3), 2)
            q = sub.choice(templates).format(
                la=labels[i], lb=labels[j], lc=labels[k],
                w1=w1, w2=w2, w3=w3)
            return q, str(val), img

        if qtype == "arithmetic_chain":
            if len(pts) < 3:
                return None
            i, j, k = sub.sample(range(len(pts)), 3)
            mult = sub.choice([2, 3, 4])
            val = round(mult * (pts[i] + pts[j]) - pts[k], 2)
            q = sub.choice(templates).format(
                la=labels[i], lb=labels[j], lc=labels[k], k=mult)
            if val == int(val):
                return q, str(int(val)), img
            return q, str(val), img

        if qtype == "leftmost":
            idx = min(range(len(pts)), key=lambda k: pts[k])
            q = sub.choice(templates)
            return q, labels[idx], img

        if qtype == "rightmost":
            idx = max(range(len(pts)), key=lambda k: pts[k])
            q = sub.choice(templates)
            return q, labels[idx], img

        if qtype == "range":
            rng_val = round(max(pts) - min(pts), 2)
            q = sub.choice(templates)
            if rng_val == int(rng_val):
                return q, str(int(rng_val)), img
            return q, str(rng_val), img

        return None

if __name__ == "__main__":
    env = NumberLineQA()
    for lv in [0, 3, 6, 9]:
        for s in range(3):
            ok = env.generate(s, {"level": lv})
            print(f"L{lv} s{s}: ok={ok}, a={env._answer}, q={env._question[:60]}")
