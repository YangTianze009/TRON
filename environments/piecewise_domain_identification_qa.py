"""
Piecewise Domain Identification QA.

2026-05-05 R5 B5B (FINAL): REWRITTEN as pure single-letter MCQ. Given a
piecewise function plot with branches drawn in distinct colors, identify which
colored branch matches a stated domain (e.g. "x ≥ 0", "x < -2").

Per-level design:
  L0/L1 — 4-MCQ. 2-piece function, ask "Which branch covers x ≥ 0?".
  L2-L4 — 5-MCQ. 3-piece. 10/15% trap.
  L5-L7 — 5-MCQ. 4-piece. 20% trap.
  L8/L9 — 5-MCQ. 5-piece. 30/40% trap. Tighter distractor domains.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_COLOR_NAMES = [
    ("red", "#e74c3c"),
    ("blue", "#3498db"),
    ("green", "#2ecc71"),
    ("orange", "#f39c12"),
    ("purple", "#9b59b6"),
    ("teal", "#1abc9c"),
]


def _fmt_interval(lo: float, hi: float, lo_inf=False, hi_inf=False) -> str:
    """Pretty-print an interval (lo, hi) — use ∞ when bounded by domain."""
    lo_s = "−∞" if lo_inf else f"{int(lo)}" if abs(lo - int(lo)) < 1e-3 else f"{lo:g}"
    hi_s = "∞" if hi_inf else f"{int(hi)}" if abs(hi - int(hi)) < 1e-3 else f"{hi:g}"
    return f"[{lo_s}, {hi_s}]"


class PiecewiseDomainIdentificationQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive (axes layout)
    ENV_NAME = "piecewise_domain_identification"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        if level <= 1:
            return {
                "n_pieces": 2, "n_options": 4,
                "trap_rate": 0.0, "tight": False,
            }
        if level <= 4:
            return {
                "n_pieces": 3, "n_options": 5,
                "trap_rate": 0.10 if level <= 3 else 0.15,
                "tight": False,
            }
        if level <= 7:
            return {
                "n_pieces": 4, "n_options": 5,
                "trap_rate": 0.20, "tight": False,
            }
        # L8/L9
        return {
            "n_pieces": 5, "n_options": 5,
            "trap_rate": 0.40, "tight": True,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1289)
        for _ in range(20):
            res = self._try_generate(cfg, rng)
            if res is not None:
                return res
        return None

    def _try_generate(self, cfg, rng):
        n_pieces = cfg["n_pieces"]
        # Domain: [-5, 5]. Pick n_pieces - 1 distinct integer breakpoints.
        cand = list(range(-4, 5))
        if n_pieces - 1 > len(cand):
            return None
        breakpoints = sorted(rng.sample(cand, n_pieces - 1))
        bounds = [-5.0] + [float(b) for b in breakpoints] + [5.0]
        # pieces[i] covers [bounds[i], bounds[i+1]]
        pieces = []
        for i in range(n_pieces):
            lo, hi = bounds[i], bounds[i + 1]
            ptype = rng.choice(["linear", "constant"])
            if ptype == "linear":
                a = rng.choice([-2, -1, 1, 2])
                b = rng.randint(-3, 3)
                func = lambda x, _a=a, _b=b: _a * x + _b
            else:
                c = rng.choice([-3, -2, -1, 1, 2, 3])
                func = lambda x, _c=c: _c
            pieces.append({"lo": lo, "hi": hi, "func": func})

        # Pick n_pieces distinct colors
        colors = list(_COLOR_NAMES)
        rng.shuffle(colors)
        colors = colors[:n_pieces]
        # Assign each piece to a (color name, hex) pair
        for i, pc in enumerate(pieces):
            pc["color_name"] = colors[i][0]
            pc["color_hex"] = colors[i][1]

        # Pick the target piece, ask about its domain
        target_idx = rng.randint(0, n_pieces - 1)
        target_pc = pieces[target_idx]
        target_color = target_pc["color_name"]
        domain_str = _fmt_interval(
            target_pc["lo"], target_pc["hi"],
            lo_inf=(target_idx == 0), hi_inf=(target_idx == n_pieces - 1),
        )

        # Build options: which colored piece has domain `domain_str`?
        n_options = cfg["n_options"]
        opt_letters = ["A", "B", "C", "D", "E"][:n_options]
        last_letter = opt_letters[-1]

        # Distractor color names (from other pieces). For 5-MCQ, last is "No correct".
        # We need at most n_options - 1 color choices.
        all_color_names = [pc["color_name"] for pc in pieces]
        wrong_colors = [c for c in all_color_names if c != target_color]
        rng.shuffle(wrong_colors)

        if n_options == 4:
            # 4 colors total (or pad with extra)
            need_distractors = 3
            if len(wrong_colors) < need_distractors:
                # pad with unused color names
                used = set(all_color_names)
                for cn, _hex in _COLOR_NAMES:
                    if cn not in used and cn != target_color:
                        wrong_colors.append(cn)
                        if len(wrong_colors) >= need_distractors:
                            break
            if len(wrong_colors) < need_distractors:
                return None
            options = [target_color] + wrong_colors[:need_distractors]
            rng.shuffle(options)
            correct_letter = opt_letters[options.index(target_color)]
        else:
            # 5-MCQ with "No correct answer" trap
            n_value_slots = 4
            use_trap = (rng.random() < cfg["trap_rate"])
            if len(wrong_colors) < n_value_slots:
                used = set(all_color_names)
                for cn, _hex in _COLOR_NAMES:
                    if cn not in used and cn != target_color:
                        wrong_colors.append(cn)
                        if len(wrong_colors) >= n_value_slots + 1:
                            break
            if use_trap and len(wrong_colors) >= n_value_slots:
                chosen = wrong_colors[:n_value_slots]
                rng.shuffle(chosen)
                options = list(chosen)
                options.append("No correct answer")
                correct_letter = last_letter
            else:
                if len(wrong_colors) < n_value_slots - 1:
                    return None
                chosen = [target_color] + wrong_colors[:n_value_slots - 1]
                rng.shuffle(chosen)
                options = list(chosen)
                options.append("No correct answer")
                correct_letter = opt_letters[options.index(target_color)]

        opt_lines = "\n".join(
            f"{opt_letters[i]}. {options[i]}" for i in range(n_options)
        )
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"
        stems = [
            (f"The graph shows a piecewise function with {n_pieces} branches "
             f"drawn in distinct colors. Which colored branch is defined on "
             f"the domain x ∈ {domain_str}?"),
            (f"As shown in the diagram, a piecewise function f(x) is plotted "
             f"with {n_pieces} differently-colored branches. Which color "
             f"corresponds to the branch with domain x ∈ {domain_str}?"),
            (f"In the figure, the piecewise function is broken into "
             f"{n_pieces} colored pieces. Identify the color of the piece "
             f"whose domain is x ∈ {domain_str}."),
        ]
        stem = rng.choice(stems)
        question = (
            f"{stem}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )

        img = self._render(pieces, rng)
        return question, correct_letter, img

    def _render(self, pieces, rng):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7.5 * sc, 5.0 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])

        for i, pc in enumerate(pieces):
            xs = np.linspace(pc["lo"], pc["hi"], 200)
            ys = np.array([pc["func"](float(x)) for x in xs])
            ax.plot(xs, ys, color=pc["color_hex"], linewidth=2.6,
                    label=pc["color_name"])
            # Open/closed dots at boundaries
            if i > 0:
                ax.plot(pc["lo"], pc["func"](pc["lo"]), "o",
                        markerfacecolor="white",
                        markeredgecolor=pc["color_hex"],
                        markersize=8, markeredgewidth=2, zorder=5)
            if i < len(pieces) - 1:
                ax.plot(pc["hi"], pc["func"](pc["hi"]), "o",
                        markerfacecolor=pc["color_hex"],
                        markeredgecolor=pc["color_hex"],
                        markersize=8, zorder=5)

        ax.axhline(0, color="#333333", linewidth=1.0)
        ax.axvline(0, color="#333333", linewidth=1.0)
        ax.grid(True, color="#cccccc", linewidth=0.6, alpha=0.7)
        ax.set_xlim(-5.5, 5.5)
        ax.set_xticks(np.arange(-5, 6, 1))
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("f(x)", fontsize=11)
        ax.set_title("Piecewise function (colored branches)",
                     fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=9)

        return self.fig_to_pil(fig, dpi=style["dpi"])


if __name__ == "__main__":
    env = PiecewiseDomainIdentificationQA()
    for level in (0, 3, 6, 9):
        for seed in (1, 7, 42):
            ok = env.generate(seed=seed, parameter={"level": level})
            print(f"L{level} s{seed}: ok={ok} A={env._answer}")
