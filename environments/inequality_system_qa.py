"""
System of inequalities QA — feasible region identification, vertex optimization.
Targets: Functions (image-only angle reasoning), algebraic reasoning.

Capabilities: V5 (graph reading), R3 (algebraic reasoning), R5 (multi-step)

2026-05-03 (X18 / reference reasoning_val): added `chart_yesno_threshold`
question type. Renders the line(s) and shaded region as before, but asks
"Does any (x,y) in the feasible region satisfy y > T?" / "Is any specific
x such that y(x) > T?" — answer is bare `Yes` / `No`. Mirrors reference
samples IDX 349 / IDX 523 / IDX 211 (see design notes X18).
The yes/no mode uses the SAME render (lines + shaded feasible region) so
the visual style stays consistent with existing levels; only the question
phrasing + answer change.
"""
import random, math
from typing import Dict, Optional, Tuple
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv

class InequalitySystemQA(StandaloneVisualEnv):
    ENV_NAME = "inequality_system"

    def _level_config(self, level: int) -> dict:
        """Return difficulty config for a given level (0-9).

        chart_yesno_threshold (X18 / reference) mixed into L0..L4 so the model
        also trains on Y/N threshold reading (the most common X18 form).
        """
        configs = {
            0: {'question_type': 'is_point_in_region'},
            1: {'question_type': 'chart_yesno_threshold'},
            2: {'question_type': 'count_vertices'},
            3: {'question_type': 'chart_yesno_threshold'},
            4: {'question_type': 'find_vertex'},
            5: {'question_type': 'max_at_vertex'},
            6: {'question_type': 'chart_yesno_threshold'},
            7: {'question_type': 'region_area'},
            8: {'question_type': 'region_area'},
            9: {'question_type': 'region_area'},
        }
        return configs.get(max(0, min(9, level)), configs[0])

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[Tuple[str, str, Image.Image]]:

        # ── Level routing (auto-generated) ──
        level = max(0, min(9, int(parameter.get("level", 0))))
        lcfg = self._level_config(level)
        if "question_type" not in parameter or parameter.get("question_type") is None:
            parameter = dict(parameter, **lcfg)
        # Level-aware sub_rng — previously `rng = self._rng` produced the
        # SAME lines and vertices for L0 and L9 at the same seed, which made
        # the image structurally identical across levels. (Bug fix 2026-04-17.)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 2087)
        np_rng = np.random.RandomState(seed)

        question_type = parameter.get("question_type", rng.choice([
            "count_vertices", "is_point_in_region", "find_vertex",
            "max_at_vertex", "region_area",
        ]))

        # Two simple inequalities: y <= ax + b, y >= cx + d, x >= 0, y >= 0
        a = rng.choice([-2, -1, -0.5])
        b = rng.randint(4, 8)
        c = rng.choice([0.5, 1, 2])
        d = rng.randint(0, 2)

        # Find intersection of y = ax + b and y = cx + d
        if abs(a - c) < 0.01:
            return None
        x_int = (d - b) / (a - c)
        y_int = a * x_int + b

        if x_int < 0 or y_int < 0:
            return None

        # Vertices of feasible region: {y <= a x + b, y >= c x + d, x>=0, y>=0}.
        # Filter candidate corners against ALL four constraints so we never
        # include an infeasible point in the polygon (bug fix 2026-04-17).
        x_line1 = -b / a if a != 0 else 100
        x_line2 = -d / c if c != 0 else 0
        candidates = [
            (0.0, float(b)),          # line1 ∩ y-axis
            (0.0, float(d)),          # line2 ∩ y-axis
            (float(x_int), float(y_int)),  # line1 ∩ line2
            (float(x_line1), 0.0),    # line1 ∩ x-axis
            (float(x_line2), 0.0),    # line2 ∩ x-axis
            (0.0, 0.0),
        ]
        EPS = 0.05
        vertices = []
        for vx, vy in candidates:
            if vx < -EPS or vy < -EPS:
                continue
            if vy > a * vx + b + EPS:
                continue
            if vy < c * vx + d - EPS:
                continue
            vertices.append((round(max(0.0, vx), 1), round(max(0.0, vy), 1)))
        n_vertices = len(set(vertices))

        # Render
        style = self._random_style()
        fig, ax = plt.subplots(figsize=(7, 6))
        self._apply_style(fig, ax, style)
        palette = style["palette"]

        x = np.linspace(-1, 10, 300)
        y1 = a * x + b
        y2 = c * x + d

        ax.plot(x, y1, color=palette[0], linewidth=2, label=f"y = {a}x + {b}")
        ax.plot(x, y2, color=palette[1], linewidth=2, label=f"y = {c}x + {d}")

        # Shade feasible region
        y_upper = np.minimum(y1, 10)
        y_lower = np.maximum(y2, 0)
        mask = (y_upper >= y_lower) & (x >= 0)
        ax.fill_between(x[mask], y_lower[mask], y_upper[mask], alpha=0.2, color=palette[2])

        # Mark vertices
        for vx, vy in vertices:
            ax.plot(vx, vy, 'ko', markersize=6)
            ax.annotate(f"({vx},{vy})", xy=(vx, vy), xytext=(vx + 0.3, vy + 0.3),
                       fontsize=9)

        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.axvline(x=0, color='gray', linewidth=0.5)
        ax.set_xlim(-1, max(x_line1, 8) + 1)
        ax.set_ylim(-1, b + 2)
        ax.legend(fontsize=10)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_title("System of Inequalities", fontsize=13, fontweight="bold")

        # Generate Q&A
        if question_type == "count_vertices":
            question = "How many vertices does the feasible (shaded) region have?"
            answer = n_vertices
        elif question_type == "is_point_in_region":
            px = rng.randint(0, int(x_int))
            py = rng.randint(0, int(y_int))
            in_region = (py <= a * px + b) and (py >= c * px + d) and px >= 0 and py >= 0
            question = f"Is the point ({px}, {py}) inside the feasible region? Answer Yes or No."
            answer = "Yes" if in_region else "No"
        elif question_type == "find_vertex":
            question = f"What are the coordinates of the intersection point of the two lines? Give as (x, y)."
            answer = f"({round(x_int, 1)}, {round(y_int, 1)})"
        elif question_type in ("max_at_vertex", "optimize"):
            # Maximize x + y over vertices. If there's a tie, ask for max x
            # instead so the answer stays unique (GT is unambiguous).
            scores_xy = [v[0] + v[1] for v in vertices]
            best_score = max(scores_xy)
            top = [v for v, s in zip(vertices, scores_xy) if abs(s - best_score) < 1e-6]
            if len(top) == 1:
                best = top[0]
                question = "At which vertex is x + y maximized? Give coordinates as (x, y)."
            else:
                # Fall back to max-x (among original vertices); if that is
                # still tied, pick max-y within the tie. This yields a unique
                # coordinate pair under any constraint-vertex configuration.
                best = max(vertices, key=lambda v: (v[0], v[1]))
                question = "At which vertex is x maximized (break ties by largest y)? Give coordinates as (x, y)."
            answer = f"({best[0]}, {best[1]})"
        elif question_type == "region_area":
            uniq = list({v for v in vertices})
            if len(uniq) >= 3:
                # Order vertices CCW around centroid so shoelace gives the
                # correct polygon area (bug fix 2026-04-17).
                cx = sum(v[0] for v in uniq) / len(uniq)
                cy = sum(v[1] for v in uniq) / len(uniq)
                vs = sorted(uniq, key=lambda v: math.atan2(v[1] - cy, v[0] - cx))
                n = len(vs)
                area = 0
                for i in range(n):
                    j = (i + 1) % n
                    area += vs[i][0] * vs[j][1] - vs[j][0] * vs[i][1]
                area = abs(area) / 2
                question = "Estimate the area of the feasible region (round to 1 decimal)."
                answer = round(area, 1)
            else:
                return None
        elif question_type == "chart_yesno_threshold":
            # X18: "Does any (x,y) in the shaded region satisfy y > T?"
            # Use the upper line y = a*x + b (a < 0, b > 0). Max y on the
            # feasible region is at x = max(0, x_line2_y_intercept) where
            # both constraints meet. Use the larger of (b, y_int) as ceiling.
            # Compute true ceiling: feasible y-max is y_int (line1 ∩ line2)
            # if y_int <= b else b (when (0, b) is feasible).
            # Both candidates are vertices; just take the maximum.
            if not vertices:
                return None
            y_max_in_region = max(v[1] for v in vertices)
            # Pick threshold T offset from y_max by clear margin (avoid ties).
            margin = max(0.5, 0.10 * (b - 0))
            # 50/50 yes/no: half the time T < y_max - margin (Yes); else >.
            want_yes = rng.random() < 0.5
            if want_yes:
                T = y_max_in_region - margin - rng.uniform(0.2, 1.0)
                T = max(0.5, T)
                answer = "Yes"
            else:
                T = y_max_in_region + margin + rng.uniform(0.2, 1.0)
                answer = "No"
            T_val = round(T, 1)
            phrasing = rng.choice([
                f"Does any point (x, y) in the shaded feasible region satisfy y > {T_val}? Answer Yes or No.",
                f"Looking at the shaded region in the figure, is there any specific x such that y > {T_val} for some point in the region? Answer Yes or No.",
                f"From the figure, does the feasible region contain any point with y-coordinate greater than {T_val}? Answer Yes or No.",
            ])
            question = phrasing
        else:
            return None

        return question, str(answer), self.fig_to_pil(fig, dpi=style["dpi"])
