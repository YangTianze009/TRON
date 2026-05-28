"""
Shape Counting Analogy QA environment (batch 2 Part B, 2026-04-14).

Goal: conditional counting with attribute filters. "How many blue
triangles are there, if we exclude big ones?" Targets a logic benchmark Attribute
Reasoning, a puzzle benchmark analogical, a logic benchmark Quantitative
Reasoning.

Difficulty axes:
  A) Pattern A n_shapes (6..15).
  B) Pattern C n_condition_terms (1..4 filters per query).
  C) Pattern B distractor_gap at L≥7.

Format: 4-way MCQ (letter).
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_SHAPE_NAMES = ["circle", "square", "triangle", "hexagon", "diamond",
                "star", "pentagon"]
_SHAPE_NAMES_L0 = ["circle", "square", "triangle"]      # smaller pool at L0
_COLOR_NAMES = {"#e74c3c": "red", "#3498db": "blue",
                "#2ecc71": "green", "#f1c40f": "yellow",
                "#9b59b6": "purple", "#ff7f50": "orange",
                "#17a2b8": "teal"}
_COLOR_NAMES_L0 = {"#e74c3c": "red", "#3498db": "blue",
                   "#2ecc71": "green"}          # smaller pool at L0

# Multiple question templates for variety.
_QUERY_TEMPLATES = [
    "How many {q} are there?",
    "Count the number of {q} shown.",
    "The image contains several figures; how many {q} appear?",
    "Determine how many {q} are displayed.",
]

class ShapeCountingAnalogyQA(StandaloneVisualEnv):
    ENV_NAME = "shape_counting_analogy"

    def _level_config(self, level: int) -> Dict:
        # 2026-05-04 R4: full-gradient redesign per a logic benchmark attribute reasoning.
        # L0: trivial single-attribute (3 shapes, 5 figures, 1 condition, no size)
        # L1-L2: 6-7 figures, color-or-shape filter (1 cond)
        # L3-L4: 8-9 figures, color+shape filter (2 cond), introduces size
        # L5-L6: 11-12 figures, +big filter (3 cond), tight distractors
        # L7-L8: 14-16 figures, +negation (4 cond), all attributes
        # L9: 20 figures, 5 conditions (color+shape+size+negation+second-shape)
        level = max(0, min(level, 9))
        if level == 0:
            return {
                "n_shapes":          5,
                "n_conditions":      1,
                "tight_distractors": False,
                "shape_pool":        list(_SHAPE_NAMES_L0),
                "color_pool":        list(_COLOR_NAMES_L0.keys()),
                "big_small":         False,
            }
        if level <= 2:
            return {
                "n_shapes":          6 + level,
                "n_conditions":      1,
                "tight_distractors": False,
                "shape_pool":        list(_SHAPE_NAMES[:5]),
                "color_pool":        list(_COLOR_NAMES.keys())[:5],
                "big_small":         False,
            }
        if level <= 4:
            return {
                "n_shapes":          8 + (level - 3),  # L3=8, L4=9
                "n_conditions":      2,
                "tight_distractors": level >= 4,
                "shape_pool":        list(_SHAPE_NAMES[:6]),
                "color_pool":        list(_COLOR_NAMES.keys())[:6],
                "big_small":         True,
            }
        if level <= 6:
            return {
                "n_shapes":          11 + (level - 5),  # L5=11, L6=12
                "n_conditions":      3,
                "tight_distractors": True,
                "shape_pool":        list(_SHAPE_NAMES),
                "color_pool":        list(_COLOR_NAMES.keys()),
                "big_small":         True,
            }
        if level <= 8:
            return {
                "n_shapes":          14 + (level - 7) * 2,  # L7=14, L8=16
                "n_conditions":      4,
                "tight_distractors": True,
                "shape_pool":        list(_SHAPE_NAMES),
                "color_pool":        list(_COLOR_NAMES.keys()),
                "big_small":         True,
            }
        # L9: 5 conditions (adds second-shape filter via OR) on 20 figures
        return {
            "n_shapes":          20,
            "n_conditions":      5,
            "tight_distractors": True,
            "shape_pool":        list(_SHAPE_NAMES),
            "color_pool":        list(_COLOR_NAMES.keys()),
            "big_small":         True,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["n_shapes"]

        for _ in range(30):
            r = self._try_generate(rng, cfg)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng: random.Random,
                      cfg: Dict) -> Optional[Tuple[str, str, Image.Image]]:
        n = cfg["n_shapes"]
        # Random attributes for each shape.
        color_keys = list(cfg.get("color_pool", list(_COLOR_NAMES.keys())))
        shape_pool = list(cfg.get("shape_pool", list(_SHAPE_NAMES)))
        big_small = cfg.get("big_small", True)
        shapes = []
        for _ in range(n):
            shapes.append({
                "shape": rng.choice(shape_pool),
                "color": rng.choice(color_keys),
                "big": (big_small and rng.random() < 0.4),
            })

        # Build a condition query from filters.
        nc = cfg["n_conditions"]
        conditions = []
        # Ensure at least one target that matches.
        max_attempts = 8
        for attempt in range(max_attempts):
            conditions = []
            # At L0, prefer shape-only condition half the time.
            if nc == 1 and rng.random() < 0.5:
                target_shape = rng.choice(shape_pool)
                conditions.append(("shape", target_shape))
            else:
                target_color = rng.choice(color_keys)
                conditions.append(("color", target_color))
            if nc >= 2:
                target_shape = rng.choice(shape_pool)
                conditions.append(("shape", target_shape))
            if nc >= 3 and big_small:
                conditions.append(("big", True))
            if nc >= 4:
                # Add a negation (not shape X)
                neg_shape = rng.choice([s for s in shape_pool
                                         if (nc < 2 or s != target_shape)])
                conditions.append(("not_shape", neg_shape))

            def _match(sh):
                for kind, val in conditions:
                    if kind == "color" and sh["color"] != val:
                        return False
                    if kind == "shape" and sh["shape"] != val:
                        return False
                    if kind == "big" and sh["big"] != val:
                        return False
                    if kind == "not_shape" and sh["shape"] == val:
                        return False
                return True

            count = sum(1 for s in shapes if _match(s))
            if count >= 1 and count < n:
                break
        else:
            return None

        gt = count

        # Build query text — pattern: "big <color> <shape>s (excluding <neg_shape>s)"
        is_big = False
        color_adj = None
        head = "shapes"
        excluded = None
        for kind, val in conditions:
            if kind == "big":
                is_big = True
            elif kind == "color":
                color_adj = _COLOR_NAMES[val]
            elif kind == "shape":
                head = val + "s"
            elif kind == "not_shape":
                excluded = val
        parts = []
        if is_big:
            parts.append("big")
        if color_adj:
            parts.append(color_adj)
        parts.append(head)
        query_text = " ".join(parts)
        if excluded:
            query_text += f" (excluding {excluded}s)"

        # Do NOT mention the total count in question text (moved to image
        # title) — otherwise "{n} figures" could help rule out implausible
        # answers. The total is visible in the image title only.
        given_templates = [
            "Look at the figures shown in the image.",
            "The image displays several figures.",
            "A collection of figures is shown above.",
            "Study the arrangement of figures in the image.",
        ]
        given_text = rng.choice(given_templates)
        ask_text = rng.choice(_QUERY_TEMPLATES).format(q=query_text)

        # Distractors
        tight = cfg["tight_distractors"]
        if tight:
            pool = [gt - 2, gt - 1, gt + 1, gt + 2]
        else:
            pool = [gt - 3, gt - 2, gt + 2, gt + 3, gt - 1, gt + 1]
        pool = [p for p in pool if p >= 0 and p != gt]
        rng.shuffle(pool)
        distractors = pool[:3]
        if len(distractors) < 3:
            for k in (-4, 4, 5, -5, 6):
                cand = gt + k
                if cand >= 0 and cand != gt and cand not in distractors:
                    distractors.append(cand)
                if len(distractors) >= 3:
                    break
        if len(distractors) < 3:
            return None

        options_vals = [gt] + distractors[:3]
        rng.shuffle(options_vals)
        if options_vals.count(gt) > 1:
            return None
        answer_letter = chr(ord("A") + options_vals.index(gt))
        options_str = [str(v) for v in options_vals]

        question = (
            f"{given_text} {ask_text}\n"
            + "\n".join(f"  ({chr(ord('A') + i)}) {options_str[i]}" for i in range(4))
            + "\nAnswer with the single letter of the correct option."
        )

        image = self._render(shapes, given_text, ask_text, options_str)
        return question, answer_letter, image

    def _draw(self, ax, cx, cy, sh):
        shape = sh["shape"]
        color = sh["color"]
        size = 0.55 if sh["big"] else 0.33

        if shape == "circle":
            p = mpatches.Circle((cx, cy), size, facecolor=color,
                                 edgecolor="#1a1a1a", linewidth=1.1)
        elif shape == "square":
            p = mpatches.Rectangle((cx - size, cy - size), 2 * size, 2 * size,
                                    facecolor=color, edgecolor="#1a1a1a",
                                    linewidth=1.1)
        elif shape == "triangle":
            p = mpatches.Polygon([(cx, cy + size),
                                   (cx - size, cy - size),
                                   (cx + size, cy - size)],
                                  closed=True, facecolor=color,
                                  edgecolor="#1a1a1a", linewidth=1.1)
        elif shape == "hexagon":
            pts = [(cx + size * math.cos(math.radians(60 * i + 30)),
                    cy + size * math.sin(math.radians(60 * i + 30)))
                   for i in range(6)]
            p = mpatches.Polygon(pts, closed=True, facecolor=color,
                                  edgecolor="#1a1a1a", linewidth=1.1)
        elif shape == "diamond":
            p = mpatches.Polygon([(cx, cy + size), (cx + size, cy),
                                   (cx, cy - size), (cx - size, cy)],
                                  closed=True, facecolor=color,
                                  edgecolor="#1a1a1a", linewidth=1.1)
        elif shape == "star":
            pts = []
            for i in range(10):
                ang = math.radians(90 + i * 36)
                r = size if i % 2 == 0 else size * 0.45
                pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
            p = mpatches.Polygon(pts, closed=True, facecolor=color,
                                  edgecolor="#1a1a1a", linewidth=1.1)
        elif shape == "pentagon":
            pts = [(cx + size * math.cos(math.radians(90 + 72 * i)),
                    cy + size * math.sin(math.radians(90 + 72 * i)))
                   for i in range(5)]
            p = mpatches.Polygon(pts, closed=True, facecolor=color,
                                  edgecolor="#1a1a1a", linewidth=1.1)
        else:
            p = mpatches.Circle((cx, cy), size, facecolor=color,
                                 edgecolor="#1a1a1a", linewidth=1.1)
        ax.add_patch(p)

    def _render(self, shapes, given_text, ask_text, options) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        ff = style["font_family"]
        fs = style["font_size_base"]

        n = len(shapes)
        ncol = 4 if n <= 8 else 5
        nrow = int(math.ceil(n / ncol))

        fig = plt.figure(figsize=(11.0 * sc, 6.0 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax_f = fig.add_subplot(1, 2, 1)
        ax_t = fig.add_subplot(1, 2, 2)
        ax_f.set_aspect("equal")
        ax_f.axis("off")
        ax_t.axis("off")

        cell = 1.8
        for i, sh in enumerate(shapes):
            col = i % ncol
            row = i // ncol
            cx = col * cell + 1
            cy = (nrow - 1 - row) * cell + 1
            self._draw(ax_f, cx, cy, sh)
        ax_f.set_xlim(-0.5, ncol * cell + 0.5)
        ax_f.set_ylim(-0.5, nrow * cell + 0.5)
        sca_title_pool = [f"{n} figures", "Shape Collection",
                          "Figure Display", f"{n} shapes", "Counting Task"]
        ax_f.set_title(self._rng.choice(sca_title_pool),
                       fontsize=fs + 1, family=ff)

        ax_t.set_xlim(0, 10)
        ax_t.set_ylim(0, 12)
        ax_t.text(0.3, 11.5, "Query:", fontsize=fs + 1, fontweight="bold",
                  family=ff, ha="left", va="top", color="#2c3e50")
        y = 10.8
        for ln in self._wrap(given_text + " " + ask_text, 42):
            ax_t.text(0.3, y, ln, fontsize=fs, family=ff,
                      ha="left", va="top", color="#1a1a1a")
            y -= 0.55
        y -= 0.3
        ax_t.text(0.3, y, "Options:", fontsize=fs + 1, fontweight="bold",
                  family=ff, ha="left", va="top", color="#2c3e50")
        y -= 0.55
        for i, o in enumerate(options):
            ax_t.text(0.5, y, f"({chr(ord('A') + i)}) {o}",
                      fontsize=fs, family=ff, ha="left", va="top",
                      color="#1a1a1a")
            y -= 0.55

        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05,
                            wspace=0.15)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    @staticmethod
    def _wrap(text: str, width: int = 40) -> List[str]:
        out, cur = [], ""
        for word in text.split():
            if len(cur) + len(word) + 1 > width:
                out.append(cur)
                cur = word
            else:
                cur = (cur + " " + word).strip()
        if cur:
            out.append(cur)
        return out

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_b2b"
    os.makedirs(out_dir, exist_ok=True)
    env = ShapeCountingAnalogyQA()
    for level in (0, 3, 6):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[seed={s} L{level}] FAILED")
                continue
            path = os.path.join(
                out_dir, f"shape_counting_analogy_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[seed={s} L{level}] saved {path}")
            print(f"  Q (first 100): {env.get_instruction()[:100]}")
            print(f"  A: {env._answer}")
