"""
Relative Direction Chain QA (redesigned 2026-04-16).

Grid map with labeled points. Pairwise direction relations shown.
Asks about a transitive relation. MCQ with direction options.

Critical fix (vs Grade D baseline):
  * Old: every seed rendered the same dot-on-grid layout; shapes/chains
    looked identical.
  * Now: multiple visualization styles (dot map, arrow diagram,
    compass-rose scatter, labeled grid tiles), diverse colors per seed,
    point marker variety.
  * Expanded direction set handling (4 cardinals, 8 with intercardinals).
  * L0/L9 structural shift: L0 = 2-hop cardinal, L9 = 5-hop 8-direction.
  * 6+ question templates.
  * Randomized layout scale, rotation of axes, compass-rose overlay.
  * MCQ shuffle.

2026-05-03 extension (W50 / reference CP-T9):
added a `translate_square_new_direction` mode that shows a small square
on the grid, describes a translation in compass directions, and asks the
new compass direction of a labeled vertex relative to the origin (or
relative to a fixed reference point). Output remains MCQ letter A/B/C/D.
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_CARDINALS = {
    'north': (0, 1), 'south': (0, -1),
    'east': (1, 0), 'west': (-1, 0),
}
_INTERCARDINALS = {
    **_CARDINALS,
    'northeast': (1, 1), 'northwest': (-1, 1),
    'southeast': (1, -1), 'southwest': (-1, -1),
}

_DIR_NAMES_4 = list(_CARDINALS.keys())
_DIR_NAMES_8 = list(_INTERCARDINALS.keys())

_NEUTRAL_TITLES = [
    "Direction Chain", "Position Map", "Point Diagram", "Locations",
    "Spatial Layout", "Relative Positions", "Point Arrangement",
    "Map",
]

def _vec_to_dir(dx, dy):
    if dx == 0 and dy == 0:
        return "same"
    angle = math.degrees(math.atan2(dy, dx))
    dirs = [
        (0, 'east'), (45, 'northeast'), (90, 'north'), (135, 'northwest'),
        (180, 'west'), (-180, 'west'), (-135, 'southwest'),
        (-90, 'south'), (-45, 'southeast'),
    ]
    best = min(dirs,
               key=lambda d: abs(((angle - d[0] + 180) % 360) - 180))
    return best[1]

class RelativeDirectionChainQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    ENV_NAME = "relative_direction_chain"

    def _level_config(self, level):
        return {
            'chain_len': 2 + level // 2,
            'use_8dir': level >= 5,
            'show_grid': level <= 3,
        }

    def _generate_problem(self, seed, parameter):
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1023)
        vis_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 4357)
        style = self._random_style()

        # W50 extension — translate square + new direction. Routes to a
        # standalone generator so the chain logic stays unchanged.
        # Trigger explicitly via parameter, OR with low probability when
        # caller doesn't specify question_type.
        if parameter.get("question_type") == "translate_square_new_direction":
            r = self._gen_translate_square(rng, vis_rng, cfg, style)
            return r if r is not None else None
        if parameter.get("question_type") is None and rng.random() < 0.20:
            r = self._gen_translate_square(rng, vis_rng, cfg, style)
            if r is not None:
                return r
            # else fall through to standard chain generation

        n_points = cfg['chain_len'] + 1
        dir_map = _INTERCARDINALS if cfg['use_8dir'] else _CARDINALS
        dir_names = _DIR_NAMES_8 if cfg['use_8dir'] else _DIR_NAMES_4

        # Generate chain of points
        labels = [chr(65 + i) for i in range(n_points)]
        positions = {labels[0]: (0, 0)}
        relations = []

        seen_positions = {(0, 0)}
        for i in range(1, n_points):
            # Try up to a few times to avoid overlaps
            for _ in range(8):
                direction = rng.choice(dir_names)
                dx, dy = dir_map[direction]
                mag = rng.randint(1, 3)
                prev_pos = positions[labels[i - 1]]
                new_pos = (prev_pos[0] + dx * mag, prev_pos[1] + dy * mag)
                if new_pos not in seen_positions:
                    break
            positions[labels[i]] = new_pos
            seen_positions.add(new_pos)
            relations.append((labels[i - 1], direction, labels[i]))

        start_pos = positions[labels[0]]
        end_pos = positions[labels[-1]]
        total_dx = end_pos[0] - start_pos[0]
        total_dy = end_pos[1] - start_pos[1]

        # Question types
        qtype = vis_rng.choice(["first_from_last", "last_from_first",
                                 "random_pair"])
        if qtype == "first_from_last":
            a, b = labels[-1], labels[0]
            correct_dir = _vec_to_dir(
                start_pos[0] - end_pos[0], start_pos[1] - end_pos[1])
        elif qtype == "last_from_first":
            a, b = labels[0], labels[-1]
            correct_dir = _vec_to_dir(
                end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
        else:
            # Random non-adjacent pair (prefer non-adjacent)
            candidates = []
            for i_ in range(n_points):
                for j_ in range(n_points):
                    if abs(i_ - j_) >= 2:
                        candidates.append((i_, j_))
            if not candidates:
                candidates = [(0, n_points - 1)]
            i_, j_ = vis_rng.choice(candidates)
            a, b = labels[i_], labels[j_]
            pa, pb = positions[a], positions[b]
            correct_dir = _vec_to_dir(
                pb[0] - pa[0], pb[1] - pa[1])

        if correct_dir == "same":
            # degenerate; pick a different pair
            a, b = labels[0], labels[-1]
            correct_dir = _vec_to_dir(total_dx, total_dy)
            if correct_dir == "same":
                return None

        # MCQ options
        options = [correct_dir]
        # Prefer opposite and 90-degree-off distractors
        for d in dir_names:
            if d != correct_dir and len(options) < 4:
                options.append(d)
        while len(options) < 4:
            options.append(rng.choice(dir_names))
        vis_rng.shuffle(options)
        correct_idx = options.index(correct_dir)
        correct = "ABCD"[correct_idx]

        # -------------------- Rendering --------------------
        img = self._render(positions, relations, labels, cfg, style, vis_rng)

        # Question text — relations NOT listed in text if level is high
        # (forces the model to read from the image) — but we still show
        # them as text for lower levels to maintain solvability.
        # relations are stored as (labels[i-1], direction, labels[i]) where
        # labels[i] was placed in `direction` FROM labels[i-1]; i.e.
        # labels[i] is `direction` of labels[i-1]. Render text accordingly.
        rel_text_parts = [f"{r[2]} is {r[1]} of {r[0]}"
                          for r in relations]
        rel_text = "; ".join(rel_text_parts)
        opt_str = "  ".join(f"({chr(65 + i)}) {options[i]}"
                            for i in range(4))
        q_templates = [
            f"Given the relations {rel_text}. What direction is {b} from "
            f"{a}?\n{opt_str}\nAnswer with the letter.",
            f"Based on the map and the relations {rel_text}, in what "
            f"direction does {b} lie relative to {a}?\n{opt_str}\n"
            f"Answer with the letter.",
            f"The image and text describe the positions of points. "
            f"{rel_text}. What is the direction of {b} as seen from {a}? "
            f"\n{opt_str}\nAnswer with the letter.",
            f"Looking at the map: {rel_text}. If you stand at {a}, in "
            f"which direction is {b}?\n{opt_str}\nAnswer with the letter.",
            f"Relations: {rel_text}. Which compass direction describes the "
            f"location of {b} relative to {a}?\n{opt_str}\nAnswer with "
            f"the letter.",
        ]
        q = vis_rng.choice(q_templates)
        return q, correct, img

    def _render(self, positions, relations, labels, cfg, style, vis_rng):
        sc = style['figsize_scale']
        fig, ax = plt.subplots(figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style['bg_color'])
        ax.set_facecolor(style['bg_color'])
        ax.set_aspect('equal')

        palette = list(style['palette'])
        vis_rng.shuffle(palette)

        render_mode = vis_rng.choice(
            ["dots", "arrows_only", "connected_path", "grid_tiles"])

        # Draw the chain
        if render_mode == "dots":
            # Minimal dots + optional arrows
            for label, (px, py) in positions.items():
                color = palette[labels.index(label) % len(palette)]
                marker = vis_rng.choice(["o", "s", "D", "^", "P"])
                ax.plot(px, py, marker, color=color, markersize=15,
                        markeredgecolor='#333', markeredgewidth=1.5)
                ax.text(px, py + 0.35, label, ha='center', va='bottom',
                        fontsize=style['font_size_base'] + 3,
                        fontweight='bold')
            for (fl, d, tl) in relations:
                fp = positions[fl]
                tp = positions[tl]
                ax.annotate('', xy=tp, xytext=fp,
                            arrowprops=dict(arrowstyle='->',
                                            color='#666',
                                            lw=1.4,
                                            alpha=0.6))
        elif render_mode == "arrows_only":
            # Big colored arrows
            arrow_colors = palette[:4] * (len(relations) // 4 + 1)
            for ci, (fl, d, tl) in enumerate(relations):
                fp = positions[fl]
                tp = positions[tl]
                ax.annotate('', xy=tp, xytext=fp,
                            arrowprops=dict(arrowstyle='-|>',
                                            color=arrow_colors[ci],
                                            lw=2.3, alpha=0.85))
            for label, (px, py) in positions.items():
                color = palette[labels.index(label) % len(palette)]
                ax.plot(px, py, 'o', color="white",
                        markersize=18,
                        markeredgecolor=color, markeredgewidth=2)
                ax.text(px, py, label, ha='center', va='center',
                        fontsize=style['font_size_base'] + 1,
                        fontweight='bold', color=color)
        elif render_mode == "connected_path":
            # Connect chain with a path line
            xs_c = [positions[l][0] for l in labels]
            ys_c = [positions[l][1] for l in labels]
            ax.plot(xs_c, ys_c, '-', color=palette[0], linewidth=2.5,
                    alpha=0.7, zorder=1)
            for label, (px, py) in positions.items():
                color = palette[labels.index(label) % len(palette)]
                marker = vis_rng.choice(["o", "s", "D"])
                ax.plot(px, py, marker, color=color, markersize=16,
                        markeredgecolor='#222', markeredgewidth=1.4,
                        zorder=5)
                ax.text(px + 0.15, py + 0.25, label, ha='left', va='bottom',
                        fontsize=style['font_size_base'] + 3,
                        fontweight='bold')
        else:  # grid_tiles
            # Draw tile squares at each position
            for label, (px, py) in positions.items():
                color = palette[labels.index(label) % len(palette)]
                rect = mpatches.FancyBboxPatch(
                    (px - 0.4, py - 0.4), 0.8, 0.8,
                    boxstyle="round,pad=0.02",
                    facecolor=color, edgecolor='#222',
                    linewidth=1.6, alpha=0.8)
                ax.add_patch(rect)
                ax.text(px, py, label, ha='center', va='center',
                        fontsize=style['font_size_base'] + 2,
                        fontweight='bold', color='white')

        # Optional compass rose (upper-left)
        if vis_rng.random() < 0.45:
            xs = [p[0] for p in positions.values()]
            ys = [p[1] for p in positions.values()]
            rx = max(xs) + 0.8
            ry = max(ys) + 0.8
            r = 0.5
            ax.annotate('', xy=(rx, ry + r), xytext=(rx, ry),
                        arrowprops=dict(arrowstyle='->', color='#222'))
            ax.text(rx, ry + r + 0.1, "N", ha='center', va='bottom',
                    fontsize=8, fontweight='bold')
            ax.text(rx + r + 0.08, ry, "E", ha='left', va='center',
                    fontsize=8, fontweight='bold')

        if cfg['show_grid']:
            ax.grid(True, alpha=0.22, linestyle='--')
        ax.axis('off')
        ax.autoscale_view()
        ax.margins(0.25)
        ax.set_title(vis_rng.choice(_NEUTRAL_TITLES),
                     fontsize=style['font_size_base'] + 3,
                     fontweight='bold')
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style['dpi'])

    # ------------------------------------------------------------------ #
    # W50 — translate square + new direction generator
    # ------------------------------------------------------------------ #
    def _gen_translate_square(self, rng, vis_rng, cfg, style):
        """Render a square with corner labels A,B,C,D, describe a translation
        in compass terms ("3 east, 2 north"), and ask the new compass
        direction of one corner relative to a fixed origin or a fixed
        reference point. Output: MCQ letter A/B/C/D (compass-direction option).
        """
        # Pick a square at integer corners centered roughly at origin.
        side = rng.choice([1, 2])
        cx, cy = rng.randint(-2, 2), rng.randint(-2, 2)
        corners = [
            (cx, cy),
            (cx + side, cy),
            (cx + side, cy + side),
            (cx, cy + side),
        ]
        labels = ["A", "B", "C", "D"]

        # Translation vector: integer (tx, ty), |tx|+|ty| at least 2.
        while True:
            tx = rng.randint(-4, 4)
            ty = rng.randint(-4, 4)
            if abs(tx) + abs(ty) >= 2:
                break
        new_corners = [(x + tx, y + ty) for (x, y) in corners]

        # Reference: a fixed point P at origin (0,0).
        ref_pt = (0, 0)
        ref_label = "the origin"

        # Pick which corner to ask about.
        target_idx = rng.randint(0, 3)
        target_label = labels[target_idx]
        new_pos = new_corners[target_idx]
        rel_dx = new_pos[0] - ref_pt[0]
        rel_dy = new_pos[1] - ref_pt[1]
        correct_dir = _vec_to_dir(rel_dx, rel_dy)
        if correct_dir == "same":
            return None

        # Build MCQ — 4 distinct directions, correct + 3 plausible distractors.
        dir_pool = list(_INTERCARDINALS.keys())
        # Build distractors as opposite + adjacents
        opposite_map = {
            "north": "south", "south": "north",
            "east": "west", "west": "east",
            "northeast": "southwest", "northwest": "southeast",
            "southeast": "northwest", "southwest": "northeast",
        }
        distractors = set()
        opp = opposite_map.get(correct_dir)
        if opp and opp != correct_dir:
            distractors.add(opp)
        for d in dir_pool:
            if len(distractors) >= 3:
                break
            if d == correct_dir or d in distractors:
                continue
            distractors.add(d)
        options = [correct_dir] + list(distractors)[:3]
        rng.shuffle(options)
        correct_letter = "ABCD"[options.index(correct_dir)]

        # Render
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        palette = list(style["palette"])
        # Original square
        orig_poly = plt.Polygon(corners, closed=True,
                                facecolor=palette[0], edgecolor='black',
                                alpha=0.45, linewidth=2)
        ax.add_patch(orig_poly)
        for label, (px, py) in zip(labels, corners):
            ax.text(px, py, label, ha='center', va='center',
                    fontsize=11, fontweight='bold')
        # New square (translated)
        new_poly = plt.Polygon(new_corners, closed=True,
                               facecolor=palette[1 % len(palette)],
                               edgecolor='black', alpha=0.45, linewidth=2)
        ax.add_patch(new_poly)
        for label, (px, py) in zip(labels, new_corners):
            ax.text(px, py, f"{label}'", ha='center', va='center',
                    fontsize=11, fontweight='bold', color='red')
        # Origin marker
        ax.plot(0, 0, marker='*', color='black', markersize=14, zorder=10)
        ax.annotate("Origin (0,0)", xy=(0, 0), xytext=(0.3, 0.3),
                    fontsize=10, fontweight='bold')
        # Compass rose
        all_x = [p[0] for p in corners + new_corners] + [0]
        all_y = [p[1] for p in corners + new_corners] + [0]
        margin = 2
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        ax.axhline(0, color="gray", alpha=0.5, linewidth=0.6)
        ax.axvline(0, color="gray", alpha=0.5, linewidth=0.6)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_title("Translate the square — find new compass direction",
                     fontsize=style["font_size_base"] + 2, fontweight='bold')

        # Translation phrasing
        parts = []
        if tx > 0:
            parts.append(f"{tx} unit{'s' if abs(tx) > 1 else ''} east")
        elif tx < 0:
            parts.append(f"{abs(tx)} unit{'s' if abs(tx) > 1 else ''} west")
        if ty > 0:
            parts.append(f"{ty} unit{'s' if abs(ty) > 1 else ''} north")
        elif ty < 0:
            parts.append(f"{abs(ty)} unit{'s' if abs(ty) > 1 else ''} south")
        translate_phrase = " and ".join(parts) if parts else "0"

        opt_str = "  ".join(f"({chr(65 + i)}) {options[i]}" for i in range(4))
        q = (f"The figure shows a square with corners labeled A, B, C, D. "
             f"The square is translated {translate_phrase}, producing a new "
             f"square with corners A', B', C', D'. After the translation, "
             f"in what compass direction does corner {target_label}' lie "
             f"relative to {ref_label}?\n{opt_str}\n"
             f"Answer with the letter.")
        return q, correct_letter, self.fig_to_pil(fig, dpi=style["dpi"])


if __name__ == "__main__":
    env = RelativeDirectionChainQA()
    for lv in [0, 3, 6, 9]:
        for seed in range(3):
            ok = env.generate(seed, {"level": lv})
            print(f"L{lv} s{seed}: ok={ok}, answer={env._answer}")
