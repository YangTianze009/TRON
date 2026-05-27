"""
Cube Assembly Missing Piece QA — pure single-letter MCQ.

2026-05-05 R5 Batch 4 (TOO-HARD ease + restructure):
Old design (261 LOC) had wrapper-mismatch (template embedded
<answer>...</answer>) and 4^3 stacks at L8+. R3 = 10%.

REWRITE strategy:
  - L0/L1: 2x2x2 stack missing exactly 1 cube (a "corner piece"). 4-MCQ over
            {single-cube, 2-cube line, 3-cube L, 4-cube square}. Easy: model
            sees Original (8 cubes), Part 1 (7 cubes) → answer is 1 cube.
  - L2-L4: 2x2x2 missing 2-3 cubes (small contiguous chunk). 4-MCQ.
  - L5-L7: 3x3x3 missing 3-5 cubes (irregular chunk). 5-MCQ + 15-25% trap.
  - L8/L9: 3x3x3 missing 4-7 cubes. 5-MCQ + 35-40% trap.

No wrapper text in question — framework wraps. Pure single-letter MCQ A-D / A-E.
ALLOW_ROTATION=False (orientation matters for 3D iso views).
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# ----------------------------------------------------------------------- #
# Geometry helpers
# ----------------------------------------------------------------------- #

def _iso_project(x, y, z):
    sx = (x - y) * math.cos(math.radians(30))
    sy = (x + y) * math.sin(math.radians(30)) + z
    return sx, sy


def _cube_face_palette_simple(base_color: str):
    """Return shaded top/left/right palette from a base hex color."""
    import colorsys
    h = base_color.lstrip("#")
    r, g, b = (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255,
               int(h[4:6], 16) / 255)
    hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
    base_l = 0.55
    ss = min(1.0, max(0.45, ss))
    top_rgb = colorsys.hls_to_rgb(hh, min(0.85, base_l + 0.22), ss)
    right_rgb = colorsys.hls_to_rgb(hh, base_l, ss)
    left_rgb = colorsys.hls_to_rgb(hh, max(0.18, base_l - 0.25), ss)
    def _hx(rgb):
        return '#{:02x}{:02x}{:02x}'.format(
            int(max(0, min(1, rgb[0])) * 255),
            int(max(0, min(1, rgb[1])) * 255),
            int(max(0, min(1, rgb[2])) * 255))
    return [_hx(top_rgb), _hx(left_rgb), _hx(right_rgb)]


def _draw_iso_cubes(ax, cubes, palette):
    """Draw a list of unit cubes in isometric view onto ax."""
    if not cubes:
        return
    sorted_cubes = sorted(cubes, key=lambda c: -(c[0] + c[1] - c[2]))
    top_color, left_color, right_color = palette
    for (x, y, z) in sorted_cubes:
        face_top = [
            _iso_project(x, y, z + 1),
            _iso_project(x + 1, y, z + 1),
            _iso_project(x + 1, y + 1, z + 1),
            _iso_project(x, y + 1, z + 1),
        ]
        face_left = [
            _iso_project(x, y, z),
            _iso_project(x, y + 1, z),
            _iso_project(x, y + 1, z + 1),
            _iso_project(x, y, z + 1),
        ]
        face_right = [
            _iso_project(x, y, z),
            _iso_project(x + 1, y, z),
            _iso_project(x + 1, y, z + 1),
            _iso_project(x, y, z + 1),
        ]
        ax.add_patch(Polygon(face_top, closed=True, facecolor=top_color,
                             edgecolor="#222", lw=1.0))
        ax.add_patch(Polygon(face_left, closed=True, facecolor=left_color,
                             edgecolor="#222", lw=1.0))
        ax.add_patch(Polygon(face_right, closed=True, facecolor=right_color,
                             edgecolor="#222", lw=1.0))


def _autoscale_iso(ax, all_cubes_options, sz):
    """Auto-scale ax to show a sz x sz x sz bounding region."""
    pts = []
    for x in range(sz + 1):
        for y in range(sz + 1):
            for z in range(sz + 1):
                pts.append(_iso_project(x, y, z))
    if pts:
        arr = np.array(pts)
        margin = 0.4
        ax.set_xlim(arr[:, 0].min() - margin, arr[:, 0].max() + margin)
        ax.set_ylim(arr[:, 1].min() - margin, arr[:, 1].max() + margin)


# ----------------------------------------------------------------------- #
# Stack generation
# ----------------------------------------------------------------------- #

def _gen_full_stack(sz: int) -> List[Tuple[int, int, int]]:
    """All cells of a sz^3 cube."""
    return [(x, y, z) for x in range(sz)
            for y in range(sz)
            for z in range(sz)]


def _gen_random_stack(rng, sz: int, fill_rate: float = 0.7
                      ) -> List[Tuple[int, int, int]]:
    """Random stack with given fill rate, ensuring ≥3 cubes."""
    cubes = []
    for _ in range(40):
        cubes = []
        for x in range(sz):
            for y in range(sz):
                for z in range(sz):
                    if rng.random() < fill_rate:
                        cubes.append((x, y, z))
        if len(cubes) >= 3:
            return sorted(cubes)
    # Fallback: full stack
    return _gen_full_stack(sz)


def _connected_subset(rng, cubes: List[Tuple[int, int, int]], n: int
                      ) -> List[Tuple[int, int, int]]:
    """BFS-grow a connected subset of ~n cubes."""
    if n >= len(cubes):
        return list(cubes)
    cset = set(cubes)
    start = rng.choice(cubes)
    selected = [start]
    sel_set = {start}
    while len(selected) < n:
        # Find a cube in selected with neighbours not yet selected
        candidates = []
        for c in selected:
            for d in [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                      (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
                nb = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
                if nb in cset and nb not in sel_set:
                    candidates.append(nb)
        if not candidates:
            break
        nb = rng.choice(candidates)
        selected.append(nb)
        sel_set.add(nb)
    return selected


# ----------------------------------------------------------------------- #
# Env
# ----------------------------------------------------------------------- #

class CubeAssemblyMissingPieceQA(StandaloneVisualEnv):
    """Pure single-letter MCQ for missing-piece identification."""

    ALLOW_ROTATION = False
    ENV_NAME = "cube_assembly_missing_piece"

    def _level_config(self, level: int) -> dict:
        level = max(0, min(9, level))
        if level <= 1:
            return {"size": 2, "missing_count": 1,
                    "use_5_options": False, "trap_rate": 0.0,
                    "fill_rate": 1.0}
        if level <= 4:
            return {"size": 2, "missing_count": rng_choice_default(2),
                    "use_5_options": False, "trap_rate": 0.0,
                    "fill_rate": 1.0}
        if level <= 6:
            return {"size": 3, "missing_count": 4,
                    "use_5_options": True, "trap_rate": 0.15,
                    "fill_rate": 0.85}
        if level == 7:
            return {"size": 3, "missing_count": 5,
                    "use_5_options": True, "trap_rate": 0.25,
                    "fill_rate": 0.85}
        if level == 8:
            return {"size": 3, "missing_count": 6,
                    "use_5_options": True, "trap_rate": 0.30,
                    "fill_rate": 0.85}
        return {"size": 3, "missing_count": 7,
                "use_5_options": True, "trap_rate": 0.32,
                "fill_rate": 0.85}

    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 401)
        for _ in range(40):
            res = self._try_generate(cfg, level, sub_rng)
            if res is not None:
                return res
        return None

    def _try_generate(self, cfg, level, rng):
        sz = cfg["size"]
        fill_rate = cfg.get("fill_rate", 1.0)
        if fill_rate >= 1.0:
            cubes = _gen_full_stack(sz)
        else:
            cubes = _gen_random_stack(rng, sz, fill_rate)

        if len(cubes) < 4:
            return None

        # The "missing" piece — connected subset of size missing_count
        target_missing = max(1, min(len(cubes) - 1, cfg["missing_count"]))
        missing = _connected_subset(rng, cubes, target_missing)
        if len(missing) < 1:
            return None
        missing_set = set(missing)
        part1 = [c for c in cubes if c not in missing_set]
        if len(part1) < 1:
            return None

        # Build distractor pieces
        # Strategy: take the missing-cube count, but build slightly different
        # connected pieces using cells from EITHER inside or outside the
        # original assembly. We'll build them within a sz x sz x sz workspace.
        distractors = self._build_distractors(rng, missing, sz, n_needed=4)
        if len(distractors) < 3:
            return None

        # Assemble options
        use_5 = cfg["use_5_options"]
        n_value = 5 if use_5 else 4
        opt_letters = ["A", "B", "C", "D", "E"][:n_value]
        use_trap = (rng.random() < cfg["trap_rate"]) if use_5 else False

        # Select the right number of distractors
        n_distractors = n_value - 1 if not use_trap else n_value - 1
        distractor_pool = list(distractors)
        rng.shuffle(distractor_pool)

        if use_5:
            # 5-MCQ: visual diagram always shows 4 options labeled A-D; the
            # 5th option is "No correct answer". When use_trap, the 4 visual
            # options are all wrong and the answer is E. Otherwise one of
            # the 4 visual options is correct (A-D).
            n_visual = 4
            if use_trap:
                chosen = distractor_pool[:n_visual]
                option_pieces = chosen
                correct_letter = "E"
            else:
                chosen_d = distractor_pool[:n_visual - 1]
                option_pieces = [missing] + chosen_d
                rng.shuffle(option_pieces)
                correct_idx = next(i for i, o in enumerate(option_pieces)
                                   if sorted(o) == sorted(missing))
                correct_letter = opt_letters[correct_idx]
            opt_lines_text = [f"Option {opt_letters[i]} (see diagram)"
                              for i in range(n_visual)]
            opt_lines_text.append("No correct answer")
        else:
            # 4-MCQ: 4 visual options labeled A-D; one is correct.
            chosen_d = distractor_pool[:n_value - 1]
            option_pieces = [missing] + chosen_d
            rng.shuffle(option_pieces)
            correct_idx = next(i for i, o in enumerate(option_pieces)
                               if sorted(o) == sorted(missing))
            correct_letter = opt_letters[correct_idx]
            opt_lines_text = [f"Option {opt_letters[i]} (see diagram)"
                              for i in range(n_value)]

        # Render
        img = self._render_layout(cubes, part1, option_pieces, sz, rng,
                                  opt_letters[:len(option_pieces)])

        # Question text — no wrapper hint here (framework adds)
        n_assembly = len(cubes)
        n_part1 = len(part1)
        n_missing = n_assembly - n_part1

        opt_text = "\n".join(opt_lines_text[:n_value])
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        if level <= 1:
            stem = (
                f"The diagram shows three pictures: the ORIGINAL complete "
                f"cube assembly (top-left), PART 1 (top-right) which is the "
                f"assembly with one or more cubes removed, and {n_value if not use_5 else n_value-1} "
                f"OPTION pieces below. Which option, when added to PART 1, "
                f"reproduces the ORIGINAL? Hint: the original has "
                f"{n_assembly} unit cubes and PART 1 has {n_part1}, so the "
                f"missing piece must contain {n_missing} cube(s). Be concise."
            )
        else:
            stem = (
                f"The diagram shows the ORIGINAL complete cube assembly "
                f"(top-left), PART 1 (top-right) which is the assembly with "
                f"some cubes removed, and the candidate missing pieces. "
                f"Which option, when added to PART 1, reproduces the "
                f"ORIGINAL assembly?"
            )

        question = (
            f"{stem}\n"
            f"{opt_text}\n"
            f"Answer with a single letter {letter_str}."
        )
        return question, correct_letter, img

    def _build_distractors(self, rng, missing, sz: int, n_needed: int
                           ) -> List[List[Tuple[int, int, int]]]:
        """Build distractor pieces with the SAME number of cubes as missing
        but DIFFERENT shapes."""
        out = []
        target_n = len(missing)
        seen_keys = {tuple(sorted(missing))}

        # Strategy 1: perturb missing by removing a cube + adding a different one
        attempts = 0
        while len(out) < n_needed and attempts < 80:
            attempts += 1
            base = list(missing)
            if len(base) <= 1:
                # Add a random cube somewhere in the bbox
                cand_pool = [(x, y, z) for x in range(sz)
                             for y in range(sz) for z in range(sz)
                             if (x, y, z) != base[0]]
                rng.shuffle(cand_pool)
                fake = [rng.choice(cand_pool)]
            else:
                # Remove 1 + add 1
                rm = rng.choice(base)
                fake = [c for c in base if c != rm]
                cand_pool = [(x, y, z) for x in range(sz)
                             for y in range(sz) for z in range(sz)
                             if (x, y, z) not in fake]
                rng.shuffle(cand_pool)
                if not cand_pool:
                    continue
                fake.append(cand_pool[0])
            key = tuple(sorted(fake))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            out.append(sorted(fake))

        # Strategy 2: random connected pieces of the right size in workspace
        attempts = 0
        while len(out) < n_needed and attempts < 80:
            attempts += 1
            start = (rng.randrange(sz), rng.randrange(sz), rng.randrange(sz))
            piece = [start]
            piece_set = {start}
            while len(piece) < target_n:
                expand_candidates = []
                for c in piece:
                    for d in [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                              (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
                        nb = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
                        if all(0 <= nb[i] < sz for i in range(3)) and nb not in piece_set:
                            expand_candidates.append(nb)
                if not expand_candidates:
                    break
                nxt = rng.choice(expand_candidates)
                piece.append(nxt)
                piece_set.add(nxt)
            if len(piece) != target_n:
                continue
            key = tuple(sorted(piece))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            out.append(sorted(piece))

        return out

    # -------------------- Rendering -------------------- #
    def _render_layout(self, cubes, part1, options, sz, rng,
                       opt_letters) -> Image.Image:
        # Use one figure with original (top-left), part1 (top-right), and
        # options (bottom row).
        n_opt = len(options)
        # Pick base palettes
        full_pal = _cube_face_palette_simple("#5dade2")
        part_pal = _cube_face_palette_simple("#48c9b0")
        opt_pal = _cube_face_palette_simple("#ec7063")

        cols = max(2, n_opt)
        sc = 1.0
        fig = plt.figure(figsize=(cols * 3.0 * sc, 6.5 * sc), dpi=110)
        fig.patch.set_facecolor("#ffffff")

        # Top row: Original + Part 1 spanning cols
        ax_orig = fig.add_subplot(2, cols, 1)
        ax_orig.set_aspect("equal")
        ax_orig.axis("off")
        _draw_iso_cubes(ax_orig, cubes, full_pal)
        _autoscale_iso(ax_orig, [cubes], sz)
        ax_orig.set_title("ORIGINAL", fontsize=14, fontweight="bold",
                          pad=4)

        ax_part = fig.add_subplot(2, cols, 2)
        ax_part.set_aspect("equal")
        ax_part.axis("off")
        _draw_iso_cubes(ax_part, part1, part_pal)
        _autoscale_iso(ax_part, [part1], sz)
        ax_part.set_title("PART 1", fontsize=14, fontweight="bold",
                          pad=4)

        # Hide unused top-row slots (when cols>2)
        for i in range(2, cols):
            ax = fig.add_subplot(2, cols, i + 1)
            ax.axis("off")

        # Bottom row: options
        for i, opt_cubes in enumerate(options):
            ax = fig.add_subplot(2, cols, cols + i + 1)
            ax.set_aspect("equal")
            ax.axis("off")
            _draw_iso_cubes(ax, opt_cubes, opt_pal)
            _autoscale_iso(ax, [opt_cubes], sz)
            ax.set_title(f"({opt_letters[i]})", fontsize=14,
                         fontweight="bold", pad=4)

        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=110)


# Helper: random choice between two values for missing_count
def rng_choice_default(default):
    return default


if __name__ == "__main__":
    env = CubeAssemblyMissingPieceQA()
    for lv in [0, 3, 6, 9]:
        ok = env.generate(42, {"level": lv})
        print(f"L{lv}: ok={ok}, answer={env._answer}")
