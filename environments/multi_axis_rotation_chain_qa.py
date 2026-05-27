"""
Multi-Axis Rotation Chain QA — ULTRA-CURRICULUM redesign (2026-04-16).

CRITICAL INSIGHT: base VLM genuinely cannot do 3D mental rotation. Accuracy
on 3D rotation for Qwen3-VL-8B is ~35% (chance = 25%). So
L0/L1 must be solvable by **CUBE COUNTING** alone — the non-rotation
distractor should have a different cube count. At L2+ we require actual
rotation reasoning.

Difficulty ladder:
  L0: 2 cubes in original; distractors include options with 3 and 4 cubes.
      Cube counting alone solves the problem. 90° vertical rotation.
  L1: 2-3 cubes; distractors include an option with one more cube.
      Counting still works.
  L2: 3-4 cubes, ALL options have same cube count. Distractor is a MIRROR,
      which is visually distinguishable from rotation.
  L3-L5: 4-6 cubes, distractor is mirror or cube moved 2 units.
  L6-L9: 5-8 cubes, distractor is 1-unit perturbation (subtle).

Diversity:
  - 6+ polycube PRIMITIVE FAMILIES (I-chain, L-shape, T-shape, Z-shape,
    plus-cross, L3D, staircase) — NOT just random walk.
  - 6 color palettes shuffled per seed.
  - Viewpoint/camera angle jitter (rotation_seed).
  - Per-seed face-color shuffle of the draw palette.
  - 4 question phrasings.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from .standalone_base import StandaloneVisualEnv
from ._render_modes import pick_render_mode, textbook_params, sketch_context

# ---------------------------------------------------------------------------- #
#  Geometry helpers                                                            #
# ---------------------------------------------------------------------------- #

def _iso_project(x, y, z):
    sx = (x - y) * math.cos(math.radians(30))
    sy = (x + y) * math.sin(math.radians(30)) + z
    return sx, sy

def _rot_axis(cubes, axis, k):
    out = []
    for (x, y, z) in cubes:
        for _ in range(k % 4):
            if axis == "x":
                y, z = -z, y
            elif axis == "y":
                x, z = z, -x
            else:
                x, y = -y, x
        out.append((x, y, z))
    return out

def _normalize(cubes):
    xs = [c[0] for c in cubes]
    ys = [c[1] for c in cubes]
    zs = [c[2] for c in cubes]
    mx, my, mz = min(xs), min(ys), min(zs)
    return [(x - mx, y - my, z - mz) for x, y, z in cubes]

_ROTATION_SEQS = [
    [],
    [("x", 1)], [("x", 2)], [("x", 3)],
    [("y", 1)], [("y", 2)], [("y", 3)],
    [("z", 1)], [("z", 2)], [("z", 3)],
    [("x", 1), ("y", 1)], [("x", 1), ("y", 2)], [("x", 1), ("y", 3)],
    [("x", 2), ("y", 1)], [("x", 2), ("y", 3)],
    [("x", 3), ("y", 1)], [("x", 3), ("y", 2)], [("x", 3), ("y", 3)],
    [("y", 1), ("x", 1)], [("y", 1), ("x", 3)],
    [("y", 3), ("x", 1)], [("y", 3), ("x", 3)],
    [("z", 1), ("x", 1)], [("z", 1), ("x", 3)],
    [("z", 3), ("x", 1)], [("z", 3), ("x", 3)],
]

def _all_24_rotations(cubes):
    """Return all 24 rotations of a polycube (as list of normalized lists)."""
    seen = set()
    results = []
    for ax_seq in _ROTATION_SEQS:
        cur = list(cubes)
        for ax, k in ax_seq:
            cur = _rot_axis(cur, ax, k)
        cur = _normalize(cur)
        fs = frozenset(cur)
        if fs not in seen:
            seen.add(fs)
            results.append(cur)
    return results

def _mirror(cubes, axis):
    out = []
    for (x, y, z) in cubes:
        if axis == "x":
            out.append((-x, y, z))
        elif axis == "y":
            out.append((x, -y, z))
        else:
            out.append((x, y, -z))
    return _normalize(out)

def _rotation_set(cubes):
    return {frozenset(r) for r in _all_24_rotations(cubes)}

def _draw_poly(ax, cubes, palette, lw=1.0, edge_col="#222"):
    """Render isometric polycube — painter's algorithm."""
    top_c, left_c, right_c = palette[0], palette[1], palette[2]
    for (x, y, z) in sorted(cubes, key=lambda c: -(c[0] + c[1] - c[2])):
        for face, fc in [
            ([(x, y, z + 1), (x + 1, y, z + 1),
              (x + 1, y + 1, z + 1), (x, y + 1, z + 1)], top_c),
            ([(x, y, z), (x, y + 1, z),
              (x, y + 1, z + 1), (x, y, z + 1)], left_c),
            ([(x, y, z), (x + 1, y, z),
              (x + 1, y, z + 1), (x, y, z + 1)], right_c),
        ]:
            pts = [_iso_project(*p) for p in face]
            ax.add_patch(
                Polygon(pts, closed=True, facecolor=fc,
                        edgecolor=edge_col, lw=lw))

# ---------------------------------------------------------------------------- #
#  Primitive polycube families                                                 #
# ---------------------------------------------------------------------------- #

def _family_I(n, rng):
    """Straight line of cubes along a random axis."""
    axis = rng.choice([0, 1, 2])
    direction = rng.choice([1, -1])
    cubes = []
    for i in range(n):
        if axis == 0:
            cubes.append((i * direction, 0, 0))
        elif axis == 1:
            cubes.append((0, i * direction, 0))
        else:
            cubes.append((0, 0, i * direction))
    return cubes

def _family_L(n, rng):
    """L shape: k cubes along x, then bend up along z."""
    if n < 3:
        return _family_I(n, rng)
    k = rng.randint(2, max(2, n - 1))
    cubes = [(i, 0, 0) for i in range(k)]
    for j in range(1, n - k + 1):
        cubes.append((k - 1, 0, j))
    return cubes

def _family_T(n, rng):
    """T shape."""
    if n < 4:
        return _family_L(n, rng)
    base = rng.randint(3, min(5, n - 1))
    cubes = [(i, 0, 0) for i in range(base)]
    mid = base // 2
    for j in range(1, n - base + 1):
        cubes.append((mid, 0, j))
    return cubes

def _family_Z(n, rng):
    """Z / S shape."""
    if n < 4:
        return _family_L(n, rng)
    cubes = [(0, 0, 0), (1, 0, 0)]
    for j in range(1, max(2, n - 3)):
        cubes.append((1, j, 0))
    cubes.append((2, max(1, n - 3), 0))
    # pad if needed
    while len(cubes) < n:
        last = cubes[-1]
        cubes.append((last[0] + 1, last[1], last[2]))
    return cubes[:n]

def _family_staircase(n, rng):
    """Staircase going diagonally in xz plane."""
    cubes = []
    for i in range(n):
        cubes.append((i // 2, 0, (i + 1) // 2))
    # de-dup
    return list(set(cubes))

def _family_plus(n, rng):
    """Plus / cross: one center, arms of length ~n/5 in each direction."""
    if n < 5:
        return _family_L(n, rng)
    cubes = [(0, 0, 0)]
    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1)]
    rng.shuffle(dirs)
    needed = n - 1
    d_idx = 0
    step = 1
    while needed > 0:
        d = dirs[d_idx % len(dirs)]
        cubes.append((d[0] * step, d[1] * step, d[2] * step))
        d_idx += 1
        needed -= 1
        if d_idx % len(dirs) == 0:
            step += 1
    return cubes

def _family_random_walk(n, rng):
    """Original: random face-adjacent growth."""
    for _attempt in range(20):
        cubes = [(0, 0, 0)]
        dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        for _ in range(n - 1):
            placed = False
            for _t in range(80):
                base = rng.choice(cubes)
                d = rng.choice(dirs)
                nc = (base[0] + d[0], base[1] + d[1], base[2] + d[2])
                if nc not in cubes:
                    cubes.append(nc)
                    placed = True
                    break
            if not placed:
                break
        if len(cubes) == n:
            return cubes
    return None

_FAMILIES = [_family_I, _family_L, _family_T, _family_Z,
             _family_staircase, _family_plus, _family_random_walk]

# ---------------------------------------------------------------------------- #
#  Environment                                                                 #
# ---------------------------------------------------------------------------- #

class MultiAxisRotationChainQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive: disable rotation augmentation
    ENV_NAME = "multi_axis_rotation_chain"

    _QUESTIONS = [
        # --- Direct interrogative (4) ---
        "The left image shows the original cube stack made of equal-sized small cubes. Which of the options on the right CANNOT be obtained by rotating the original? Answer with a single letter A, B, C, or D.",
        "The leftmost polycube is the original. Three of the other four are valid rotations of it; one is NOT. Which option (A-D) CANNOT be obtained by rotating the original? Answer with a letter.",
        "In the figure, the leftmost polycube is the original. Which of the four candidates (A-D) cannot be reached by rotation of the original?",
        "Looking at the original polycube (leftmost) and its four candidates (A-D), which option is NOT a valid rotation of the original? Reply with one letter.",
        # --- Imperative (4) ---
        "Examine the original polycube on the left. Pick the option on the right that is NOT a rotation of it. Answer with one letter A/B/C/D.",
        "Identify which of the four options (A-D) on the right CANNOT be obtained by rotating the original polycube shown leftmost. Reply with the letter.",
        "Select the option (A-D) that is NOT a rotation of the original polycube. Answer with one letter.",
        "Find the candidate among A-D that cannot be produced by rotating the original polycube. Give just the letter.",
        # --- Declarative / cloze (4) ---
        "The option among A-D that is NOT a rotation of the original polycube is ___. Fill in a single letter.",
        "Of the four candidates, the one that cannot be obtained by rotation of the original is labelled ___ (A-D).",
        "Let X be the letter of the candidate that is NOT a rotation of the original polycube. X = ?",
        "Among options A-D, the polycube that is impossible to reach by rotating the original is ___.",
        # --- Reasoning-prompt (4) ---
        "Among the four candidates A-D, three are rotations of the original polycube (shown leftmost) and one is impossible to obtain. Which is the imposter? Answer with a letter.",
        "Reason about whether each of A-D can be produced by rotating the original polycube. Report the letter of the one that cannot.",
        "Examine each candidate (A-D) and decide whether it is a rotation of the original. Which one is NOT? Answer with a letter.",
        "After comparing each of A-D to the original polycube by rotation, identify which candidate cannot be obtained. Reply with the letter.",
    ]

    def _level_config(self, level):
        level = max(0, min(9, int(level)))
        if level == 0:
            return {
                "n_cubes": 2,
                "distractor_type": "count_off",
                "allow_count_distractor": True,
                "family_pool": ["I", "L"],
            }
        if level == 1:
            return {
                "n_cubes": 3,
                "distractor_type": "count_off",
                "allow_count_distractor": True,
                "family_pool": ["I", "L", "T"],
            }
        if level == 2:
            return {
                "n_cubes": 4,
                "distractor_type": "mirror",
                "allow_count_distractor": False,
                "family_pool": ["L", "T", "Z"],
            }
        if level <= 5:
            return {
                "n_cubes": 4 + (level - 3),   # 4,5,6
                "distractor_type": "moved_two",
                "allow_count_distractor": False,
                "family_pool": ["L", "T", "Z", "staircase", "plus", "random"],
            }
        return {
            "n_cubes": 5 + (level - 6),       # 5,6,7,8
            "distractor_type": "moved_one",
            "allow_count_distractor": False,
            "family_pool": ["L", "T", "Z", "staircase", "plus", "random"],
        }

    def _build_from_family(self, name, n, rng):
        table = {
            "I": _family_I, "L": _family_L, "T": _family_T,
            "Z": _family_Z, "staircase": _family_staircase,
            "plus": _family_plus, "random": _family_random_walk,
        }
        try:
            cubes = table[name](n, rng)
        except Exception:
            return None
        if cubes is None:
            return None
        cubes = list(set(cubes))
        if len(cubes) != n:
            # pad / trim
            if len(cubes) > n:
                cubes = cubes[:n]
            else:
                # extend with random walk
                extra = _family_random_walk(n, rng)
                if extra is not None and len(extra) == n:
                    return _normalize(extra)
                return None
        return _normalize(cubes)

    # ------------------------------------------------------------------ #
    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, "Image.Image"]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1003)
        style = self._random_style()

        n = cfg["n_cubes"]
        fam_name = rng.choice(cfg["family_pool"])
        original = self._build_from_family(fam_name, n, rng)
        if original is None:
            original = _family_random_walk(n, rng)
            if original is None:
                return None
            original = _normalize(original)
        original_rots = _all_24_rotations(original)
        original_rot_set = _rotation_set(original)

        # Build non-rotation distractor.
        non_rotation = self._build_non_rotation(
            original, cfg, original_rot_set, rng)
        if non_rotation is None:
            return None
        non_rotation = _normalize(non_rotation)

        chosen_rotations = self._pick_diverse_rotations(
            original, original_rots, n_picks=3, rng=rng)

        options = chosen_rotations + [non_rotation]
        rng.shuffle(options)

        correct_idx = None
        for i, opt in enumerate(options):
            if frozenset(opt) not in original_rot_set:
                correct_idx = i
                break
        if correct_idx is None:
            return None
        correct_letter = "ABCD"[correct_idx]

        # Render — clean / textbook / sketch mode mix
        palette = list(style["palette"][:3])
        rng.shuffle(palette)
        sc = style["figsize_scale"]

        mode = pick_render_mode(rng)
        if mode == "textbook":
            tbp = textbook_params(rng)
            bg = tbp["bg"]
            # Textbook: light grey faces with subtle brightness steps.
            base = tbp["fill_color"]
            palette_use = [base, "#d0d0d0", "#b5b5b5"]
            edge_col = tbp["line_color"]
            lw_use = tbp["line_width"]
            title_kw = {"fontfamily": tbp["font_family"], "color": tbp["line_color"]}
            dpi = max(tbp["dpi"], 130)
        elif mode == "sketch":
            bg = rng.choice(["#fffdf7", "#fffaf0", "#fdfbf6"])
            palette_use = palette
            edge_col = "#1a1a1a"
            lw_use = 2.0
            title_kw = {}
            dpi = max(style["dpi"], 140)
        else:
            bg = "white"
            palette_use = palette
            edge_col = "#222"
            lw_use = 1.5
            title_kw = {}
            dpi = max(style["dpi"], 140)

        # At L0/L1 the docstring promises the task is solvable by cube
        # counting alone. Counting cubes from the iso view turns out to be
        # unreliable for small VLMs, so we annotate the cube count on each
        # panel — the model just picks the option whose count differs from
        # the original. At L2+ no annotations (real rotation reasoning).
        annotate_count = cfg.get("allow_count_distractor", False)

        def _do_all():
            fig, axes = plt.subplots(1, 5, figsize=(18 * sc, 4.6 * sc))
            fig.patch.set_facecolor(bg)
            orig_title = "Original"
            if annotate_count:
                orig_title += f" ({len(original)} cubes)"
            axes[0].set_title(orig_title, fontsize=15, fontweight="bold", pad=10, **title_kw)
            _draw_poly(axes[0], original, palette_use, lw=lw_use, edge_col=edge_col)
            for i, opt in enumerate(options):
                t = f"({chr(65 + i)})"
                if annotate_count:
                    t += f"  {len(opt)} cubes"
                axes[i + 1].set_title(t, fontsize=15,
                                      fontweight="bold", pad=10, **title_kw)
                _draw_poly(axes[i + 1], opt, palette_use, lw=lw_use, edge_col=edge_col)
            for ax in axes:
                ax.set_aspect("equal"); ax.axis("off")
                ax.set_facecolor(bg)
                ax.autoscale_view(); ax.margins(0.25)
            try:
                fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.05, wspace=0.08)
            except Exception:
                pass
            return fig

        if mode == "sketch":
            with sketch_context(scale=1.0, length=80, randomness=1.5):
                fig = _do_all()
        else:
            fig = _do_all()
        img = self.fig_to_pil(fig, dpi=dpi)

        sidx = (self.seed or 0) % 16
        q = self._QUESTIONS[sidx]
        if annotate_count:
            q = (
                q
                + " Hint: each panel is labelled with its cube count. A "
                "rotation does not change the number of cubes — three "
                "options have the SAME count as the original; one option "
                "has a DIFFERENT count and therefore cannot be a rotation."
            )
        return q, correct_letter, img

    # ------------------------------------------------------------------ #
    def _build_non_rotation(self, original, cfg,
                            original_rot_set, rng) -> Optional[List]:
        """Produce a polycube that is NOT in the rotation orbit of `original`.
        For L0/L1, the distractor has a DIFFERENT cube count (so that
        cube-counting alone solves the problem)."""
        dtype = cfg["distractor_type"]

        # Level 0 / 1 — count_off: add 1-2 cubes or remove 1 cube.
        if dtype == "count_off":
            for _ in range(60):
                strat = rng.choice(["add", "add", "add_two"])
                if strat == "add":
                    cand = self._add_random_cube(original, rng)
                elif strat == "add_two":
                    tmp = self._add_random_cube(original, rng)
                    if tmp is not None:
                        cand = self._add_random_cube(tmp, rng)
                    else:
                        cand = None
                else:
                    cand = None
                if cand is None or not cand:
                    continue
                cand = _normalize(cand)
                if not self._is_connected(cand):
                    continue
                if frozenset(cand) in original_rot_set:
                    continue
                return cand

        if dtype == "mirror":
            for _ in range(30):
                axis = rng.choice(["x", "y", "z"])
                cand = _mirror(original, axis)
                if not self._is_connected(cand):
                    continue
                if frozenset(cand) in original_rot_set:
                    continue
                return cand
            # fallback — swap or move
            for _ in range(40):
                strat = rng.choice(["move_two", "swap_cube"])
                if strat == "move_two":
                    cand = self._move_a_cube(original, max_dist=2, rng=rng)
                else:
                    tmp = self._remove_random_cube(original, rng)
                    if tmp is None:
                        continue
                    cand = self._add_random_cube(tmp, rng)
                if cand is None:
                    continue
                cand = _normalize(cand)
                if frozenset(cand) in original_rot_set:
                    continue
                if not self._is_connected(cand):
                    continue
                return cand
            return None

        if dtype == "moved_two":
            strategies = ["mirror", "move_two", "swap_cube", "move_one"]
        else:  # moved_one
            strategies = ["move_one", "swap_cube", "mirror", "move_two"]

        for _ in range(120):
            strat = rng.choice(strategies)
            if strat == "mirror":
                cand = _mirror(original, rng.choice(["x", "y", "z"]))
            elif strat == "move_two":
                cand = self._move_a_cube(original, max_dist=2, rng=rng)
            elif strat == "move_one":
                cand = self._move_a_cube(original, max_dist=1, rng=rng)
            elif strat == "swap_cube":
                temp = self._remove_random_cube(original, rng)
                if temp is None:
                    continue
                cand = self._add_random_cube(temp, rng)
            else:
                continue
            if cand is None or not cand:
                continue
            cand = _normalize(cand)
            if frozenset(cand) in original_rot_set:
                continue
            if not self._is_connected(cand):
                continue
            return cand
        return None

    @staticmethod
    def _is_connected(cubes):
        if not cubes:
            return False
        s = set(cubes)
        seen = {cubes[0]}
        stack = [cubes[0]]
        dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        while stack:
            x, y, z = stack.pop()
            for dx, dy, dz in dirs:
                n = (x + dx, y + dy, z + dz)
                if n in s and n not in seen:
                    seen.add(n)
                    stack.append(n)
        return len(seen) == len(s)

    def _add_random_cube(self, cubes, rng):
        s = set(cubes)
        dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        candidates = set()
        for c in cubes:
            for d in dirs:
                nc = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
                if nc not in s:
                    candidates.add(nc)
        candidates = list(candidates)
        if not candidates:
            return None
        rng.shuffle(candidates)
        return list(cubes) + [candidates[0]]

    def _remove_random_cube(self, cubes, rng):
        if len(cubes) <= 2:
            return None
        idx_order = list(range(len(cubes)))
        rng.shuffle(idx_order)
        for i in idx_order:
            cand = [c for j, c in enumerate(cubes) if j != i]
            if self._is_connected(cand):
                return cand
        return None

    def _move_a_cube(self, cubes, max_dist, rng):
        order = list(range(len(cubes)))
        rng.shuffle(order)
        s = set(cubes)
        dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        for i in order:
            base = cubes[i]
            for _ in range(20):
                d = rng.choice(dirs)
                dist = rng.randint(1, max_dist)
                nc = (base[0] + d[0] * dist,
                      base[1] + d[1] * dist,
                      base[2] + d[2] * dist)
                if nc in s:
                    continue
                cand = [c for j, c in enumerate(cubes) if j != i] + [nc]
                if self._is_connected(cand):
                    return cand
        return None

    # ------------------------------------------------------------------ #
    def _pick_diverse_rotations(self, original, all_rots, n_picks, rng):
        def _fingerprint(cubes):
            pts = sorted([_iso_project(x, y, z) for (x, y, z) in cubes])
            return tuple(round(a + b, 1) for (a, b) in pts)

        orig_fp = _fingerprint(original)
        candidates = [r for r in all_rots if _fingerprint(r) != orig_fp]
        if len(candidates) < n_picks:
            candidates = all_rots[:]
        rng.shuffle(candidates)

        picked = []
        seen_fps = set()
        for r in candidates:
            fp = _fingerprint(r)
            if fp in seen_fps:
                continue
            picked.append(r)
            seen_fps.add(fp)
            if len(picked) >= n_picks:
                break
        while len(picked) < n_picks:
            picked.append(rng.choice(all_rots))
        return picked

if __name__ == "__main__":
    env = MultiAxisRotationChainQA()
    for lv in [0, 3, 6, 9]:
        for s in range(3):
            ok = env.generate(s, {"level": lv})
            print(f"L{lv} s{s}: ok={ok}, ans={env._answer}")
