"""
Three View Projection Ultra Easy QA environment — warmup rescue.

Template: three_view_projection_warmup_qa.py (simplified for L0).

Goal: rescue batch-1 `three_view_projection_warmup` which stuck at
L0 = 0.2 on three-view projection. This env
starts at L0 with a SINGLE cube (top view is always 1x1), and the
distractors have wildly different cube counts. At L9 there are 6
cubes in an asymmetric structure.

Difficulty schedule (multi-axis, continuous):
  Axis 1 (primary): n_cubes = 1 + level // 2    -> 1..6
  Axis 2           : distractor_mode              large-size-difference -> one-cell
  Axis 3 (optional): show_front_and_side = level <= 3

Output format is constant: 4-option MCQ, single letter.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

def _iso_project(x, y, z):
    sx = (x - y) * math.cos(math.radians(30))
    sy = (x + y) * math.sin(math.radians(30)) + z
    return sx, sy

def _normalize_cubes(cubes):
    if not cubes:
        return []
    xs = [c[0] for c in cubes]
    ys = [c[1] for c in cubes]
    zs = [c[2] for c in cubes]
    mx, my, mz = min(xs), min(ys), min(zs)
    return [(x - mx, y - my, z - mz) for (x, y, z) in cubes]

def _dims(cubes):
    xs = [c[0] for c in cubes]
    ys = [c[1] for c in cubes]
    zs = [c[2] for c in cubes]
    return max(xs) + 1, max(ys) + 1, max(zs) + 1

def _top_view(cubes):
    gx, gy, _ = _dims(cubes)
    grid = np.zeros((gy, gx), dtype=int)
    for (x, y, _z) in cubes:
        grid[gy - 1 - y, x] = 1
    return grid

def _front_view(cubes):
    gx, _, gz = _dims(cubes)
    grid = np.zeros((gz, gx), dtype=int)
    for (x, _y, z) in cubes:
        grid[gz - 1 - z, x] = 1
    return grid

def _side_view(cubes):
    _, gy, gz = _dims(cubes)
    grid = np.zeros((gz, gy), dtype=int)
    for (_x, y, z) in cubes:
        grid[gz - 1 - z, gy - 1 - y] = 1
    return grid

def _views(cubes):
    return _top_view(cubes), _front_view(cubes), _side_view(cubes)

def _views_equal(a, b):
    t1, f1, s1 = a
    t2, f2, s2 = b
    return (t1.shape == t2.shape and np.array_equal(t1, t2)
            and f1.shape == f2.shape and np.array_equal(f1, f2)
            and s1.shape == s2.shape and np.array_equal(s1, s2))

class ThreeViewProjectionUltraEasyQA(StandaloneVisualEnv):
    ENV_NAME = "three_view_projection_ultra_easy"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0=1 cube, distractors differ in cube count (trivial). Previously
        # L3 jumped to 3 cubes AND mixed_n=False (same cube count across
        # distractors), crushing pass-rate (1.00 → 0.20). Now mixed_n stays
        # True through L3 and n_cubes grows more gradually.
        # Iter 3 (2026-04-17): L3=0.20 persisted after iter-2's mixed_n fix.
        # The jump from 2 cubes (L2) to 3 cubes at L3 is the root problem.
        # Keep L3 on n_cubes=2 with harder visual config (no show_both
        # silhouette) so difficulty grows via view sparsity not polycube
        # count. Start 3-cube shapes at L4 with mixed_n=True for gentle
        # ramp.
        if level == 0:
            return dict(n_cubes=1, show_both=True, tight=False, mixed_n=True)
        if level == 1:
            return dict(n_cubes=2, show_both=True, tight=False, mixed_n=True)
        if level == 2:
            return dict(n_cubes=2, show_both=True, tight=False, mixed_n=True)
        if level == 3:
            return dict(n_cubes=2, show_both=False, tight=False, mixed_n=True)
        if level == 4:
            return dict(n_cubes=3, show_both=True, tight=False, mixed_n=True)
        if level == 5:
            return dict(n_cubes=4, show_both=True, tight=False, mixed_n=False)
        if level == 6:
            return dict(n_cubes=4, show_both=False, tight=False, mixed_n=False)
        if level == 7:
            return dict(n_cubes=5, show_both=False, tight=True, mixed_n=False)
        if level == 8:
            return dict(n_cubes=5, show_both=False, tight=True, mixed_n=False,
                        super_tight=True)
        # L9 iter-4 (2026-04-17): passrate=0.85 borderline. Bump to 7 cubes
        # AND enable super_tight so distractors are perturbed (±1 cube).
        return dict(n_cubes=7, show_both=False, tight=True, mixed_n=False,
                    super_tight=True)

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["n_cubes"]
        self._sub_rng = sub_rng

        for _ in range(25):
            r = self._try_once(sub_rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_once(self, sub_rng, cfg, level):
        n = cfg["n_cubes"]

        candidates = self._shape_bank(sub_rng, level)
        sub_rng.shuffle(candidates)
        candidates = [_normalize_cubes(c) for c in candidates]
        same_n = [c for c in candidates if len(c) == n]
        if not same_n:
            return None
        correct = same_n[0]
        correct_views = _views(correct)

        pool = []
        for c in candidates:
            if len(c) == n and not _views_equal(_views(c), correct_views):
                pool.append(c)

        if cfg.get("tight"):
            for _ in range(30):
                perturbed = list(correct)
                s = sub_rng.choice(perturbed)
                dx, dy, dz = sub_rng.choice([(1, 0, 0), (-1, 0, 0),
                                             (0, 1, 0), (0, -1, 0), (0, 0, 1)])
                nc = (s[0] + dx, s[1] + dy, s[2] + dz)
                cand = _normalize_cubes(perturbed + [nc])
                if len(set(cand)) == n + 1 and not _views_equal(_views(cand), correct_views):
                    pool.append(cand)

        if cfg.get("super_tight"):
            # Also try removing a cube (gives n-1 cube distractor)
            for _ in range(20):
                remove_idx = sub_rng.randint(0, len(correct) - 1)
                perturbed = [c for i, c in enumerate(correct) if i != remove_idx]
                if len(perturbed) >= 1:
                    cand = _normalize_cubes(perturbed)
                    if not _views_equal(_views(cand), correct_views):
                        pool.append(cand)

        if cfg.get("mixed_n"):
            for alt_n in [n + 1, n + 2, n + 3, max(1, n - 1)]:
                for c in candidates:
                    if len(c) == alt_n:
                        pool.append(c)
                        break

        # Key distractors by their TOP VIEW (the visible MCQ option shape),
        # not by the underlying cube set. Two different polycubes can share
        # the same top-view silhouette, which would produce duplicate visual
        # options otherwise.
        def _topview_key(cubes):
            tv = _top_view(_normalize_cubes(cubes))
            return (tv.shape, tv.tobytes())

        seen_keys = {_topview_key(correct)}
        distractors = []
        for c in pool:
            key = _topview_key(c)
            if key not in seen_keys:
                distractors.append(c)
                seen_keys.add(key)
            if len(distractors) >= 3:
                break
        tries = 0
        while len(distractors) < 3 and tries < 40:
            tries += 1
            cand = self._random_polycube(sub_rng, n + sub_rng.randint(-1, 2))
            key = _topview_key(cand)
            if key not in seen_keys:
                distractors.append(cand)
                seen_keys.add(key)
        if len(distractors) < 3:
            return None

        options = [correct] + distractors[:3]
        order = list(range(4))
        sub_rng.shuffle(order)
        shuffled = [options[i] for i in order]
        correct_letter = chr(ord("A") + order.index(0))

        views_str = "front and side" if cfg.get("show_both") else "front"
        stem = self._rng.choice([
            f"The isometric view of a 3D polycube structure is shown on the left, together with its {views_str} view(s). Which of the four options on the right correctly shows the TOP view of this structure? Answer with a single letter.",
            f"A 3D cube-built shape is drawn on the left (with its {views_str} view as reference). Pick the option below that correctly shows the TOP view. Answer with a single letter (A, B, C, or D).",
            f"Study the 3D structure and its {views_str} view. Which option shows the correct TOP projection? Answer with one letter.",
        ])

        image = self._render(correct, shuffled, cfg, sub_rng)
        return stem, correct_letter, image

    def _shape_bank(self, rng, level):
        banks = []
        banks.append([(0, 0, 0)])
        banks.append([(0, 0, 0), (1, 0, 0)])
        banks.append([(0, 0, 0), (0, 0, 1)])
        banks.append([(0, 0, 0), (0, 1, 0)])
        banks.append([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
        banks.append([(0, 0, 0), (1, 0, 0), (2, 0, 0)])
        banks.append([(0, 0, 0), (1, 0, 0), (1, 0, 1)])
        banks.append([(0, 0, 0), (1, 0, 0), (0, 0, 1)])
        banks.append([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)])
        banks.append([(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0)])
        banks.append([(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 0, 1)])
        banks.append([(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)])
        banks.append([(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 0, 1)])
        banks.append([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
        banks.append([(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (0, 2, 0)])
        banks.append([(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 0, 1)])
        banks.append([(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (0, 1, 1)])
        banks.append([(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0)])
        # 7-cube shapes for L9 iter-4
        banks.append([(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0),
                      (2, 1, 0), (1, 0, 1)])
        banks.append([(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (0, 2, 0),
                      (0, 0, 1), (1, 0, 1)])
        banks.append([(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0),
                      (0, 0, 1), (0, 1, 1)])
        return banks

    def _random_polycube(self, rng, n):
        n = max(1, n)
        cubes = [(0, 0, 0)]
        while len(cubes) < n:
            s = rng.choice(cubes)
            dx, dy, dz = rng.choice([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
            nc = (s[0] + dx, s[1] + dy, s[2] + dz)
            if nc not in cubes:
                cubes.append(nc)
        return _normalize_cubes(cubes)

    def _render(self, correct, shuffled, cfg, sub_rng):
        style = self._random_style()
        sc = style["figsize_scale"]
        palette = list(style["palette"])
        sub_rng.shuffle(palette)
        fig = plt.figure(figsize=(12 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])

        ax_iso = fig.add_subplot(2, 4, (1, 2))
        ax_iso.set_facecolor(style["bg_color"])
        self._draw_iso(ax_iso, correct, palette, sub_rng)
        iso_title = sub_rng.choice([
            "3D structure (isometric)", "3D polycube", "Isometric view",
            "Cube assembly (3D)"])
        ax_iso.set_title(iso_title, fontsize=10, fontweight="bold")

        front = _front_view(_normalize_cubes(correct))
        grid_fill = sub_rng.choice(["#34495e", "#2b2d42", "#1d3557", "#606c38", "#3f0d0d"])
        if cfg.get("show_both"):
            ax_front = fig.add_subplot(2, 4, 3)
            ax_front.set_facecolor(style["bg_color"])
            self._draw_grid(ax_front, front, grid_fill)
            ax_front.set_title("Front view", fontsize=9, fontweight="bold")
            side = _side_view(_normalize_cubes(correct))
            ax_side = fig.add_subplot(2, 4, 4)
            ax_side.set_facecolor(style["bg_color"])
            self._draw_grid(ax_side, side, grid_fill)
            ax_side.set_title("Side view", fontsize=9, fontweight="bold")
        else:
            ax_front = fig.add_subplot(2, 4, (3, 4))
            ax_front.set_facecolor(style["bg_color"])
            self._draw_grid(ax_front, front, grid_fill)
            ax_front.set_title("Front view", fontsize=9, fontweight="bold")

        for i, cand in enumerate(shuffled):
            ax = fig.add_subplot(2, 4, 5 + i)
            ax.set_facecolor(style["bg_color"])
            self._draw_grid(ax, _top_view(_normalize_cubes(cand)), grid_fill)
            ax.set_title(f"({chr(ord('A') + i)})", fontsize=10, fontweight="bold")

        suptitle = sub_rng.choice([
            "Three-View Projection (Top View?)",
            "Which option shows the top view?",
            "Match the TOP projection",
            "Select the correct top view",
        ])
        fig.suptitle(suptitle,
                     fontsize=style["font_size_base"] + 2, fontweight="bold")
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_iso(self, ax, cubes, palette, sub_rng=None):
        shifted = sorted(_normalize_cubes(cubes), key=lambda c: -(c[0] + c[1] - c[2]))
        top_c = palette[0]
        left_c = palette[1 % len(palette)]
        right_c = palette[2 % len(palette)]
        for (x, y, z) in shifted:
            pts = [
                _iso_project(x, y, z + 1),
                _iso_project(x + 1, y, z + 1),
                _iso_project(x + 1, y + 1, z + 1),
                _iso_project(x, y + 1, z + 1),
            ]
            ax.add_patch(Polygon(pts, closed=True, facecolor=top_c,
                                 edgecolor="black", lw=1.2))
            pts = [
                _iso_project(x, y, z),
                _iso_project(x, y + 1, z),
                _iso_project(x, y + 1, z + 1),
                _iso_project(x, y, z + 1),
            ]
            ax.add_patch(Polygon(pts, closed=True, facecolor=left_c,
                                 edgecolor="black", lw=1.2))
            pts = [
                _iso_project(x, y, z),
                _iso_project(x + 1, y, z),
                _iso_project(x + 1, y, z + 1),
                _iso_project(x, y, z + 1),
            ]
            ax.add_patch(Polygon(pts, closed=True, facecolor=right_c,
                                 edgecolor="black", lw=1.2))
        pts = []
        for (x, y, z) in shifted:
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        pts.append(_iso_project(x + dx, y + dy, z + dz))
        if pts:
            arr = np.array(pts)
            mg = 0.4
            ax.set_xlim(arr[:, 0].min() - mg, arr[:, 0].max() + mg)
            ax.set_ylim(arr[:, 1].min() - mg, arr[:, 1].max() + mg)
        ax.set_aspect("equal")
        ax.axis("off")

    def _draw_grid(self, ax, grid, color):
        h, w = grid.shape
        for i in range(h):
            for j in range(w):
                if grid[i, j]:
                    rect = Rectangle((j, h - 1 - i), 1, 1,
                                     facecolor=color, edgecolor="black", linewidth=1.0)
                    ax.add_patch(rect)
        ax.set_xlim(-0.5, w + 0.5)
        ax.set_ylim(-0.5, h + 0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

if __name__ == "__main__":
    import os
    os.makedirs("/tmp/env_check_b2a", exist_ok=True)
    env = ThreeViewProjectionUltraEasyQA()
    for level in [0, 3, 6]:
        for seed in [1, 2, 3]:
            ok = env.generate(seed=seed, parameter={"level": level})
            if not ok:
                print(f"[three_view_projection_ultra_easy] L{level} s{seed} FAILED")
                continue
            out = f"/tmp/env_check_b2a/three_view_projection_ultra_easy_s{seed}_L{level}.png"
            env.render().save(out)
            print(f"saved {out} | A={env._answer} | ncubes={env._primary_complexity_feature}")
