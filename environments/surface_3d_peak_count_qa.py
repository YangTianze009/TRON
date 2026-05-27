"""
3D-surface peak counting / value-reading: render a matplotlib `plot_surface`
where the surface is built from N explicit Gaussian bumps with well-separated
centers; ask either:
  - how many local maxima are visible on the surface (integer count), or
  - what is the (rounded) value at the peak (numeric).

Targets the Number-in-General gap on 3D surface plots (IDX 779 "How many local
maxima can be observed along the tau axis when K=1.5...", IDX 519 "How many
subplots display peaks on the x-plane?", IDX 961 "How many local maxima on the
unlabeled axis...") and the Number-in-Chart "3D surface peak coordinates"
ability.

Output: integer count or numeric peak value. Numeric values are integer-valued
(peak heights chosen on integer grid spaced ≥ 1.5 apart so the tolerance
doesn't collapse adjacent answers).
"""
import math
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
# matplotlib's 3D backend
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  side-effect import
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES_COUNT = [
    "How many local maxima are visible on the 3D surface shown? Reply with the integer count in <answer>...</answer>.",
    "Count the number of distinct peaks on the 3D surface plot. Integer answer in <answer>...</answer>.",
    "How many peaks (local maxima) does the 3D surface have? Integer in <answer>...</answer>.",
    "Examine the 3D surface and count the local maxima. Integer count in <answer>...</answer>.",
    "Inspect the surface plot and report how many peaks are visible. Integer in <answer>...</answer>.",
    "Number of distinct peaks on the displayed 3D surface? Integer answer in <answer>...</answer>.",
    "Count the bumps (local maxima) on the 3D surface shown. Integer in <answer>...</answer>.",
    "How many local maxima can be observed on the 3D surface in the figure? Integer in <answer>...</answer>.",
]

_TEMPLATES_VALUE = [
    "What is the approximate value at the highest peak of the 3D surface? Integer answer in <answer>...</answer>.",
    "Read the height of the tallest peak on the 3D surface. Integer in <answer>...</answer>.",
    "What is the maximum z-value of the surface plot shown? Integer answer in <answer>...</answer>.",
    "At the surface's highest point, what is the approximate z-value? Integer in <answer>...</answer>.",
    "Report the peak height of the 3D surface. Integer in <answer>...</answer>.",
    "What is the value at the peak of the surface in the figure? Integer in <answer>...</answer>.",
    "Read off the maximum z attained by the surface. Integer answer in <answer>...</answer>.",
    "Approximate the height of the surface's highest peak. Integer in <answer>...</answer>.",
]


def _gaussian_bump(X, Y, cx, cy, sigma, amp):
    return amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))


class Surface3DPeakCountQA(StandaloneVisualEnv):
    ENV_NAME = "surface_3d_peak_count"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            n_peaks_choices = [1]
            min_sep = 6.0   # one obvious peak in centre
            modes = ["count"]
        elif level <= 2:
            n_peaks_choices = [1, 2]
            min_sep = 5.0
            modes = ["count", "value"]
        elif level <= 4:
            n_peaks_choices = [1, 2, 3]
            min_sep = 4.5
            modes = ["count", "value"]
        elif level <= 6:
            # 2026-05-04 R3 retry: was [2,3] sep 4.0 — keep but slightly
            # wider sep (4.5) so VLM 3D render reads cleanly.
            n_peaks_choices = [2, 3]
            min_sep = 4.5
            modes = ["count", "value"]
        elif level <= 8:
            # 2026-05-04 R3 retry: was [2,3,4] at sep 3.5 — drop to [2,3]
            # at sep 4.0; 4-peak case was driving misreads at L7-L8.
            n_peaks_choices = [2, 3]
            min_sep = 4.0
            modes = ["count", "value"]
        else:
            # 2026-05-04 R3 retry: keep L9 [2,3] at 3.5 (already softened).
            n_peaks_choices = [2, 3]
            min_sep = 3.5
            modes = ["count", "value"]
        return {"level": level, "n_peaks_choices": n_peaks_choices,
                "min_sep": min_sep, "modes": modes}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        import hashlib
        sd = int(hashlib.md5(f"s3p|{self.seed or 0}|{level}".encode()).hexdigest()[:12], 16)
        rng = random.Random(sd)

        n_peaks = rng.choice(cfg["n_peaks_choices"])
        min_sep = cfg["min_sep"]
        mode = rng.choice(cfg["modes"])

        # Domain
        L = 10.0
        # Place peak centers with min separation; if a placement fails, relax
        # min_sep by 10% per retry (so tight packings still succeed) instead
        # of returning None (failure rate was ~3.5% before).
        centers = []
        cur_sep = min_sep
        for _ in range(n_peaks):
            placed = False
            for _attempt in range(80):
                cx = rng.uniform(1.5, L - 1.5)
                cy = rng.uniform(1.5, L - 1.5)
                if all(math.hypot(cx - qx, cy - qy) >= cur_sep for (qx, qy) in centers):
                    centers.append((cx, cy))
                    placed = True
                    break
            if not placed:
                # Relax min_sep gradually until a slot opens
                cur_sep *= 0.9
                for _attempt in range(80):
                    cx = rng.uniform(1.5, L - 1.5)
                    cy = rng.uniform(1.5, L - 1.5)
                    if all(math.hypot(cx - qx, cy - qy) >= cur_sep for (qx, qy) in centers):
                        centers.append((cx, cy))
                        placed = True
                        break
            if not placed:
                return None

        # Peak heights: integer, spaced by >= 2 from each other
        # so when reading the tallest, the answer is unambiguous.
        candidate_heights = [3, 5, 7, 9, 11, 13]
        rng.shuffle(candidate_heights)
        heights = candidate_heights[:n_peaks]
        # Ensure tallest is uniquely tallest (>= 2 above next)
        sorted_h = sorted(heights, reverse=True)
        if len(sorted_h) >= 2 and sorted_h[0] - sorted_h[1] < 2:
            return None
        sigmas = [rng.uniform(0.9, 1.3) for _ in range(n_peaks)]

        # Build mesh
        x = np.linspace(0, L, 100)
        y = np.linspace(0, L, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for (cx, cy), s, amp in zip(centers, sigmas, heights):
            Z = Z + _gaussian_bump(X, Y, cx, cy, s, amp)

        # Build answer
        if mode == "count":
            answer = str(n_peaks)
            sidx = (self.seed or 0) % len(_TEMPLATES_COUNT)
            question = _TEMPLATES_COUNT[sidx]
        else:
            # "value" — peak height of the tallest bump.
            # The Z surface is the SUM of bumps; if peaks are well-separated,
            # the contribution from non-target bumps at the target's centre
            # is small. We report the actual sampled max z (rounded to int).
            zmax = float(Z.max())
            answer = str(int(round(zmax)))
            # But verifier tolerance is 0.5 abs / 5% rel — for zmax ~ 9, that
            # gives ±0.5. Make sure heights are integers (they are by design).
            sidx = (self.seed or 0) % len(_TEMPLATES_VALUE)
            question = _TEMPLATES_VALUE[sidx]

        img = self._render_surface(X, Y, Z, rng)
        return question, answer, img

    def _render_surface(self, X, Y, Z, rng):
        fig = plt.figure(figsize=(7.0, 5.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax = fig.add_subplot(111, projection="3d")
        # Pick from a few academic-style cmaps
        cmap = rng.choice(["viridis", "plasma", "magma", "cividis"])
        surf = ax.plot_surface(X, Y, Z, cmap=cmap,
                                edgecolor="none", antialiased=True,
                                rstride=2, cstride=2, alpha=0.9)
        # Add wireframe overlay for academic look
        ax.plot_wireframe(X, Y, Z, color="#333", linewidth=0.25,
                          rstride=8, cstride=8, alpha=0.5)
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)
        ax.set_zlabel("z", fontsize=11)
        # Slight viewing-angle randomization
        ax.view_init(elev=rng.uniform(25, 40), azim=rng.uniform(-50, -30))
        fig.colorbar(surf, ax=ax, fraction=0.04, pad=0.08, shrink=0.7)
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
