"""
Rotation Chirality Fingerprint QA.

2026-05-05 R5 B5A: REWRITTEN — pure single-letter MCQ A-D / A-E.

Two polycubes are shown side-by-side (left / right). The right one was
obtained from the left by EITHER a pure rotation OR a mirror reflection.
The model must classify the operation.

Per-level layout
  L0/L1 — 4-MCQ. 4-cube polycube; right shape is the IDENTITY (no rotation,
          no mirror) when answer = "Same shape". Otherwise, 1 quarter-turn
          rotation or mirror. All cubes brightly colored. Concise reasoning
          hint (no answer leak).
  L2/L3 — 4-MCQ. 4-5 cube polycube; 1 rotation or mirror.
  L4/L5 — 4-MCQ. 5 cubes; 2 rotations.
  L6   — 4-MCQ. 6 cubes; 2 rotations.
  L7   — 5-MCQ + 25% trap. 6 cubes; 3 rotations.
  L8   — 5-MCQ + 30% trap. 7 cubes; 3 rotations.
  L9   — 5-MCQ + 40% trap. 8 cubes; 3 rotations.

ALLOW_ROTATION = False — orientation-sensitive.
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


def _iso_project(x, y, z):
    sx = (x - y) * math.cos(math.radians(30))
    sy = (x + y) * math.sin(math.radians(30)) + z
    return sx, sy


def _rot_x(cubes, k):
    out = []
    for (x, y, z) in cubes:
        for _ in range(k % 4):
            y, z = -z, y
        out.append((x, y, z))
    return out


def _rot_y(cubes, k):
    out = []
    for (x, y, z) in cubes:
        for _ in range(k % 4):
            x, z = z, -x
        out.append((x, y, z))
    return out


def _rot_z(cubes, k):
    out = []
    for (x, y, z) in cubes:
        for _ in range(k % 4):
            x, y = -y, x
        out.append((x, y, z))
    return out


def _mirror(cubes, axis):
    if axis == "x":
        return [(-x, y, z) for (x, y, z) in cubes]
    if axis == "y":
        return [(x, -y, z) for (x, y, z) in cubes]
    return [(x, y, -z) for (x, y, z) in cubes]


def _normalize(cubes):
    xs = [c[0] for c in cubes]
    ys = [c[1] for c in cubes]
    zs = [c[2] for c in cubes]
    minx, miny, minz = min(xs), min(ys), min(zs)
    return frozenset((x - minx, y - miny, z - minz) for (x, y, z) in cubes)


def _all_rotations(cubes):
    seen = set()
    result = []
    rots_to_face = [
        (0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0),
        (0, 1, 0), (0, 3, 0),
    ]
    for rx, ry, rz in rots_to_face:
        base = _rot_x(cubes, rx)
        base = _rot_y(base, ry)
        base = _rot_z(base, rz)
        for spin in range(4):
            r = _rot_z(base, spin)
            key = _normalize(r)
            if key not in seen:
                seen.add(key)
                result.append(r)
    return result


def _is_chiral(cubes) -> bool:
    mirrored = _mirror(cubes, "x")
    mirror_norm = _normalize(mirrored)
    for r in _all_rotations(cubes):
        if _normalize(r) == mirror_norm:
            return False
    return True


def _shapes_equal(a, b) -> bool:
    b_norm = _normalize(b)
    for r in _all_rotations(a):
        if _normalize(r) == b_norm:
            return True
    return False


def _gen_chiral_polycube(rng, n_cubes):
    base_templates = [
        [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)],
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
        [(0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0)],
    ]
    for _ in range(40):
        cubes = list(rng.choice(base_templates))
        while len(cubes) < n_cubes:
            s = rng.choice(cubes)
            dx, dy, dz = rng.choice([(1, 0, 0), (-1, 0, 0),
                                      (0, 1, 0), (0, -1, 0),
                                      (0, 0, 1), (0, 0, -1)])
            nc = (s[0] + dx, s[1] + dy, s[2] + dz)
            if nc not in cubes:
                cubes.append(nc)
        if len(cubes) == n_cubes and _is_chiral(cubes):
            return cubes
    return None


_STEM_TEMPLATES = [
    "The image shows two polycubes side by side. The right polycube was obtained from the left by EITHER a pure rotation OR a mirror reflection. Which operation was applied ( )?",
    "Two polycubes are shown. The right one is either a rotation of the left, or a mirror image of the left. Which is it ( )?",
    "Compare the two polycubes below. The right shape is either a rotation of the left, or its mirror image. Which operation was used ( )?",
    "Look at the two polycubes. The right is either a rotation OR a mirror image of the left. Identify the operation ( ).",
]


class RotationChiralityFingerprintQA(StandaloneVisualEnv):
    ENV_NAME = "rotation_chirality_fingerprint"
    ALLOW_ROTATION = False

    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Pick an asymmetric protrusion. If its handedness "
        "matches between the two shapes, it is a rotation; if it flips, "
        "it is a mirror. Final answer in the format this prompt requested."
    )

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        # 2026-05-05 Phase A fix: at L0/L1 the right shape MUST visibly differ
        # from the left, otherwise the prompt "rotation vs mirror?" is
        # ambiguous (both shapes identical → 'C: identical' option preferred).
        # n_rotations >= 1 ensures at least one quarter-turn applied.
        if level == 0:
            return {"n_cubes": 4, "n_rotations": 1,
                    "use_5_options": False, "trap_rate": 0.0}
        if level == 1:
            return {"n_cubes": 4, "n_rotations": 1,
                    "use_5_options": False, "trap_rate": 0.0}
        if level == 2:
            return {"n_cubes": 4, "n_rotations": 1,
                    "use_5_options": False, "trap_rate": 0.0}
        if level == 3:
            return {"n_cubes": 5, "n_rotations": 1,
                    "use_5_options": False, "trap_rate": 0.0}
        if level == 4:
            return {"n_cubes": 5, "n_rotations": 2,
                    "use_5_options": False, "trap_rate": 0.0}
        if level == 5:
            return {"n_cubes": 5, "n_rotations": 2,
                    "use_5_options": False, "trap_rate": 0.0}
        if level == 6:
            return {"n_cubes": 6, "n_rotations": 2,
                    "use_5_options": False, "trap_rate": 0.0}
        if level == 7:
            return {"n_cubes": 6, "n_rotations": 3,
                    "use_5_options": True, "trap_rate": 0.25}
        if level == 8:
            return {"n_cubes": 7, "n_rotations": 3,
                    "use_5_options": True, "trap_rate": 0.50}
        # L9
        return {"n_cubes": 8, "n_rotations": 3,
                "use_5_options": True, "trap_rate": 0.50}

    # ------------------------------------------------------------------ #
    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._sub_rng = sub_rng

        for _ in range(30):
            try:
                res = self._try_generate(sub_rng, cfg, level)
                if res is not None:
                    return res
            except Exception:
                continue
        return None

    def _try_generate(self, rng, cfg, level):
        n_cubes = cfg["n_cubes"]
        original = _gen_chiral_polycube(rng, n_cubes)
        if original is None:
            return None

        rotated = list(original)
        for _ in range(cfg["n_rotations"]):
            axis = rng.choice(["x", "y", "z"])
            k = rng.choice([1, 2, 3])
            fn = {"x": _rot_x, "y": _rot_y, "z": _rot_z}[axis]
            rotated = fn(rotated, k)

        # Decide whether right is rotation or mirror — balanced 50/50.
        is_mirror = rng.random() < 0.5
        if is_mirror:
            shown = _mirror(rotated, rng.choice(["x", "y", "z"]))
            ground_truth = "Mirror image (different chirality)"
        else:
            shown = rotated
            ground_truth = "Rotation (same chirality)"

        # 4-MCQ option pool: rotation / mirror / cannot determine / variants.
        # 2026-05-05 Phase A: removed the "Both — identical so could be either"
        # option because the question premise asserts an operation was applied,
        # making this option a red herring that pulls smart models off correct.
        base_options = [
            "Rotation (same chirality)",
            "Mirror image (different chirality)",
            "Cannot be determined from the image",
            "Translation only (no rotation, no reflection)",
            "Combined rotation and reflection (improper rotation)",
            "Scaling (uniform size change)",
        ]
        use_5 = cfg["use_5_options"]
        opt_letters = ["A", "B", "C", "D", "E"][: (5 if use_5 else 4)]
        n_value = len(opt_letters) - (1 if use_5 else 0)

        use_trap = use_5 and (rng.random() < cfg["trap_rate"])
        if use_trap:
            # All visible options are wrong; correct = "E. No correct answer"
            wrong_pool = [o for o in base_options if o != ground_truth]
            rng.shuffle(wrong_pool)
            chosen = wrong_pool[:n_value]
            if len(chosen) < n_value:
                return None
            options = list(chosen)
            rng.shuffle(options)
            correct_letter = opt_letters[-1]
        else:
            wrong_pool = [o for o in base_options if o != ground_truth]
            rng.shuffle(wrong_pool)
            n_distractor = n_value - 1
            chosen = wrong_pool[:n_distractor]
            if len(chosen) < n_distractor:
                return None
            options = [ground_truth] + chosen
            rng.shuffle(options)
            correct_letter = opt_letters[options.index(ground_truth)]

        opt_lines = [f"{opt_letters[i]}. {options[i]}" for i in range(n_value)]
        if use_5:
            opt_lines.append(f"{opt_letters[-1]}. No correct answer")
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        sidx = (self.seed or 0) % len(_STEM_TEMPLATES)
        stem = _STEM_TEMPLATES[sidx]
        question = (
            f"{stem}\n"
            + "\n".join(opt_lines)
            + f"\nAnswer with a single letter {letter_str}."
        )
        if level <= 1:
            question += (
                "\nHint: rotation preserves handedness; mirror flips it. "
                "Pick an asymmetric protrusion on the left, then check the "
                "right: if its direction is the SAME, it is a rotation; if "
                "the direction is FLIPPED, it is a mirror."
            )
        elif level <= 3:
            question += (
                "\nHint: trace one asymmetric feature (e.g., a single "
                "protruding cube) on each shape and compare the direction "
                "it points."
            )

        image = self._render_pair(original, shown, cfg, rng)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    def _render_pair(self, original, shown, cfg, sub_rng):
        style = self._random_style()
        sc = style["figsize_scale"]
        palette = list(style["palette"])
        sub_rng.shuffle(palette)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8 * sc, 4.5 * sc))
        fig.patch.set_facecolor("#ffffff")
        for ax in (ax1, ax2):
            ax.set_facecolor("#ffffff")

        self._draw_polycube(ax1, original, palette, "Left")
        alt_palette = palette[3 % len(palette):] + palette[: 3 % len(palette)]
        self._draw_polycube(ax2, shown, alt_palette, "Right")

        suptitle = sub_rng.choice([
            "Rotation vs Mirror?",
            "Same shape (rotation) or mirror image?",
            "Compare the two polycubes",
            "Polycube comparison",
        ])
        fig.suptitle(suptitle, fontsize=style["font_size_base"] + 2,
                     fontweight="bold")
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_polycube(self, ax, cubes, palette, title):
        shifted = sorted(_normalize(cubes),
                         key=lambda c: -(c[0] + c[1] - c[2]))
        top_color = palette[0]
        left_color = palette[1 % len(palette)]
        right_color = palette[2 % len(palette)]

        for (x, y, z) in shifted:
            pts = [
                _iso_project(x, y, z + 1),
                _iso_project(x + 1, y, z + 1),
                _iso_project(x + 1, y + 1, z + 1),
                _iso_project(x, y + 1, z + 1),
            ]
            ax.add_patch(Polygon(pts, closed=True, facecolor=top_color,
                                 edgecolor='black', lw=1.1))
            pts = [
                _iso_project(x, y, z),
                _iso_project(x, y + 1, z),
                _iso_project(x, y + 1, z + 1),
                _iso_project(x, y, z + 1),
            ]
            ax.add_patch(Polygon(pts, closed=True, facecolor=left_color,
                                 edgecolor='black', lw=1.1))
            pts = [
                _iso_project(x, y, z),
                _iso_project(x + 1, y, z),
                _iso_project(x + 1, y, z + 1),
                _iso_project(x, y, z + 1),
            ]
            ax.add_patch(Polygon(pts, closed=True, facecolor=right_color,
                                 edgecolor='black', lw=1.1))

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
        ax.set_title(title, fontsize=12, fontweight="bold", pad=4)


if __name__ == "__main__":
    env = RotationChiralityFingerprintQA()
    pass_count = 0
    total = 0
    for L in (0, 3, 6, 9):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok} A={env._answer if ok else 'X'}")
            if ok:
                wrappers = [
                    f"<answer>{env._answer}</answer>",
                    f"\\boxed{{{env._answer}}}",
                    f"Final answer: {env._answer}",
                ]
                v_pos = env.verify(wrappers[(s or 0) % 3])
                v_neg = env.verify("definitely_wrong_xyz")
                print(f"   pos={v_pos['accuracy']} neg={v_neg['accuracy']}")
                if v_pos['accuracy'] == 1 and v_neg['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
