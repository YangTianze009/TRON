"""
Ptolemy's Theorem on a Cyclic Quadrilateral QA (D55, P3 — reference plane
geometry).

Reference an external reference:
  "Let ABCD be a cyclic convex quadrilateral, AD = 1, DC = 6, BC = 7,
   AB = 8. Determine the value of AC * BD."  Ans: 55

Ptolemy's theorem: in a cyclic quadrilateral ABCD with diagonals AC and
BD,
    AC * BD = AB * CD + AD * BC.

Given the four sides, the env asks for AC * BD (the product of the
diagonals). Optional variants ask for one diagonal length given the other
and the four sides, or for a missing side given everything else.

Verifier: float / integer (`\\boxed{55}` or bare 55).

Difficulty:
  L0..L2 — sides ≤ 10, ask AC * BD = product of diagonals.
  L3..L5 — sides ≤ 15, ask AC * BD; non-integer answer rounded to 2 d.p.
  L6..L7 — sides ≤ 20, ask one diagonal given the other.
  L8..L9 — sides ≤ 25, ask the missing side given the other three sides
            and AC*BD.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


def _dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


class PtolemyQuadQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "ptolemy_quad"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {"max_side": 10, "qtype": "product_diagonals"}
        if level <= 5:
            return {"max_side": 15, "qtype": "product_diagonals"}
        if level <= 7:
            return {"max_side": 20, "qtype": "one_diagonal"}
        return {"max_side": 25, "qtype": "missing_side"}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 5077 + level * 257 + 11)

        for _ in range(40):
            r = self._try_generate(rng, cfg)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg):
        # Pick four points on a circle (radius R, four angles), compute
        # the chord-side lengths as the actual distances. This guarantees
        # ABCD is cyclic by construction.
        R = rng.uniform(3.0, 5.0)
        # Four angles in increasing order, gaps each ≥ 30°
        angles = [0.0]
        for k in range(3):
            angles.append(angles[-1] + math.radians(rng.uniform(40, 130)))
        # Total wrap < 360°
        if angles[-1] >= math.radians(330):
            return None
        # Optional rotation
        rot = math.radians(rng.uniform(0, 360))
        angles = [a + rot for a in angles]

        pts = [(R * math.cos(a), R * math.sin(a)) for a in angles]
        A, B, C, D = pts

        AB = _dist(A, B)
        BC = _dist(B, C)
        CD = _dist(C, D)
        DA = _dist(D, A)
        AC = _dist(A, C)
        BD = _dist(B, D)

        # Verify Ptolemy holds (sanity).
        ptolemy_lhs = AC * BD
        ptolemy_rhs = AB * CD + DA * BC
        if abs(ptolemy_lhs - ptolemy_rhs) > 0.05 * max(ptolemy_lhs, 1.0):
            return None

        # Try to round each side to a nearby integer if scaling permits.
        # Scale so AB rounds to a clean integer, then check the others.
        max_side = cfg["max_side"]
        # Scale factor candidates: choose so AB ∈ {2, 3, ..., max_side}
        target_AB = rng.randint(2, max(3, max_side - 1))
        scale = target_AB / AB
        AB_s = AB * scale
        BC_s = BC * scale
        CD_s = CD * scale
        DA_s = DA * scale
        AC_s = AC * scale
        BD_s = BD * scale

        # Reject if any side is too small or too big
        for s_val in (AB_s, BC_s, CD_s, DA_s):
            if s_val < 1.0 or s_val > max_side + 5:
                return None

        # Round sides to integers and check Ptolemy still close.
        AB_i = round(AB_s)
        BC_i = round(BC_s)
        CD_i = round(CD_s)
        DA_i = round(DA_s)
        # If integer rounding distorts too much, give up this seed
        for orig, rounded in [(AB_s, AB_i), (BC_s, BC_i), (CD_s, CD_i), (DA_s, DA_i)]:
            if abs(orig - rounded) > 0.4:
                return None
        # Compute target with INTEGER sides
        prod_int = AB_i * CD_i + DA_i * BC_i
        if prod_int <= 0:
            return None

        sides = {"AB": AB_i, "BC": BC_i, "CD": CD_i, "DA": DA_i}

        qtype = cfg["qtype"]

        if qtype == "product_diagonals":
            answer = prod_int  # AC * BD
            ans_str = str(answer)
            question = (
                f"Let ABCD be a cyclic convex quadrilateral inscribed in a "
                f"circle, as shown. AB = {AB_i}, BC = {BC_i}, CD = {CD_i}, "
                f"DA = {DA_i}. Determine the value of AC * BD (the product "
                f"of the diagonals)."
            )
        elif qtype == "one_diagonal":
            # Give AC, ask for BD (or vice versa).
            # We need AC to be a "nice" value to give the model.
            # Pick AC as integer rounded (use the chord-distance scaled).
            AC_i = round(AC_s)
            if AC_i < 2 or abs(AC_s - AC_i) > 0.4:
                return None
            BD_target = prod_int / AC_i
            BD_target_r = round(BD_target, 2)
            if BD_target_r <= 0.5 or BD_target_r > max_side + 10:
                return None
            answer = BD_target_r
            if abs(answer - round(answer)) < 0.005:
                ans_str = str(int(round(answer)))
            else:
                ans_str = f"{answer:g}"
            question = (
                f"In cyclic quadrilateral ABCD inscribed in a circle (as "
                f"shown), AB = {AB_i}, BC = {BC_i}, CD = {CD_i}, "
                f"DA = {DA_i}, and the diagonal AC = {AC_i}. Find the "
                f"length of the other diagonal BD. Provide your answer as "
                f"a decimal number."
            )
        else:  # missing_side: give 3 sides + AC*BD, ask for 4th side
            # Hide DA. Give AB, BC, CD, and product P = AC*BD. Then
            # DA = (P - AB*CD) / BC.
            P_int = prod_int
            denom = BC_i
            numer = P_int - AB_i * CD_i
            if denom == 0 or numer <= 0:
                return None
            DA_target = numer / denom
            if abs(DA_target - round(DA_target)) > 0.05:
                return None
            DA_target_r = round(DA_target)
            if DA_target_r != DA_i or DA_target_r < 1:
                return None
            answer = DA_target_r
            ans_str = str(answer)
            question = (
                f"ABCD is a cyclic quadrilateral inscribed in a circle (as "
                f"shown). AB = {AB_i}, BC = {BC_i}, CD = {CD_i}. The "
                f"product of the diagonals is AC * BD = {P_int}. Find the "
                f"length of side DA. Provide your answer as an integer."
            )

        # Render the cyclic quad with side labels
        img = self._render(pts, sides, qtype, AC_i if qtype == "one_diagonal" else None,
                           prod_int if qtype == "missing_side" else None)
        return question, ans_str, img

    # ------------------------------------------------------------------ #
    def _render(self, pts, sides, qtype, given_diag, given_prod) -> Image.Image:
        A, B, C, D = pts
        fig, ax = plt.subplots(1, 1, figsize=(6.0, 6.0), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        # Centre + radius (assume circle is centred near origin)
        cx = sum(p[0] for p in pts) / 4
        cy = sum(p[1] for p in pts) / 4
        # Circle through points (use first chord-distance as radius proxy
        # using actual circumradius computed from centre origin)
        R = max(_dist((cx, cy), p) for p in pts)
        circle = plt.Circle((cx, cy), R, fill=False,
                            edgecolor="#3498db", linewidth=1.5)
        ax.add_patch(circle)

        # Draw sides
        order = [A, B, C, D, A]
        for i in range(4):
            p, q = order[i], order[i + 1]
            ax.plot([p[0], q[0]], [p[1], q[1]],
                    color="#2c3e50", linewidth=2.0)

        # Draw diagonals lightly
        ax.plot([A[0], C[0]], [A[1], C[1]],
                color="#7f8c8d", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.plot([B[0], D[0]], [B[1], D[1]],
                color="#7f8c8d", linewidth=1.0, linestyle="--", alpha=0.7)

        # Vertex labels
        labels = ["A", "B", "C", "D"]
        for p, lbl in zip(pts, labels):
            # Push label outward from centre
            dx = p[0] - cx
            dy = p[1] - cy
            d_norm = math.hypot(dx, dy) + 1e-9
            lx = p[0] + (dx / d_norm) * 0.6
            ly = p[1] + (dy / d_norm) * 0.6
            ax.plot(p[0], p[1], "o", color="#c0392b", markersize=6)
            ax.text(lx, ly, lbl, fontsize=15, fontweight="bold",
                    ha="center", va="center", color="#2c3e50")

        # Side-length labels at midpoints
        side_keys = [("AB", A, B), ("BC", B, C), ("CD", C, D), ("DA", D, A)]
        for key, p, q in side_keys:
            mx = (p[0] + q[0]) / 2
            my = (p[1] + q[1]) / 2
            # Push outward slightly
            dx = mx - cx
            dy = my - cy
            d_norm = math.hypot(dx, dy) + 1e-9
            lx = mx + (dx / d_norm) * 0.35
            ly = my + (dy / d_norm) * 0.35
            ax.text(lx, ly, f"{key}={sides[key]}", fontsize=11,
                    color="#1565c0", fontweight="bold", ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              edgecolor="none", alpha=0.85))

        # Diagonal labels (when given)
        if qtype == "one_diagonal":
            mx = (A[0] + C[0]) / 2
            my = (A[1] + C[1]) / 2
            ax.text(mx, my + 0.15, f"AC={given_diag}", fontsize=10,
                    color="#27ae60", fontweight="bold", ha="center",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.85))

        margin = R + 1.6
        ax.set_xlim(cx - margin, cx + margin)
        ax.set_ylim(cy - margin, cy + margin)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Cyclic quadrilateral ABCD",
                     fontsize=12, fontweight="bold")

        return self.fig_to_pil(fig, dpi=110)


if __name__ == "__main__":
    env = PtolemyQuadQA()
    pass_count = 0
    total = 0
    for L in (0, 3, 6, 9):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok}; A={env._answer if ok else 'X'}")
            if ok:
                v = env.verify(f"\\boxed{{{env._answer}}}")
                v2 = env.verify(f"<answer>{env._answer}</answer>")
                v3 = env.verify("definitely_wrong_xyz")
                print(f"   boxed={v['accuracy']} answer={v2['accuracy']} wrong={v3['accuracy']}")
                if v['accuracy'] == 1 and v2['accuracy'] == 1 and v3['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
