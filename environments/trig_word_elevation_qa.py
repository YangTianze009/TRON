"""
Trig Word — Elevation/Depression QA (M39/M40/M43/M46 merged, P1).

Single env covering four merged reference templates:
  TG-T2 elevation θ + distance → height
  TG-T3 depression from cliff → distance
  TG-T6 two-position triangulate height
  TG-T9 ladder-wall

Render a vertical pole / cliff / ladder with horizontal ground; observer at
distance d, angle of elevation/depression θ marked. Compute height (or other
side) using sin/cos/tan with special angles {30°, 45°, 60°} so answer is
clean.

2026-05-03 extension (M41 / TG-T4 incline + M44 / TG-T7 special angles):
added two new modes:
  - `incline_angle`: given rise and base of an incline, find the incline
    angle in degrees (uses inverse-tan on a special-angle ratio).
  - `special_angle_value`: directly compute sin/cos/tan of a special angle
    (30°, 45°, 60°) — answer is exact form (1/2, sqrt(2)/2, etc.).
"""
import math
import random
from io import BytesIO
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# Special angles with clean trig values
_SPECIAL_ANGLES = {
    30: ("30°", math.tan(math.radians(30))),
    45: ("45°", math.tan(math.radians(45))),
    60: ("60°", math.tan(math.radians(60))),
}


class TrigWordElevationQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    ENV_NAME = "trig_word_elevation"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {"modes": ["elevation_height", "incline_angle",
                              "special_angle_value"], "angles": [45]}
        if level <= 5:
            return {"modes": ["elevation_height", "ladder_wall",
                              "incline_angle", "special_angle_value"],
                    "angles": [30, 45, 60]}
        return {"modes": ["elevation_height", "ladder_wall", "depression_distance",
                          "incline_angle", "special_angle_value",
                          "two_position_triangulate"],
                "angles": [30, 45, 60]}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 5419 + level * 41 + 7)

        # 2026-05-03 (M41 / M44): incline_angle and special_angle_value modes.
        # Routed via separate generators when triggered.
        question_type = parameter.get("question_type")
        if question_type == "incline_angle":
            return self._gen_incline_angle(rng)
        if question_type == "special_angle_value":
            return self._gen_special_angle_value(rng)

        mode = rng.choice(cfg["modes"])
        if mode == "incline_angle":
            return self._gen_incline_angle(rng)
        if mode == "special_angle_value":
            return self._gen_special_angle_value(rng)

        angle = rng.choice(cfg["angles"])
        for _ in range(20):
            r = self._try_generate(rng, mode, angle)
            if r is not None:
                return r
        return None

    # ------------------------------------------------------------------ #
    # M41 / TG-T4 — incline angle
    # ------------------------------------------------------------------ #
    def _gen_incline_angle(self, rng):
        """Given rise and base, compute incline angle (special angle)."""
        # Pick a special angle and back-compute rise/base.
        angle = rng.choice([30, 45, 60])
        base = rng.choice([3, 6, 9, 12])
        if angle == 45:
            rise = base
        elif angle == 30:
            # tan(30) = 1/√3; rise = base/√3 — pick base as multiple of √3
            # so rise is integer. Use base = √3 * integer; we use base = 3
            # with rise = √3. To keep numeric clean, pick base = 3*k and ans
            # = base * tan30 (numeric), but ask for the angle in degrees.
            rise = round(base * math.tan(math.radians(30)), 2)
        else:  # 60
            rise = round(base * math.tan(math.radians(60)), 2)
        ans_str = str(angle)

        # Render right triangle as incline
        fig, ax = plt.subplots(figsize=(7, 4), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Triangle vertices: (0,0), (base, 0), (base, rise)
        ax.plot([0, base, base, 0], [0, 0, rise, 0],
                color="#222", linewidth=2)
        # Right-angle marker
        ra = min(base, rise) * 0.07
        ax.add_patch(plt.Rectangle((base - ra, 0), ra, ra, fill=False,
                                    edgecolor="#222", linewidth=1))
        ax.annotate(f"base = {base}", (base / 2, -0.4),
                    fontsize=11, ha="center", color="#444")
        ax.annotate(f"rise = {rise}", (base + 0.4, rise / 2),
                    fontsize=11, color="#444", rotation=90)
        ax.annotate("?", (0.4, 0.3), fontsize=18, color="red",
                    fontweight="bold")
        # Incline line label
        ax.set_xlim(-1, base + 2)
        ax.set_ylim(-1, rise + 2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Incline angle problem", fontsize=12, fontweight="bold")
        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()

        q = (f"As shown in the figure, an incline has a horizontal base of "
             f"{base} and a vertical rise of {rise}. What is the incline "
             f"angle (in degrees)? Place an integer in <answer>...</answer>.")
        return q, ans_str, img

    # ------------------------------------------------------------------ #
    # M44 / TG-T7 — special angle exact value
    # ------------------------------------------------------------------ #
    def _gen_special_angle_value(self, rng):
        """Compute sin/cos/tan of a special angle (30°, 45°, 60°).
        Answer: exact fraction or radical form (e.g. 1/2, sqrt(2)/2).
        """
        angle = rng.choice([30, 45, 60])
        ratio = rng.choice(["sin", "cos", "tan"])
        # Exact values:
        # sin30 = 1/2, sin45 = sqrt(2)/2, sin60 = sqrt(3)/2
        # cos30 = sqrt(3)/2, cos45 = sqrt(2)/2, cos60 = 1/2
        # tan30 = sqrt(3)/3, tan45 = 1, tan60 = sqrt(3)
        table = {
            (30, "sin"): "1/2",
            (30, "cos"): "sqrt(3)/2",
            (30, "tan"): "sqrt(3)/3",
            (45, "sin"): "sqrt(2)/2",
            (45, "cos"): "sqrt(2)/2",
            (45, "tan"): "1",
            (60, "sin"): "sqrt(3)/2",
            (60, "cos"): "1/2",
            (60, "tan"): "sqrt(3)",
        }
        ans_str = table[(angle, ratio)]

        # Render: triangle with the angle marked
        fig, ax = plt.subplots(figsize=(6, 5), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Draw a 30-60-90 or 45-45-90 triangle
        if angle in (30, 60):
            # 30-60-90 reference triangle: legs 1, sqrt(3), hypotenuse 2
            A = (0, 0); B = (math.sqrt(3), 0); C = (math.sqrt(3), 1)
            # angle 30 at A, angle 60 at B (well, depends on orientation)
        else:
            # 45-45-90: legs 1, 1, hyp sqrt(2)
            A = (0, 0); B = (1, 0); C = (1, 1)
        ax.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]],
                color="#222", linewidth=2)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 2.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"Reference triangle (special angle: {angle}°)",
                     fontsize=12, fontweight="bold")
        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()

        q = (f"Compute the exact value of {ratio}({angle}°). Give your "
             f"answer in exact form (use sqrt(N) for square roots, e.g. "
             f"'sqrt(3)/2', '1/2', 'sqrt(3)'). Place the answer in "
             f"<answer>...</answer>.")
        return q, ans_str, img

    def _try_generate(self, rng, mode, angle):
        ang_str, tan_v = _SPECIAL_ANGLES[angle]

        if mode == "elevation_height":
            # Observer at distance d from base of pole. Angle of elevation = angle.
            # Height h = d * tan(angle).
            # Pick d so h is clean integer or √3 multiple
            # For 45°: h = d
            # For 30°: h = d * (√3/3) — pick d as multiple of 3 giving clean √3 form
            # For 60°: h = d * √3 — pick d as integer giving √3 multiple
            if angle == 45:
                d = rng.randint(3, 12)
                h_val = d
                ans = str(h_val)
            elif angle == 30:
                d_choices = [3, 6, 9, 12]
                d = rng.choice(d_choices)
                # h = d * sqrt(3) / 3
                h_str_int = d // 3
                if h_str_int == 1:
                    ans = "√3"
                else:
                    ans = f"{h_str_int}√3"
                h_val = d * math.sqrt(3) / 3
            else:  # 60
                d_choices = [2, 3, 4, 5, 6]
                d = rng.choice(d_choices)
                ans = f"{d}√3" if d > 1 else "√3"
                h_val = d * math.sqrt(3)
            question = (
                f"A pole stands vertically on level ground. From a point on "
                f"the ground {d} meters away from the base of the pole, the "
                f"angle of elevation to the top is {ang_str}. Find the height "
                f"of the pole in meters. (Use exact form, e.g. √3 if needed.)"
            )
            img = self._render_pole(d, h_val, angle)
            return question, ans, img

        if mode == "depression_distance":
            # Observer on top of cliff height h, looks down at boat with angle of depression theta.
            # Distance d = h / tan(theta).
            if angle == 45:
                h = rng.randint(3, 12)
                ans = str(h)
                d_val = h
            elif angle == 30:
                h_choices = [2, 3, 4, 5]
                h = rng.choice(h_choices)
                ans = f"{h}√3" if h > 1 else "√3"
                d_val = h * math.sqrt(3)
            else:  # 60
                h_choices = [3, 6, 9, 12]
                h = rng.choice(h_choices)
                d_int = h // 3
                if d_int == 1:
                    ans = "√3"
                else:
                    ans = f"{d_int}√3"
                d_val = h / math.sqrt(3)
            question = (
                f"From the top of a cliff that is {h} meters tall, the angle "
                f"of depression to a boat at sea level is {ang_str}. Find the "
                f"horizontal distance from the boat to the foot of the cliff "
                f"in meters. (Use exact form, e.g. √3 if needed.)"
            )
            img = self._render_pole(d_val, h, angle, is_cliff=True)
            return question, ans, img

        if mode == "ladder_wall":
            # Ladder of length L leans against vertical wall; angle between
            # ladder and ground = θ. Height up the wall = L * sin(θ).
            if angle == 45:
                L = rng.choice([4, 6, 8, 10])
                # h = L * √2/2 — pick L with multiple of 2: actually L*√2/2
                # cleaner: report as fraction. Use L = 2,4,6 for clean √2 multiples
                L_int = L // 2
                if L_int == 1:
                    ans = "√2"
                else:
                    ans = f"{L_int}√2"
                h_val = L * math.sqrt(2) / 2
            elif angle == 30:
                L_choices = [2, 4, 6, 8]
                L = rng.choice(L_choices)
                # h = L/2
                ans = str(L // 2)
                h_val = L / 2
            else:  # 60
                L_choices = [2, 4, 6, 8]
                L = rng.choice(L_choices)
                # h = L * √3/2
                L_int = L // 2
                if L_int == 1:
                    ans = "√3"
                else:
                    ans = f"{L_int}√3"
                h_val = L * math.sqrt(3) / 2
            d_val = L * math.cos(math.radians(angle))
            question = (
                f"A ladder of length {L} meters leans against a vertical wall, "
                f"making an angle of {ang_str} with the ground. Find the "
                f"height (in meters) up the wall that the top of the ladder "
                f"reaches. (Use exact form, e.g. √2 or √3 if needed.)"
            )
            img = self._render_pole(d_val, h_val, angle, ladder_len=L)
            return question, ans, img

        if mode == "two_position_triangulate":
            # Observer measures from two points along same line:
            #   far point A at distance dA → angle α
            #   nearer point B at distance dB (dA-AB) → angle β  (β > α)
            # Building height h = AB / (cot α - cot β)
            # Use clean special-angle pairs: (α=30, β=45), (α=30, β=60), (α=45, β=60).
            pairs = [(30, 45), (30, 60), (45, 60)]
            a, b = rng.choice(pairs)
            # Pick AB such that h is integer or clean √
            cot_a = 1 / math.tan(math.radians(a))
            cot_b = 1 / math.tan(math.radians(b))
            denom = cot_a - cot_b  # always > 0
            if (a, b) == (30, 45):
                # denom = √3 - 1; clean h: pick AB = (√3+1)*k → h = ((√3+1)*k) / (√3-1) = ((√3+1)²*k)/2 = (4+2√3)*k/2 = (2+√3)*k. Hmm not clean.
                # Pick AB = m where h = m / (√3-1) ; multiply numerator and denominator by (√3+1):
                # h = m * (√3+1) / 2  → want this clean: m even, m=2 gives h = √3+1. m=4 → h = 2(√3+1) = 2√3+2.
                m = rng.choice([2, 4, 6])
                h_str = f"{m // 2}(√3 + 1)" if m // 2 != 1 else "(√3 + 1)"
                AB = m
                h_val = m * (math.sqrt(3) + 1) / 2
            elif (a, b) == (30, 60):
                # cot30 - cot60 = √3 - √3/3 = (3√3 - √3)/3 = 2√3/3
                # h = AB * 3/(2√3) = AB * √3/2 (clean if AB is even)
                m = rng.choice([2, 4, 6, 8])
                h_str = f"{m // 2}√3" if m // 2 != 1 else "√3"
                AB = m
                h_val = m * math.sqrt(3) / 2
            else:  # (45, 60)
                # cot45 - cot60 = 1 - √3/3 = (3-√3)/3
                # h = AB * 3/(3-√3) = AB * 3*(3+√3)/((3-√3)(3+√3)) = AB * 3*(3+√3)/6 = AB*(3+√3)/2
                m = rng.choice([2, 4, 6])
                h_str = f"{m // 2}(3 + √3)" if m // 2 != 1 else "(3 + √3)"
                AB = m
                h_val = m * (3 + math.sqrt(3)) / 2
            ans = h_str
            question = (
                f"A vertical building stands on level ground. From two points A "
                f"and B on the ground (with B closer to the building, both on the "
                f"same line as the foot of the building, AB = {AB} m), the angles "
                f"of elevation to the top of the building are {a}° at A and {b}° "
                f"at B. Find the height of the building in meters. (Use exact "
                f"form, e.g. √3 or (3+√3) if needed.)"
            )
            img = self._render_two_position(AB, a, b, h_val)
            return question, ans, img

        return None

    # ------------------------------------------------------------------ #
    def _render_two_position(self, AB, alpha, beta, h) -> Image.Image:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Coords: building at x = total, height h. A at x=0, B at x=AB.
        cot_a = 1 / math.tan(math.radians(alpha))
        total = h * cot_a
        # Ground line
        ax.plot([0, total + 1], [0, 0], color="#27ae60", linewidth=2)
        # Building
        ax.plot([total, total], [0, h], color="#34495e", linewidth=3)
        ax.plot([total - 0.3, total + 0.3], [h, h], color="#34495e", linewidth=2)
        # Sight lines
        ax.plot([0, total], [0, h], color="#c0392b", linestyle="--", linewidth=1.4)
        ax.plot([AB, total], [0, h], color="#1f77b4", linestyle="--", linewidth=1.4)
        # Position markers
        ax.plot([0], [0], 'o', color="#c0392b", markersize=8)
        ax.plot([AB], [0], 'o', color="#1f77b4", markersize=8)
        ax.text(0, -0.4, "A", fontsize=12, ha="center", fontweight="bold",
                color="#c0392b")
        ax.text(AB, -0.4, "B", fontsize=12, ha="center", fontweight="bold",
                color="#1f77b4")
        # Angle annotations
        ax.text(0.4, 0.15, f"{alpha}°", fontsize=11, color="#c0392b",
                fontweight="bold")
        ax.text(AB + 0.4, 0.15, f"{beta}°", fontsize=11, color="#1f77b4",
                fontweight="bold")
        # AB distance label
        ax.annotate("", xy=(AB, -0.7), xytext=(0, -0.7),
                    arrowprops=dict(arrowstyle="<->", color="#444", lw=1.2))
        ax.text(AB / 2, -0.95, f"AB = {AB} m", ha="center", fontsize=10,
                color="#444")
        ax.text(total + 0.15, h / 2, "h = ?", fontsize=11, color="#7b241c",
                fontweight="bold")

        ax.set_xlim(-1, total + 2)
        ax.set_ylim(-1.4, h + 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Two-position triangulate height", fontsize=12,
                     fontweight="bold")

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    # ------------------------------------------------------------------ #
    def _render_pole(self, d, h, angle, is_cliff=False, ladder_len=None) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        # Ground line
        ax.plot([0, d + 2], [0, 0], color="#27ae60", linewidth=2)
        # Pole / wall
        ax.plot([d, d], [0, h], color="#2c3e50", linewidth=3)
        # Hypotenuse from observer to top
        ax.plot([0, d], [0, h], color="#d62728", linewidth=2)

        if ladder_len:
            ax.text(d / 2 - 0.3, h / 2 + 0.3, f"L={ladder_len}",
                    fontsize=11, color="#d62728", fontweight="bold")

        # Right-angle marker at base
        ra = min(0.3, h * 0.1, d * 0.1)
        ax.plot([d - ra, d - ra, d], [0, ra, ra], color="#222", linewidth=1)

        # Angle arc at observer
        from matplotlib.patches import Arc
        arc_r = min(d * 0.25, 0.6)
        arc = Arc((0, 0), arc_r * 2, arc_r * 2, angle=0,
                  theta1=0, theta2=angle, color="#1f77b4", linewidth=2)
        ax.add_patch(arc)
        ax.text(arc_r * 1.2 * math.cos(math.radians(angle / 2)),
                arc_r * 1.2 * math.sin(math.radians(angle / 2)) + 0.05,
                f"{angle}°", fontsize=12, color="#1f77b4",
                fontweight="bold")

        # Distance label
        ax.text(d / 2, -0.4, f"{d:g} m", ha="center", va="top", fontsize=11)

        # Height marker
        ax.text(d + 0.2, h / 2, f"?", fontsize=14, color="#d62728",
                fontweight="bold")

        ax.set_xlim(-0.5, d + 2)
        ax.set_ylim(-1, max(h + 1, 4))
        ax.set_aspect("equal")
        ax.axis("off")
        return self.fig_to_pil(fig, dpi=110)


if __name__ == "__main__":
    env = TrigWordElevationQA()
    pass_count = 0
    total = 0
    for L in (0, 3, 6):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok} A={env._answer if ok else 'X'}")
            if ok:
                v = env.verify(f"\\boxed{{{env._answer}}}")
                v2 = env.verify(f"<answer>{env._answer}</answer>")
                v3 = env.verify("definitely_wrong_xyz")
                print(f"   boxed={v['accuracy']} answer={v2['accuracy']} wrong={v3['accuracy']}")
                if v['accuracy'] == 1 and v2['accuracy'] == 1 and v3['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
