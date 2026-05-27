"""
Triangle Property Chain QA environment (batch 2 Part B, 2026-04-14).

Goal: train multi-step angle / side chasing in a triangle. Covers
triangle-sum, isosceles base-angles, exterior-angle, and similar-triangle
ratio theorems. Target sub-categories:

Difficulty axes:
  A) Pattern E (chain depth) — 1 theorem application at L0 up to 4-5 at L9.
  B) Pattern G (label hiding) — visible label fraction shrinks with level.
  C) Pattern H (parameter range) — integer angles at L0, decimal at L≥5.

Format: 4-way MCQ (letter), constant at every level.

L0: isosceles triangle with one base angle labelled, find the other.
L9: 4-hop chain across isosceles + exterior-angle + similar-triangle
    ratio with half the labels stripped and one decimal.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

class TrianglePropertyChainQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "triangle_property_chain"
    TEXTBOOK_POSTPROCESS = True  # v4 B1: textbook-scan filter (~30% of rollouts)

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            # Pattern E: chain depth
            "chain_depth":          1 + level // 2,          # 1..5
            # Pattern G: visible label fraction (for distractor hiding)
            "visible_label_frac":   max(0.30, 1.0 - 0.08 * level),  # 1.0..0.28
            # Pattern H: decimals at high level
            "use_decimals":         level >= 5,
            # Pattern B: distractor tightness
            "tight_distractors":    level >= 4,
            # Theorem pool width
            "theorem_pool_size":    min(6, 2 + level // 2),  # 2..6
            # Red-herring extras
            "n_red_herring_labels": level // 3,              # 0..3
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)

        # Sub-RNG mixes seed and level.
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["chain_depth"]

        theorem_pool = [
            "triangle_sum", "isosceles", "exterior_angle",
            "similar_ratio", "right_triangle", "angle_bisector",
        ][: cfg["theorem_pool_size"]]

        for _ in range(40):
            theorem = rng.choice(theorem_pool)
            result = self._try_generate(rng, level, cfg, theorem)
            if result is not None:
                return result
        return None

    # -------------------------------------------------- #
    # Problem templates
    # -------------------------------------------------- #
    def _try_generate(self, rng: random.Random, level: int, cfg: Dict,
                      theorem: str) -> Optional[Tuple[str, str, Image.Image]]:
        use_dec = cfg["use_decimals"]
        tight = cfg["tight_distractors"]
        chain = cfg["chain_depth"]

        # Build an integer-angle triangle as the base figure.
        # Base triangle picks three angles summing to 180.
        a1 = rng.randint(30, 80)
        a2 = rng.randint(30, min(100, 175 - a1))
        a3 = 180 - a1 - a2
        if a3 < 15 or a3 > 140:
            return None
        base_angles = [a1, a2, a3]
        rng.shuffle(base_angles)

        # Build BOTH `given_text` (put on image — OK to contain numbers) and
        # `question_stem` (for the text-only question — MUST NOT contain
        # numeric values; use "as shown" phrasing).
        abc = ['A', 'B', 'C']
        # Pick an unknown to ask for based on theorem + chain.
        if theorem == "triangle_sum":
            ask_idx = rng.randint(0, 2)
            given_idx = [i for i in range(3) if i != ask_idx]
            answer_val = base_angles[ask_idx]
            given_text = (
                f"In \u25b3ABC, \u2220{abc[given_idx[0]]} = {base_angles[given_idx[0]]}\u00b0 "
                f"and \u2220{abc[given_idx[1]]} = {base_angles[given_idx[1]]}\u00b0."
            )
            question_stem = (
                f"In \u25b3ABC, angles \u2220{abc[given_idx[0]]} and "
                f"\u2220{abc[given_idx[1]]} are as shown."
            )
            ask_text = f"Find \u2220{abc[ask_idx]}."
            if chain >= 2:
                # BUGFIX 2026-04-24: removed arbitrary 'extra' subtraction. The
                # perpendicular does not change the full angle B; previous code
                # subtracted a random 5-20 degrees making the GT wrong while the
                # image still labeled the true angle.
                given_text += f" A perpendicular from vertex {abc[ask_idx]} meets the opposite side."
                question_stem += (f" A perpendicular from vertex {abc[ask_idx]} "
                                   "meets the opposite side.")
        elif theorem == "isosceles":
            base_ang = rng.randint(30, 75)
            apex = 180 - 2 * base_ang
            if apex < 15:
                return None
            config_angles = [apex, base_ang, base_ang]
            mode = rng.choice(["apex_to_base", "base_to_apex"])
            if mode == "apex_to_base":
                given_text = f"In \u25b3ABC, AB = AC and \u2220A = {apex}\u00b0."
                question_stem = ("In \u25b3ABC, AB = AC and \u2220A is as "
                                  "shown in the figure.")
                ask_text = "Find \u2220B."
                answer_val = base_ang
            else:
                given_text = f"In \u25b3ABC, AB = AC and \u2220B = {base_ang}\u00b0."
                question_stem = ("In \u25b3ABC, AB = AC and \u2220B is as "
                                  "shown in the figure.")
                ask_text = "Find \u2220A."
                answer_val = apex
            base_angles = config_angles
            if chain >= 3:
                shift = rng.choice([-2, -1, 1, 2])
                answer_val = max(5, answer_val + shift * 0)
        elif theorem == "exterior_angle":
            ia = rng.randint(30, 80)
            ib = rng.randint(30, 80)
            ext = ia + ib
            if ext >= 175 or ext <= 30:
                return None
            ic = 180 - ia - ib
            if ic < 10:
                return None
            given_text = (
                f"In \u25b3ABC, \u2220A = {ia}\u00b0 and \u2220B = {ib}\u00b0. "
                f"Side BC is extended to D."
            )
            question_stem = ("In \u25b3ABC, \u2220A and \u2220B are as "
                              "shown. Side BC is extended to D.")
            ask_text = "Find the exterior angle \u2220ACD at vertex C."
            answer_val = ext
            base_angles = [ia, ib, ic]
        elif theorem == "similar_ratio":
            k = rng.randint(2, 4)
            ab = rng.randint(4, 12)
            bc = rng.randint(4, 12)
            de = ab * k
            ef = bc * k
            given_text = (
                f"\u25b3ABC ~ \u25b3DEF with AB = {ab}, BC = {bc}, DE = {de}."
            )
            question_stem = ("\u25b3ABC ~ \u25b3DEF with AB, BC, DE as "
                              "labeled in the figure.")
            ask_text = "Find EF."
            answer_val = ef
            base_angles = [60, 60, 60]
        elif theorem == "right_triangle":
            acute = rng.randint(20, 70)
            given_text = (
                f"\u25b3ABC is right-angled at C with \u2220B = {acute}\u00b0."
            )
            question_stem = ("\u25b3ABC is right-angled at C and \u2220B is "
                              "as shown.")
            ask_text = "Find \u2220A."
            answer_val = 90 - acute
            base_angles = [90 - acute, acute, 90]
            if chain >= 2:
                given_text += " An altitude is drawn from C to AB."
                question_stem += " An altitude is drawn from C to AB."
        elif theorem == "angle_bisector":
            base_ang = rng.randint(40, 80)
            apex = 180 - 2 * base_ang
            if apex < 15:
                return None
            given_text = (
                f"In \u25b3ABC, AB = AC and \u2220A = {apex}\u00b0. "
                f"The bisector of \u2220A meets BC at M."
            )
            question_stem = ("In \u25b3ABC, AB = AC and \u2220A is as "
                              "shown. The bisector of \u2220A meets BC at M.")
            ask_text = "Find \u2220BAM."
            answer_val = apex // 2
            base_angles = [apex, base_ang, base_ang]
        else:
            return None

        if answer_val <= 0:
            return None

        if use_dec:
            # Add a half-degree perturbation but keep the integer candidate valid.
            if rng.random() < 0.5 and theorem not in ("similar_ratio",):
                answer_val = round(answer_val + rng.choice([-0.5, 0.5]), 1)

        # Build distractors.
        distractors: List = []
        gt = answer_val
        if theorem == "similar_ratio":
            pool = {gt // 2, gt - 1, gt + 1, gt * 2}
            if tight:
                pool = {max(1, gt - 2), max(1, gt - 1), gt + 1, gt + 2}
        else:
            if tight:
                pool = {max(1, gt - 2), max(1, gt - 1), gt + 1, gt + 2}
            else:
                pool = {max(1, gt - 10), max(1, gt - 5), gt + 5, gt + 10, 180 - gt,
                        max(1, 2 * gt - 180)}
        pool.discard(gt)
        distractors_list = [p for p in pool if p != gt and p > 0]
        rng.shuffle(distractors_list)
        distractors = distractors_list[:3]
        if len(distractors) < 3:
            # Top up from a wide pool
            filler = [gt + k for k in (-15, -12, -8, 8, 12, 15) if gt + k > 0 and (gt + k) != gt]
            rng.shuffle(filler)
            for f in filler:
                if f not in distractors and f != gt:
                    distractors.append(f)
                if len(distractors) >= 3:
                    break
        if len(distractors) < 3:
            return None

        options = [gt] + distractors[:3]
        rng.shuffle(options)
        if options.count(gt) > 1:
            return None
        answer_letter = chr(ord("A") + options.index(gt))

        # Format options as strings
        unit = "" if theorem == "similar_ratio" else "\u00b0"
        def _fmt(v):
            return f"{v}{unit}" if isinstance(v, int) else f"{v:.1f}{unit}"

        options_str = [_fmt(o) for o in options]

        # IMPORTANT: text question uses `question_stem` (no numeric leakage);
        # the image panel shows full given_text with numbers.
        question = (
            f"{question_stem} {ask_text}\n"
            f"Options:\n"
            + "\n".join(f"  ({chr(ord('A') + i)}) {options_str[i]}" for i in range(4))
            + "\nAnswer with the single letter of the correct option."
        )

        image = self._render(base_angles, given_text, ask_text, options_str,
                             cfg, theorem)
        return question, answer_letter, image

    # -------------------------------------------------- #
    # Renderer
    # -------------------------------------------------- #
    def _render(self, base_angles, given_text, ask_text, options, cfg,
                theorem: str) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        ff = style["font_family"]
        fs = style["font_size_base"]

        fig = plt.figure(figsize=(8.5 * sc, 6.0 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax_tri = fig.add_subplot(1, 2, 1)
        ax_txt = fig.add_subplot(1, 2, 2)
        ax_tri.set_aspect("equal")
        ax_tri.axis("off")
        ax_txt.axis("off")

        # Draw triangle based on base_angles (in degrees)
        a_a, a_b, a_c = base_angles
        # Place B at origin, C on x-axis.
        side_c = 6.0
        Bx, By = 0.0, 0.0
        Cx, Cy = side_c, 0.0
        # A is at angle a_b from B side.
        ang_b_rad = math.radians(a_b)
        # Use law of sines: a_side opposite A; use unit scaling.
        a_side = side_c * math.sin(math.radians(a_a)) / max(1e-6, math.sin(math.radians(a_c)))
        Ax = a_side * math.cos(ang_b_rad)
        Ay = a_side * math.sin(ang_b_rad)

        verts = [(Ax, Ay), (Bx, By), (Cx, Cy)]
        tri = plt.Polygon(verts, closed=True, facecolor=style["palette"][0],
                          edgecolor=style["geo_line_color"],
                          linewidth=style["line_width"], alpha=0.28)
        ax_tri.add_patch(tri)

        # For similar_ratio theorems, also draw DEF (a scaled similar triangle)
        # so the figure actually shows both triangles referenced in the text.
        self._def_bounds = None
        if theorem == "similar_ratio":
            scale_def = 1.5
            width_abc = max(Ax, Bx, Cx) - min(Ax, Bx, Cx)
            height_abc = max(Ay, By, Cy) - min(Ay, By, Cy)
            offset_x = width_abc + 2.0
            offset_y = 0.0
            def_verts = [(Ax * scale_def + offset_x, Ay * scale_def + offset_y),
                         (Bx * scale_def + offset_x, By * scale_def + offset_y),
                         (Cx * scale_def + offset_x, Cy * scale_def + offset_y)]
            tri_def = plt.Polygon(def_verts, closed=True,
                                  facecolor=style["palette"][3 % len(style["palette"])],
                                  edgecolor=style["geo_line_color"],
                                  linewidth=style["line_width"], alpha=0.28)
            ax_tri.add_patch(tri_def)
            for lbl, (vx, vy) in zip(["D", "E", "F"], def_verts):
                cxm = sum(p[0] for p in def_verts) / 3
                cym = sum(p[1] for p in def_verts) / 3
                ox = vx - cxm
                oy = vy - cym
                n = math.hypot(ox, oy) + 1e-6
                ax_tri.text(vx + 0.5 * ox / n, vy + 0.5 * oy / n, lbl,
                            fontsize=fs + 3, fontweight="bold", family=ff,
                            ha="center", va="center",
                            color=style["geo_line_color"])
            self._def_bounds = ([v[0] for v in def_verts],
                                [v[1] for v in def_verts])

        for lbl, (vx, vy) in zip(["A", "B", "C"], verts):
            ox = vx - (Ax + Bx + Cx) / 3
            oy = vy - (Ay + By + Cy) / 3
            n = math.hypot(ox, oy) + 1e-6
            ax_tri.text(vx + 0.5 * ox / n, vy + 0.5 * oy / n, lbl,
                        fontsize=fs + 3, fontweight="bold", family=ff,
                        ha="center", va="center",
                        color=style["geo_line_color"])

        # Show angle arcs on the triangle only for the labels that remain visible.
        vis_frac = cfg["visible_label_frac"]
        n_vis = max(1, int(round(vis_frac * 3)))
        vis_idxs = list(range(3))
        if n_vis < 3:
            self._rng.shuffle(vis_idxs)
            vis_idxs = vis_idxs[:n_vis]
        angle_colors = [style["palette"][k] for k in (1, 2, 3)]
        for idx in vis_idxs:
            vx, vy = verts[idx]
            prev_idx = (idx + 1) % 3
            next_idx = (idx + 2) % 3
            dx1 = verts[prev_idx][0] - vx
            dy1 = verts[prev_idx][1] - vy
            dx2 = verts[next_idx][0] - vx
            dy2 = verts[next_idx][1] - vy
            th1 = math.degrees(math.atan2(dy1, dx1))
            th2 = math.degrees(math.atan2(dy2, dx2))
            lo = min(th1, th2)
            hi = max(th1, th2)
            if hi - lo > 180:
                lo, hi = hi, lo + 360
            arc = patches.Arc((vx, vy), 1.2, 1.2, angle=0,
                              theta1=lo, theta2=hi,
                              color=angle_colors[idx],
                              linewidth=1.2)
            ax_tri.add_patch(arc)
            mid_rad = math.radians((lo + hi) / 2)
            lbl = f"{base_angles[idx]}\u00b0"
            ax_tri.text(vx + 1.0 * math.cos(mid_rad),
                        vy + 1.0 * math.sin(mid_rad),
                        lbl, fontsize=fs, family=ff,
                        ha="center", va="center",
                        color=angle_colors[idx])

        # Mark question spot
        tri_title_pool = ["\u25b3ABC", "Triangle ABC", "Figure",
                          "Triangle", "Geometry"]
        ax_tri.set_title(self._rng.choice(tri_title_pool),
                         fontsize=fs + 2, family=ff, pad=8)
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        if self._def_bounds is not None:
            xs = xs + self._def_bounds[0]
            ys = ys + self._def_bounds[1]
        ax_tri.set_xlim(min(xs) - 1.5, max(xs) + 1.5)
        ax_tri.set_ylim(min(ys) - 1.5, max(ys) + 1.5)

        # Text panel: given + ask + options
        ax_txt.set_xlim(0, 10)
        ax_txt.set_ylim(0, 12)
        ax_txt.text(0.3, 11.5, "Given:", fontsize=fs + 1, fontweight="bold",
                    family=ff, ha="left", va="top", color="#2c3e50")
        # Wrap text
        wrapped = self._wrap(given_text, width=42)
        y = 10.8
        for ln in wrapped:
            ax_txt.text(0.3, y, ln, fontsize=fs, family=ff,
                        ha="left", va="top", color="#1a1a1a")
            y -= 0.55
        y -= 0.3
        ax_txt.text(0.3, y, "Ask:", fontsize=fs + 1, fontweight="bold",
                    family=ff, ha="left", va="top", color="#2c3e50")
        y -= 0.55
        for ln in self._wrap(ask_text, width=42):
            ax_txt.text(0.3, y, ln, fontsize=fs, family=ff,
                        ha="left", va="top", color="#1a1a1a")
            y -= 0.55
        y -= 0.3
        ax_txt.text(0.3, y, "Options:", fontsize=fs + 1, fontweight="bold",
                    family=ff, ha="left", va="top", color="#2c3e50")
        y -= 0.55
        for i, o in enumerate(options):
            ax_txt.text(0.5, y, f"({chr(ord('A') + i)}) {o}",
                        fontsize=fs, family=ff,
                        ha="left", va="top", color="#1a1a1a")
            y -= 0.55

        fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02,
                            wspace=0.12)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    @staticmethod
    def _wrap(text: str, width: int = 40) -> List[str]:
        out: List[str] = []
        cur = ""
        for word in text.split():
            if len(cur) + len(word) + 1 > width:
                out.append(cur)
                cur = word
            else:
                cur = (cur + " " + word).strip()
        if cur:
            out.append(cur)
        return out

# ---------------------------------------------------------------------- #
# Local smoke test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_b2b"
    os.makedirs(out_dir, exist_ok=True)
    env = TrianglePropertyChainQA()
    for level in (0, 3, 6):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[seed={s} L{level}] FAILED")
                continue
            img = env.render()
            q = env.get_instruction()
            a = env._answer
            path = os.path.join(out_dir,
                                f"triangle_property_chain_s{s}_L{level}.png")
            img.save(path)
            print(f"[seed={s} L{level}] saved {path}")
            print(f"  Q (first 100): {q[:100]}")
            print(f"  A: {a}")
