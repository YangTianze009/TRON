"""
Angle Bisector Chain Deep QA — pure single-letter MCQ A-D / A-E with
"No correct answer".

2026-05-05 R5 Batch 3: HARDENED rewrite. Was 97.5% saturated because the env
accepted free-form numeric answers with 0.5° tolerance and the question had
the angles spelled out in the figure. Now a pure MCQ with tight numerical
distractors based on common bisector/sum errors.

Per-level design:
  L0/L1 — 4-MCQ. Single bisector AD; given two angles → find ∠ADB.
  L2/L3 — 5-MCQ + 10% trap. Adds two-bisector incenter ∠AIB = 90 + ∠C/2.
  L4/L5 — 5-MCQ + 15% trap. Three-bisector ∠BIC / ∠AIC questions.
  L6/L7 — 5-MCQ + 25% trap. Hides one input (gives ∠A+∠B and ∠B).
  L8/L9 — 5-MCQ + 40% trap. Decimal half-angle inputs; tightest distractors
          (90 - C/2, 180 - target, off-by-bisector errors).
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


def _fmt_angle(v: float) -> str:
    """Format an angle: integer when round, else 1 decimal."""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.1f}"


class AngleBisectorChainDeepQA(StandaloneVisualEnv):
    ENV_NAME = "angle_bisector_chain_deep"
    ALLOW_ROTATION = False  # axis-up labels matter

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, int(level)))
        if level <= 1:
            return {
                "variants": ["single"],
                "hide_input": False,
                "decimal_inputs": False,
                "use_5_options": False,
                "trap_rate": 0.0,
                "tight_distractors": False,
            }
        if level <= 3:
            return {
                "variants": ["single", "incenter_ab"],
                "hide_input": False,
                "decimal_inputs": False,
                "use_5_options": True,
                "trap_rate": 0.10,
                "tight_distractors": False,
            }
        if level <= 5:
            return {
                "variants": ["incenter_ab", "incenter_full"],
                "hide_input": False,
                "decimal_inputs": False,
                "use_5_options": True,
                "trap_rate": 0.15,
                "tight_distractors": True,
            }
        if level <= 7:
            return {
                "variants": ["incenter_ab", "incenter_full"],
                "hide_input": True,
                "decimal_inputs": False,
                "use_5_options": True,
                "trap_rate": 0.25,
                "tight_distractors": True,
            }
        # L8/L9
        # 2026-05-05 Phase A: bump trap_rate to 0.5 (env still saturated 95%).
        return {
            "variants": ["incenter_full", "incenter_ab"],
            "hide_input": True,
            "decimal_inputs": True,
            "use_5_options": True,
            "trap_rate": 0.50,
            "tight_distractors": True,
        }

    # ------------------------------------------------------------------ #
    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 83)

        for _ in range(15):
            try:
                res = self._try_generate(rng, cfg, level)
                if res is not None:
                    return res
            except Exception:
                continue
        return None

    def _try_generate(self, rng, cfg, level):
        if cfg["decimal_inputs"]:
            ang_a = rng.randint(40, 100) + 0.5
            ang_b = rng.randint(30, int(180 - ang_a - 20)) + 0.5
            ang_c = 180 - ang_a - ang_b
        else:
            ang_a = float(rng.randint(40, 100))
            ang_b = float(rng.randint(30, int(180 - ang_a - 20)))
            ang_c = 180 - ang_a - ang_b
        if ang_c <= 0 or ang_c >= 180:
            return None

        variant = rng.choice(cfg["variants"])
        n_bisectors = {"single": 1, "incenter_ab": 2, "incenter_full": 3}[variant]

        if variant == "single":
            # AD bisects ∠A; D on BC. ∠ADB = 180 - ∠B - ∠A/2.
            target = 180 - ang_b - ang_a / 2
            ask = "∠ADB"
            stem_core = (
                f"In triangle ABC, ∠A = {_fmt_angle(ang_a)}° and ∠B = "
                f"{_fmt_angle(ang_b)}°. AD is the angle bisector of ∠A "
                f"(D on side BC). What is the measure of ∠ADB?"
            )
        elif variant == "incenter_ab":
            # ∠AIB = 90 + ∠C/2
            target = 90 + ang_c / 2
            ask = "∠AIB"
            if cfg["hide_input"]:
                given_sum = ang_a + ang_b
                stem_core = (
                    f"In triangle ABC, the sum ∠A + ∠B = "
                    f"{_fmt_angle(given_sum)}°, with ∠B = "
                    f"{_fmt_angle(ang_b)}°. The bisectors of ∠A and ∠B "
                    f"meet at the incenter I. What is the measure of ∠AIB?"
                )
            else:
                stem_core = (
                    f"In triangle ABC, ∠A = {_fmt_angle(ang_a)}° and ∠B = "
                    f"{_fmt_angle(ang_b)}°. The bisectors of ∠A and ∠B "
                    f"meet at the incenter I. What is the measure of ∠AIB?"
                )
        else:  # incenter_full — randomly ask AIC or BIC
            sub = rng.choice(["AIC", "BIC"])
            if sub == "AIC":
                target = 90 + ang_b / 2
            else:
                target = 90 + ang_a / 2
            ask = f"∠{sub}"
            if cfg["hide_input"]:
                given_sum = ang_a + ang_b
                stem_core = (
                    f"In triangle ABC, the sum ∠A + ∠B = "
                    f"{_fmt_angle(given_sum)}°, with ∠B = "
                    f"{_fmt_angle(ang_b)}°. The three angle bisectors of "
                    f"the triangle meet at the incenter I. What is the "
                    f"measure of {ask}?"
                )
            else:
                stem_core = (
                    f"In triangle ABC, ∠A = {_fmt_angle(ang_a)}° and ∠B = "
                    f"{_fmt_angle(ang_b)}°. The three angle bisectors of "
                    f"the triangle meet at the incenter I. What is the "
                    f"measure of {ask}?"
                )

        target_str = f"{_fmt_angle(target)}°"
        distractors = self._make_angle_distractors(rng, target, ang_a,
                                                   ang_b, ang_c, variant,
                                                   cfg["tight_distractors"])
        if len(distractors) < 4:
            return None

        return self._finalize_mcq(rng, cfg, target_str,
                                  distractors, stem_core,
                                  self._render(ang_a, ang_b, ang_c, n_bisectors))

    # ------------------------------------------------------------------ #
    def _make_angle_distractors(self, rng, target, A, B, C, variant, tight):
        cand = []
        # Common errors:
        # - 90 + A/2 instead of 90 + C/2 (mixed up angle)
        cand.append(90 + A / 2)
        cand.append(90 + B / 2)
        cand.append(90 + C / 2)
        # - 90 - X/2 (sign flip on the formula)
        cand.append(90 - A / 2)
        cand.append(90 - B / 2)
        cand.append(90 - C / 2)
        # - 180 - target (supplement)
        cand.append(180 - target)
        # - target itself with bisector error (used full angle instead of half)
        if variant == "single":
            cand.append(180 - B - A)         # forgot to halve
            cand.append(180 - A - B / 2)     # bisected the wrong angle
        cand.append(target + 1)
        cand.append(target - 1)
        cand.append(target + 5)
        cand.append(target - 5)
        # Discard non-positive / >180
        seen_vals = []
        for c in cand:
            if c <= 0 or c >= 180:
                continue
            # Keep only if numerically distinct from target
            if abs(c - target) < 0.5:
                continue
            seen_vals.append(c)
        # Convert to strings, dedup
        seen_strs = set()
        out = []
        for c in seen_vals:
            s = f"{_fmt_angle(c)}°"
            if s not in seen_strs:
                seen_strs.add(s); out.append(s)
        rng.shuffle(out)
        return out

    # ------------------------------------------------------------------ #
    def _finalize_mcq(self, rng, cfg, true_str, distractor_strs, stem, image):
        distractor_strs = [d for d in distractor_strs if d != true_str]
        seen = {true_str}; clean = []
        for d in distractor_strs:
            if d not in seen:
                seen.add(d); clean.append(d)
        distractor_strs = clean
        use_5 = cfg["use_5_options"]
        opt_letters = ["A", "B", "C", "D", "E"][:5 if use_5 else 4]
        n_value = len(opt_letters) - (1 if use_5 else 0)
        if len(distractor_strs) < n_value:
            return None
        use_trap = (rng.random() < cfg["trap_rate"]) if use_5 else False
        if use_trap:
            chosen = distractor_strs[:n_value]
            options = list(chosen); rng.shuffle(options)
            correct_letter = opt_letters[-1]
        else:
            chosen = distractor_strs[:n_value - 1]
            options = [true_str] + chosen
            rng.shuffle(options)
            correct_letter = opt_letters[options.index(true_str)]
        opt_lines = "\n".join(
            f"{opt_letters[i]}. {options[i]}" for i in range(n_value)
        )
        if use_5:
            opt_lines += f"\n{opt_letters[-1]}. No correct answer"
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"
        question = (
            f"{stem}\n\n{opt_lines}\n\n"
            f"Answer with a single letter ({letter_str})."
        )
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    def _render(self, ang_a, ang_b, ang_c, n_bisectors):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(-1, 6)
        ax.set_ylim(-1, 4)
        ax.set_aspect("equal")
        ax.axis("off")
        B = (0, 0); C = (5, 0)
        AB_len = 5 * math.sin(math.radians(ang_c)) / math.sin(math.radians(ang_a))
        A = (AB_len * math.cos(math.radians(ang_b)),
             AB_len * math.sin(math.radians(ang_b)))
        ax.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]],
                color="black", lw=2.0)
        ax.text(A[0], A[1] + 0.18, "A", fontsize=13, ha="center",
                fontweight="bold")
        ax.text(B[0] - 0.3, B[1] - 0.2, "B", fontsize=13, fontweight="bold")
        ax.text(C[0] + 0.18, C[1] - 0.2, "C", fontsize=13, fontweight="bold")

        def bisector_dir(P, Q1, Q2):
            v1 = (Q1[0] - P[0], Q1[1] - P[1])
            v2 = (Q2[0] - P[0], Q2[1] - P[1])
            n1 = math.hypot(*v1) or 1
            n2 = math.hypot(*v2) or 1
            d = (v1[0] / n1 + v2[0] / n2, v1[1] / n1 + v2[1] / n2)
            n = math.hypot(*d) or 1
            return (d[0] / n, d[1] / n)

        a_bis = bisector_dir(A, B, C)
        if abs(a_bis[1]) > 1e-9:
            s_a = -A[1] / a_bis[1]
            D = (A[0] + s_a * a_bis[0], 0)
        else:
            D = ((A[0] + B[0]) / 2, 0)
        if n_bisectors >= 1:
            ax.plot([A[0], D[0]], [A[1], D[1]], color="red", lw=1.6,
                    linestyle="--")
            if n_bisectors == 1:
                ax.text(D[0], D[1] - 0.2, "D", fontsize=12, fontweight="bold",
                        color="red")
        if n_bisectors >= 2:
            BC = math.hypot(B[0] - C[0], B[1] - C[1])
            CA = math.hypot(C[0] - A[0], C[1] - A[1])
            AB_l = math.hypot(A[0] - B[0], A[1] - B[1])
            tot = BC + CA + AB_l
            I = ((BC * A[0] + CA * B[0] + AB_l * C[0]) / tot,
                 (BC * A[1] + CA * B[1] + AB_l * C[1]) / tot)
            ax.plot([B[0], I[0]], [B[1], I[1]], color="blue", lw=1.4,
                    linestyle="--")
            if n_bisectors >= 3:
                ax.plot([C[0], I[0]], [C[1], I[1]], color="green", lw=1.4,
                        linestyle="--")
            ax.scatter(*I, s=70, color="purple", zorder=5)
            ax.text(I[0] + 0.1, I[1] + 0.1, "I", fontsize=13,
                    fontweight="bold", color="purple")
        return self.fig_to_pil(fig)


if __name__ == "__main__":
    env = AngleBisectorChainDeepQA()
    for level in (0, 3, 6, 9):
        for seed in (1, 7, 42):
            ok = env.generate(seed=seed, parameter={"level": level})
            print(f"L{level} s{seed} ok={ok} a={env._answer}")
