"""
Mirror Text Inverse QA (D82, P3 — reference scientific figure).

Reference an external reference:
  "An inverse expression is placed near a mirror. The mirror shows its
   original appearance, but due to the ink dot on the mirror, part of
   the formula is obscured. Find the value of '?' in this expression."
  Ans: 132

This env shows two arithmetic expressions:
  1. The original (mirror-flipped) expression — drawn rotated 180° or
     horizontally mirrored, with one number occluded by a black ink dot.
  2. The mirror image of the expression (right-side-up, fully readable),
     with the same number replaced by '?'.

The task is to find the value of '?' by solving the equation. Because
the original (mirror-flipped) version still hides the unknown, the model
must perform the arithmetic, not just read.

Verifier: integer (`\\boxed{132}` or bare 132).

Difficulty:
  L0..L2 — single-operation A op B = C, find ?
  L3..L5 — two-operation A op B op C = D
  L6..L9 — multi-step + larger numbers
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


class MirrorTextInverseQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive: disable rotation augmentation
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "mirror_text_inverse"

    def _level_config(self, level: int) -> Dict:
        # 2026-05-04: added easier L0 mode (was 0% — VLM mental-rotation limit, attempt fix)
        # L0/L1 give the readable expression in question text (no need to read the
        # mirror image at all) + 1-digit numbers.
        """Operator-mix design note (avoids precedence ambiguity):

        L0-L4 use 2 terms — single binary op, no precedence issue, so '*'
        is allowed.

        L5+ use 3+ terms. With multiple operators the model defaults to
        standard precedence (* before +/-), but the env evaluates left-to-
        right (because the rendered equation is unparenthesized). Mixing
        '*' with '+' or '-' across 3+ terms produces ambiguity (e.g. for
        '5 + 3 * 4 = 32' the env's L-to-R answer is target=3 but std-
        precedence answer is target=27/4=6.75 — model would be graded
        wrong despite correct math). Therefore L5+ restricts to '+' / '-'
        only, where L-to-R == standard precedence and there is no
        ambiguity. Difficulty at higher levels comes from term count and
        magnitude rather than operator mix.
        """
        level = max(0, min(level, 9))
        if level <= 1:
            # L0/L1: 1-digit numbers + always '+' (easiest); we also leak the
            # readable expression in the question text below.
            return {"n_terms": 2, "max_val": 9, "ops": ["+"], "_easy_text": True}
        if level <= 2:
            return {"n_terms": 2, "max_val": 50, "ops": ["+", "-"]}
        if level <= 4:
            return {"n_terms": 2, "max_val": 99, "ops": ["+", "-", "*"]}
        if level <= 6:
            return {"n_terms": 3, "max_val": 50, "ops": ["+", "-"]}
        if level <= 8:
            return {"n_terms": 3, "max_val": 99, "ops": ["+", "-"]}
        return {"n_terms": 4, "max_val": 99, "ops": ["+", "-"]}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 6113 + level * 191 + 9)

        for _ in range(40):
            r = self._try_generate(rng, cfg)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg):
        n_terms = cfg["n_terms"]
        max_val = cfg["max_val"]
        ops = cfg["ops"]

        # Generate the expression: a op1 b op2 c ... = result
        nums = [rng.randint(2, max_val) for _ in range(n_terms)]
        op_seq = [rng.choice(ops) for _ in range(n_terms - 1)]

        # Build expression string (left-to-right evaluation)
        # We use *plain L-to-R* arithmetic (no operator precedence) so the
        # equation can be solved by the model without ambiguity.
        result = nums[0]
        for k, op in enumerate(op_seq):
            if op == "+":
                result = result + nums[k + 1]
            elif op == "-":
                result = result - nums[k + 1]
            elif op == "*":
                result = result * nums[k + 1]

        # Reject silly results
        if abs(result) > 99999 or result == 0:
            return None
        # Reject if any intermediate is negative — would distract
        # Recompute just to check intermediate sign
        running = nums[0]
        for k, op in enumerate(op_seq):
            if op == "+":
                running = running + nums[k + 1]
            elif op == "-":
                running = running - nums[k + 1]
            elif op == "*":
                running = running * nums[k + 1]
            if running < 0 and op != "-":
                return None

        # Choose which term to occlude (the "?"). Prefer occluding one of
        # the operands (not the result) so the model deduces it.
        target_idx = rng.randint(0, n_terms - 1)
        target_val = nums[target_idx]

        # Build the readable mirror form: "a + b - c = result", with target
        # replaced by '?'.
        parts = []
        for k, x in enumerate(nums):
            if k == target_idx:
                parts.append("?")
            else:
                parts.append(str(x))
            if k < len(op_seq):
                parts.append(op_seq[k])
        parts.extend(["=", str(result)])
        readable_expr = " ".join(parts)

        # The "original" (mirror-flipped) form just shows the same string
        # rendered upside-down (180° rotated). The model is asked to
        # identify the value of '?' so they can read either form.
        # Build the unflipped expression with all numbers intact (then the
        # ink dot covers `target_val` in BOTH renders, forcing inference).
        full_parts = []
        for k, x in enumerate(nums):
            full_parts.append(str(x))
            if k < len(op_seq):
                full_parts.append(op_seq[k])
        full_parts.extend(["=", str(result)])
        full_expr = " ".join(full_parts)

        # 2026-05-04 easy_text mode: leak the equation in the question text
        # so the model doesn't have to read the mirror image.
        if cfg.get("_easy_text"):
            question = (
                f"An arithmetic equation is shown in the image (placed near a "
                f"mirror; you may ignore the mirror form). The equation is: "
                f"{readable_expr}. Solve for '?'. Reply with the integer "
                f"inside <answer>...</answer>.\nExample: <answer>49</answer>"
            )
        else:
            question = (
                "The image shows an arithmetic expression placed near a mirror. "
                "Both the original (upside-down / mirror-flipped) form and the "
                "mirror's image (right-side-up) are visible. Due to an ink dot "
                "on the mirror, one of the numbers is replaced with '?'. Read "
                "the right-side-up expression in the upper box, then solve for '?' "
                "by inverse arithmetic. Reply with the integer inside "
                "<answer>...</answer>.\n"
                "Example: <answer>49</answer>"
            )
        ans_str = str(target_val)
        img = self._render(full_expr, readable_expr, target_idx, parts)
        return question, ans_str, img

    # ------------------------------------------------------------------ #
    def _render(self, full_expr, readable_expr, target_idx, readable_parts) -> Image.Image:
        fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.5), dpi=110)
        fig.patch.set_facecolor("#f5f5f5")
        ax.set_facecolor("#f5f5f5")

        # Mirror line in the middle
        ax.axhline(0, color="#7f8c8d", linewidth=2.0, linestyle="--")
        ax.text(0.02, 0.04, "mirror", fontsize=10, color="#7f8c8d",
                transform=ax.transAxes, va="bottom", ha="left",
                fontweight="bold")

        # Bottom half: original (mirror-flipped) expression — render
        # upside-down by rotating each character 180°.
        ax.text(0.5, -0.55, full_expr, fontsize=22, fontweight="bold",
                ha="center", va="center", rotation=180,
                transform=ax.transData,
                color="#2c3e50",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#fcf3cf",
                          edgecolor="#7f8c8d", linewidth=1.0))
        ax.text(0.02, 0.06, "(original, on the table)",
                fontsize=9, color="#5d6d7e",
                transform=ax.transAxes, va="bottom",
                ha="left", style="italic")

        # Top half: mirror image, right-side-up, with '?' for the occluded number.
        ax.text(0.5, 0.55, readable_expr, fontsize=22, fontweight="bold",
                ha="center", va="center", transform=ax.transData,
                color="#2c3e50",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#d6eaf8",
                          edgecolor="#7f8c8d", linewidth=1.0))
        ax.text(0.02, 0.96, "(reflection in the mirror)",
                fontsize=9, color="#5d6d7e",
                transform=ax.transAxes, va="top",
                ha="left", style="italic")

        # Black ink dot on the original to occlude `target_val`. We place
        # it roughly where the target appears in the upside-down line. For
        # simplicity, draw a circular ink dot near the centre-left/right of
        # the bottom line.
        # Approximate placement: distribute target across [-0.55, 0.55]
        n = len(readable_parts)
        # Each part is a character chunk. Compute the proportional position
        # of the target in the string.
        chunk_starts = []
        s_offset = 0
        for k, p in enumerate(readable_parts):
            chunk_starts.append(s_offset)
            s_offset += len(p) + 1  # +1 for space
        total_chars = s_offset
        if total_chars == 0:
            target_frac = 0.5
        else:
            # Note: in upside-down render, leftmost char appears on right
            target_frac = chunk_starts[target_idx] / total_chars
            # Flip horizontally because text is rotated 180°
            target_frac = 1.0 - target_frac
        ink_x = -0.55 + target_frac * 1.1
        ink = plt.Circle((ink_x, -0.55), 0.08, facecolor="#1c1c1c",
                         edgecolor="#000000", zorder=5)
        ax.add_patch(ink)

        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-1.0, 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
        return self.fig_to_pil(fig, dpi=110)


if __name__ == "__main__":
    env = MirrorTextInverseQA()
    pass_count = 0
    total = 0
    for L in (0, 3, 6, 9):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok}; A={env._answer if ok else 'X'}")
            if ok:
                v = env.verify(f"\\boxed{{{env._answer}}}")
                v2 = env.verify(f"<answer>{env._answer}</answer>")
                v3 = env.verify("definitely_wrong_xyz_no")
                print(f"   boxed={v['accuracy']} answer={v2['accuracy']} wrong={v3['accuracy']}")
                if v['accuracy'] == 1 and v2['accuracy'] == 1 and v3['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
