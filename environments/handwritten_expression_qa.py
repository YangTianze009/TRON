"""Handwritten Expression QA — diversity + difficulty redesign 2026-04-16.

Round-2 fix: too-easy at most levels → introduce different equation FAMILIES
and question operations per level (not just coefficient scaling). Added
diverse noise, paper textures, ink colours, fonts.

Level map (structural):
  L0: simplest two-step equation ax + b = c → solve x (formula shown).
  L1: one-step equation with simple ax = c or x + b = c.
  L2: standard two-step equation ax + b = c.
  L3: distributive a(bx + c) = d.
  L4: two-variable linear combo given ax + by = c and x = d → solve y.
  L5: quadratic x^2 + bx + c = 0 → sum of roots (Vieta).
  L6: quadratic x^2 + bx + c = 0 → product of roots.
  L7: system of 2 linear equations → x + y.
  L8: system of 2 linear equations → x * y.
  L9: 3-term arithmetic with three handwritten numbers (multi-step) → final.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_HANDWRITING_FONTS = ["cursive", "serif", "monospace", "fantasy"]
_INK_COLOURS = ["#232523", "#1a3c5e", "#4a1a2e", "#2d4a1a", "#1a2a3a", "#263238"]
_PAPER_COLOURS = ["#fffef5", "#fffdf0", "#fdfaf0", "#f8f6eb", "#f3f1e6"]
_RULE_COLOURS = ["#d7d4c3", "#cbd1c4", "#cfc9a8", "#c9c4b5"]

class HandwrittenExpressionQA(StandaloneVisualEnv):
    ENV_NAME = "handwritten_expression"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        fams = {
            0: "two_step_easy",
            1: "one_step",
            2: "two_step",
            3: "distributive",
            4: "two_var_given",
            5: "quadratic_sum",
            6: "quadratic_product",
            7: "system_sum",
            8: "system_product",
            9: "three_term_arith",
        }
        # L6-L9 get much heavier visual noise: scratches, strikethroughs,
        # partial occlusions that must survive OCR.
        return {
            "family": fams[level],
            "coef_max": 3 + level,
            "const_max": 6 + level * 3,
            "noise_density": 0.02 + 0.025 * level,
            "slant_max_deg": 1.5 + level * 1.0,
            "occlusion": level >= 4,
            "heavy_occlusion": level >= 6,
            "scratch_lines": level >= 6,
            "n_heavy_smudges": 0 if level < 6 else (3 + (level - 6) * 2),
            "show_formula_hint": level == 0,
            "ink_colours": _INK_COLOURS if level >= 3 else _INK_COLOURS[:2],
        }

    def _generate_problem(self, seed, parameter):
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)

        for _ in range(40):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg, level):
        fam = cfg["family"]
        hint = ""
        if cfg.get("show_formula_hint"):
            hint = " (Formula reminder: for ax + b = c, x = (c - b)/a.)"

        if fam == "two_step_easy":
            # Keep a small so it's L0-easy, but answer is shown on image not in Q
            x_val = rng.choice([v for v in range(-5, 6) if v != 0])
            a = rng.randint(2, 4)
            b = rng.randint(-6, 6)
            c = a * x_val + b
            expr = self._fmt_two_step(a, b, c)
            q = f"Solve the handwritten equation for x (integer answer).{hint}"
            return q, str(x_val), self._render([expr], cfg, rng)

        if fam == "one_step":
            kind = rng.choice(["ax=c", "x+b=c"])
            x_val = rng.choice([v for v in range(-8, 9) if v != 0])
            if kind == "ax=c":
                a = rng.randint(2, cfg["coef_max"])
                c = a * x_val
                expr = f"{a}x = {c}"
            else:
                b = rng.randint(-cfg["const_max"], cfg["const_max"])
                c = x_val + b
                if b >= 0:
                    expr = f"x + {b} = {c}"
                else:
                    expr = f"x - {abs(b)} = {c}"
            q = "Solve the handwritten equation for x (integer answer)."
            return q, str(x_val), self._render([expr], cfg, rng)

        if fam == "two_step":
            x_val = rng.choice([v for v in range(-8, 9) if v != 0])
            a = rng.randint(2, cfg["coef_max"])
            b = rng.randint(-cfg["const_max"], cfg["const_max"])
            c = a * x_val + b
            expr = self._fmt_two_step(a, b, c)
            q = "Solve the handwritten equation for x (integer answer)."
            return q, str(x_val), self._render([expr], cfg, rng)

        if fam == "distributive":
            x_val = rng.choice([v for v in range(-6, 7) if v != 0])
            a = rng.randint(2, max(3, cfg["coef_max"] // 2))
            b = rng.randint(2, max(3, cfg["coef_max"] // 2))
            c_inner = rng.randint(-cfg["const_max"] // 3, cfg["const_max"] // 3)
            d = a * (b * x_val + c_inner)
            inner = f"{b}x + {c_inner}" if c_inner >= 0 else f"{b}x - {abs(c_inner)}"
            expr = f"{a}({inner}) = {d}"
            q = "Solve the handwritten equation for x (integer answer)."
            return q, str(x_val), self._render([expr], cfg, rng)

        if fam == "two_var_given":
            x_val = rng.choice([v for v in range(-6, 7) if v != 0])
            y_val = rng.choice([v for v in range(-6, 7) if v != 0])
            a = rng.randint(1, cfg["coef_max"] // 2 + 2)
            b = rng.randint(1, cfg["coef_max"] // 2 + 2)
            rhs = a * x_val + b * y_val
            # Two equations on image: ax + by = rhs, and x = x_val
            eq1 = self._fmt_linear_2var(a, b, rhs)
            eq2 = f"x = {x_val}"
            q = ("Two handwritten equations relating x and y are shown. "
                 "Using both, determine the value of y (integer).")
            return q, str(y_val), self._render([eq1, eq2], cfg, rng)

        if fam == "quadratic_sum":
            r1 = rng.randint(-6, 6)
            r2 = rng.randint(-6, 6)
            if r1 == 0 and r2 == 0:
                r1 = rng.choice([-3, -2, -1, 1, 2, 3])
            b_coef = -(r1 + r2)
            c_coef = r1 * r2
            expr = self._fmt_quadratic(b_coef, c_coef)
            q = ("The image shows a quadratic equation in x (equal to 0). "
                 "What is the SUM of its two roots? (integer)")
            return q, str(r1 + r2), self._render([expr], cfg, rng)

        if fam == "quadratic_product":
            r1 = rng.randint(-6, 6)
            r2 = rng.randint(-6, 6)
            if r1 == 0 and r2 == 0:
                r1 = rng.choice([-3, -2, -1, 1, 2, 3])
            b_coef = -(r1 + r2)
            c_coef = r1 * r2
            expr = self._fmt_quadratic(b_coef, c_coef)
            q = ("The image shows a quadratic equation in x (equal to 0). "
                 "What is the PRODUCT of its two roots? (integer)")
            return q, str(r1 * r2), self._render([expr], cfg, rng)

        if fam in ("system_sum", "system_product"):
            x_val = rng.choice([v for v in range(-5, 6) if v != 0])
            y_val = rng.choice([v for v in range(-5, 6) if v != 0])
            for _ in range(25):
                a = rng.randint(1, cfg["coef_max"] // 2 + 1)
                b = rng.randint(1, cfg["coef_max"] // 2 + 1)
                c = rng.randint(1, cfg["coef_max"] // 2 + 1)
                d = rng.randint(1, cfg["coef_max"] // 2 + 1)
                det = a * d - b * c
                if det != 0:
                    break
            else:
                return None
            e = a * x_val + b * y_val
            f_val = c * x_val + d * y_val
            eq1 = self._fmt_linear_2var(a, b, e)
            eq2 = self._fmt_linear_2var(c, d, f_val)
            if fam == "system_sum":
                q = ("The image shows a system of two handwritten equations in x and y. "
                     "Solve the system and return the SUM x + y (integer).")
                return q, str(x_val + y_val), self._render([eq1, eq2], cfg, rng)
            else:
                q = ("The image shows a system of two handwritten equations in x and y. "
                     "Solve the system and return the PRODUCT x * y (integer).")
                return q, str(x_val * y_val), self._render([eq1, eq2], cfg, rng)

        if fam == "three_term_arith":
            # L9 iter-4 hardened 2026-04-17: THREE handwritten lines now.
            # Line 1: a = ..,  b = ..
            # Line 2: defines c as an EXPRESSION of a and b, e.g. c = a*b-k
            # Line 3: compound expression with a, b, c + nested radical +
            #         exponent — must substitute c from line 2 first.
            # Pattern line 3:
            #   (a*c + b^2) / m - √(p + q) + k*(c - b) = ?
            # Choose so that √(p+q) and all divisions resolve to integers.
            for _ in range(300):
                a = rng.randint(3, 9)
                b = rng.randint(2, 9)
                kdef = rng.randint(1, 5)
                # c is derived from a and b via c = a*b - kdef.
                c = a * b - kdef
                if c <= 0:
                    continue
                sqrt_val = rng.choice([3, 4, 5, 6, 7, 8, 9])
                radicand = sqrt_val * sqrt_val
                p = rng.randint(1, radicand - 1)
                q = radicand - p
                k = rng.randint(1, 5)
                # Pick m so that (a*c + b*b) is divisible by m → integer.
                numer = a * c + b * b
                m_candidates = [mm for mm in (2, 3, 4, 5, 6, 7)
                                if numer % mm == 0]
                if not m_candidates:
                    continue
                m = rng.choice(m_candidates)
                # Guarantee positive result
                result = numer // m - sqrt_val + k * (c - b)
                if result <= 0 or result > 9999:
                    continue
                break
            else:
                return None
            line1 = f"a = {a}     b = {b}"
            line2 = f"c = a × b − {kdef}"
            line3 = (f"(a × c + b²) ÷ {m} − √({p} + {q}) "
                     f"+ {k} × (c − b) = ?")
            q_text = (
                "Three handwritten lines are shown. Line 1 gives the "
                "values of a and b. Line 2 DEFINES c as a function of "
                "a and b — compute c first by substituting. Line 3 is "
                "the final expression using a, b, c, a nested square "
                "root, and integer division. Apply standard operator "
                "precedence (parentheses, exponents/roots, × ÷ L→R, "
                "+ − L→R). Integer answer."
            )
            return q_text, str(result), self._render(
                [line1, line2, line3], cfg, rng)
        return None

    # ------------------------------------------------------------------ #
    # Formatting helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fmt_two_step(a, b, c):
        if b >= 0:
            return f"{a}x + {b} = {c}"
        return f"{a}x - {abs(b)} = {c}"

    @staticmethod
    def _fmt_linear_2var(a, b, rhs, vars_=("x", "y")):
        parts = []
        if a == 1:
            parts.append(vars_[0])
        elif a == -1:
            parts.append(f"-{vars_[0]}")
        else:
            parts.append(f"{a}{vars_[0]}")
        if b > 0:
            parts.append(f"+ {vars_[1]}" if b == 1 else f"+ {b}{vars_[1]}")
        elif b < 0:
            parts.append(f"- {vars_[1]}" if b == -1 else f"- {abs(b)}{vars_[1]}")
        return " ".join(parts) + f" = {rhs}"

    @staticmethod
    def _fmt_quadratic(b_coef, c_coef):
        parts = ["x\u00b2"]
        if b_coef > 0:
            parts.append("+ x" if b_coef == 1 else f"+ {b_coef}x")
        elif b_coef < 0:
            parts.append("- x" if b_coef == -1 else f"- {abs(b_coef)}x")
        if c_coef > 0:
            parts.append(f"+ {c_coef}")
        elif c_coef < 0:
            parts.append(f"- {abs(c_coef)}")
        return " ".join(parts) + " = 0"

    # ------------------------------------------------------------------ #
    # Rendering: simulate handwritten text on paper with noise
    # ------------------------------------------------------------------ #

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Strict integer match (P6.6 audit). Base 5% tolerance falsely accepts
        e.g. 31.2 for GT=30 on this env's discrete-integer answers; override
        strict for integer GT, fall back to base for non-integer GT (e.g. when
        L0 returns a string label)."""
        import re as _re
        p = predicted.strip().lower().rstrip(".").replace(",", "")
        g = ground_truth.strip().lower().rstrip(".")
        if p == g:
            return True
        if not _re.match(r"^-?\d+$", g):
            return super()._check_answer(predicted, ground_truth)
        m = _re.search(r"-?\d+(?:\.\d+)?", p)
        if not m:
            return False
        try:
            v = float(m.group())
            if v != int(v):
                return False
            return int(v) == int(g)
        except (ValueError, TypeError):
            return False

    def _render(self, exprs: List[str], cfg, rng) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        h = 3.4 if len(exprs) == 1 else 4.8
        fig, ax = plt.subplots(figsize=(7.2 * sc, h * sc))
        paper = rng.choice(_PAPER_COLOURS)
        rule_c = rng.choice(_RULE_COLOURS)
        fig.patch.set_facecolor(paper)
        ax.set_facecolor(paper)
        ax.axis("off")

        # Paper ruled lines
        n_rules = rng.randint(6, 10)
        for y in np.linspace(0.0, 1.0, n_rules):
            ax.axhline(y, color=rule_c, alpha=0.55, linewidth=0.6)

        # Margin vertical line on left
        if rng.random() < 0.5:
            ax.axvline(0.08 + rng.uniform(-0.02, 0.02), color="#e3b0b0", alpha=0.5, linewidth=0.8)

        # Noise smudges
        n_sm = int(cfg["noise_density"] * 260)
        for _ in range(n_sm):
            x0 = rng.random()
            y0 = rng.random()
            s = rng.randint(4, 18)
            ax.scatter(x0, y0, s=s, c="#b0a590", alpha=rng.uniform(0.3, 0.5), zorder=1)

        # Occasional big smudge / fingerprint
        if cfg.get("occlusion"):
            for _ in range(rng.randint(1, 3)):
                ox = rng.uniform(0.1, 0.9)
                oy = rng.uniform(0.2, 0.8)
                ell = mpatches.Ellipse((ox, oy), width=rng.uniform(0.05, 0.18),
                                       height=rng.uniform(0.04, 0.12),
                                       angle=rng.uniform(-30, 30),
                                       facecolor="#c4b99a", alpha=rng.uniform(0.25, 0.42), zorder=5)
                ax.add_patch(ell)

        # Heavy smudges at L6-L9: large partially-opaque blobs that cover
        # portions of digits (forces OCR robustness).
        if cfg.get("heavy_occlusion"):
            for _ in range(cfg.get("n_heavy_smudges", 3)):
                ox = rng.uniform(0.2, 0.85)
                # Place near the middle row (where the equation sits).
                oy = rng.uniform(0.35, 0.7)
                ell = mpatches.Ellipse(
                    (ox, oy), width=rng.uniform(0.10, 0.22),
                    height=rng.uniform(0.06, 0.14),
                    angle=rng.uniform(-45, 45),
                    facecolor=rng.choice(["#9c8064", "#a89368", "#6c5d42"]),
                    alpha=rng.uniform(0.35, 0.55), zorder=12)
                ax.add_patch(ell)

        # Scratches/strikethroughs at L6-L9
        if cfg.get("scratch_lines"):
            for _ in range(rng.randint(2, 5)):
                sx1 = rng.uniform(0.1, 0.9)
                sx2 = sx1 + rng.uniform(0.05, 0.3) * rng.choice([-1, 1])
                sy1 = rng.uniform(0.3, 0.75)
                sy2 = sy1 + rng.uniform(-0.05, 0.05)
                ax.plot([sx1, sx2], [sy1, sy2], color="#4d3b22",
                        alpha=rng.uniform(0.3, 0.55),
                        linewidth=rng.uniform(1.2, 2.3), zorder=11)

        hf = rng.choice(_HANDWRITING_FONTS)
        ink = rng.choice(cfg["ink_colours"])

        if len(exprs) == 1:
            rot = rng.uniform(-cfg["slant_max_deg"], cfg["slant_max_deg"])
            fsize = rng.randint(32, 38)
            ax.text(0.5, 0.52, exprs[0], fontsize=fsize, fontweight="bold",
                    ha="center", va="center", family=hf, style="italic",
                    rotation=rot, color=ink, zorder=10, transform=ax.transAxes)
        else:
            y_positions = {
                2: [0.68, 0.32],
                3: [0.75, 0.50, 0.25],
            }.get(len(exprs), [0.75, 0.50, 0.25, 0.12][:len(exprs)])
            for i, expr in enumerate(exprs):
                rot = rng.uniform(-cfg["slant_max_deg"], cfg["slant_max_deg"])
                fsize = rng.randint(26, 32)
                ax.text(0.5, y_positions[i], expr, fontsize=fsize, fontweight="bold",
                        ha="center", va="center", family=hf, style="italic",
                        rotation=rot, color=ink, zorder=10, transform=ax.transAxes)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        return self.fig_to_pil(fig, dpi=style["dpi"])
