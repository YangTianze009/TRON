"""
Parabola Translate to Vertex Form QA.

A quadratic curve is shown together with a translation vector (h, k) given in
the figure title / caption. The student must output the translated curve in
vertex form `y = a(x - p)^2 + q`.

Source curve is given either in standard form y = ax^2 + bx + c (drawn from
its plot, with the equation also printed in the figure to remove OCR
ambiguity) or in factored form y = a(x - r1)(x - r2). After translating the
graph by (h, k) (move right by h and up by k), the new vertex is
(orig_vertex_x + h, orig_vertex_y + k); the leading coefficient stays a.

Difficulty axes:
  - L0..L2: a = 1 only, integer translation, simple curve y = x^2.
  - L3..L5: a in {-2, -1, 1, 2}, larger translation magnitudes.
  - L6..L9: float translations and float a possible.
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "The image shows a quadratic curve and a translation by (h, k). Apply the translation to the curve and write the new equation in vertex form `y = a(x - p)^2 + q`.",
    "Translate the parabola in the figure by the vector (h, k) shown. Output the resulting curve in vertex form `y = a(x - p)^2 + q`.",
    "A parabola is plotted with its formula given in the figure. Translate it by (h, k) and report the equation in vertex form.",
    "Apply the translation (h, k) to the parabola shown and rewrite the result in vertex form `y = a(x - p)^2 + q`.",
    "Find the vertex-form equation of the parabola in the figure after it has been translated by (h, k). Format `y = a(x - p)^2 + q`.",
    "Translate the displayed quadratic curve by (h, k). Express the translated curve in vertex form `y = a(x - p)^2 + q`.",
    "The parabola in the image is translated by the vector (h, k). What is the new equation in vertex form?",
    "As shown in the figure, compute the vertex-form equation of the translated parabola. Translation: (h, k). Format `y = a(x - p)^2 + q`.",
    "Given the parabola shown, translate it by (h, k) and write the result in vertex form.",
    "As shown in the figure, shift the curve by (h, k); write the resulting parabola in vertex form `y = a(x - p)^2 + q`.",
    "Apply the displayed translation to the parabola and output the new vertex-form equation.",
    "As shown in the figure, after translating by (h, k), what does the new parabola look like in vertex form?",
    "Translate the curve drawn in the image by (h, k). Write the new equation as `y = a(x - p)^2 + q`.",
    "The curve y = ax^2 + bx + c shown in the figure is translated by (h, k). Give the new vertex-form equation.",
    "As shown in the figure, find the new equation in vertex form `y = a(x - p)^2 + q` after translating the parabola by (h, k).",
    "Translate the parabola in the image by the indicated vector (h, k); write the result in vertex form.",
    "The diagram shows a parabola and a translation. Apply the translation; report the new equation in vertex form.",
]


def _fmt_int_or_float(x):
    """Format a number compactly: integer if integral, else 1-decimal float."""
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:g}"


def _fmt_a(a):
    """Format leading coefficient: blank for 1, '-' for -1, else number."""
    if abs(a - 1) < 1e-9:
        return ""
    if abs(a + 1) < 1e-9:
        return "-"
    return _fmt_int_or_float(a)


def _build_vertex_form_str(a, p, q):
    """Return canonical 'y = a(x - p)^2 + q' string (compact)."""
    a_str = _fmt_a(a)
    # (x - p)^2: if p positive -> 'x - p'; if p negative -> 'x + |p|'; if zero -> 'x'
    if abs(p) < 1e-9:
        x_part = "x^2"
    elif p > 0:
        x_part = f"(x - {_fmt_int_or_float(p)})^2"
    else:
        x_part = f"(x + {_fmt_int_or_float(-p)})^2"
    body = f"{a_str}{x_part}"
    if abs(q) < 1e-9:
        return f"y = {body}"
    if q > 0:
        return f"y = {body} + {_fmt_int_or_float(q)}"
    return f"y = {body} - {_fmt_int_or_float(-q)}"


def _parse_vertex_form(s: str):
    """Parse a vertex-form expression like 'y = a(x - p)^2 + q' or 'a(x-p)^2+q'.
    Returns (a, p, q) tuple of floats, or None on failure.
    Tolerates: leading 'y =', missing leading coefficient (1 or -1), '+'/'-'
    around p and q, '**2' in place of '^2', spaces.
    """
    import re
    if s is None:
        return None
    txt = s.strip()
    # Strip leading 'y =' if present
    txt = re.sub(r'^\s*y\s*=\s*', '', txt, flags=re.IGNORECASE)
    txt = txt.replace('**', '^').replace(' ', '')
    # Pattern: optional a, then (x[+-]p)^2, then optional [+-]q
    # a may be int/float with optional sign, or empty/'-'
    pat = re.compile(
        r'^([+-]?\d*\.?\d*)\(x([+-])(\d+\.?\d*)\)\^?2([+-]\d+\.?\d*)?$'
    )
    m = pat.match(txt)
    if m:
        a_raw, sign, p_abs, q_raw = m.group(1), m.group(2), m.group(3), m.group(4)
        if a_raw in ('', '+'):
            a_val = 1.0
        elif a_raw == '-':
            a_val = -1.0
        else:
            try:
                a_val = float(a_raw)
            except ValueError:
                return None
        try:
            p_val = float(p_abs) if sign == '-' else -float(p_abs)
            q_val = float(q_raw) if q_raw else 0.0
            return a_val, p_val, q_val
        except ValueError:
            return None
    # Also accept the no-shift x^2 case: '(x)^2' or 'x^2'
    pat_zero = re.compile(
        r'^([+-]?\d*\.?\d*)(?:\(x\)|x)\^?2([+-]\d+\.?\d*)?$'
    )
    m2 = pat_zero.match(txt)
    if m2:
        a_raw, q_raw = m2.group(1), m2.group(2)
        if a_raw in ('', '+'):
            a_val = 1.0
        elif a_raw == '-':
            a_val = -1.0
        else:
            try:
                a_val = float(a_raw)
            except ValueError:
                return None
        try:
            q_val = float(q_raw) if q_raw else 0.0
            return a_val, 0.0, q_val
        except ValueError:
            return None
    return None


class ParabolaTranslateVertexFormQA(StandaloneVisualEnv):
    ENV_NAME = "parabola_translate_vertex_form"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {
                "a_pool": [1],
                "vertex_orig_range": 0 if level == 0 else 1 + level,
                "translate_range": 1 + level,
                "allow_float": False,
            }
        if level <= 5:
            return {
                "a_pool": [-2, -1, 1, 2],
                "vertex_orig_range": 2 + level,
                "translate_range": 3 + level,
                "allow_float": False,
            }
        return {
            "a_pool": [-3, -2, -1, 1, 2, 3],
            "vertex_orig_range": 3 + level,
            "translate_range": 3 + level,
            "allow_float": level >= 8,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1043 + level * 83 + 17)

        a = rng.choice(cfg["a_pool"])
        if cfg["allow_float"] and rng.random() < 0.4:
            # half-integer leading coefficient occasionally
            a = a * 0.5 if abs(a) >= 2 else a

        # Original vertex (h0, k0)
        v_rng = cfg["vertex_orig_range"]
        if v_rng == 0:
            h0, k0 = 0, 0
        else:
            h0 = rng.randint(-v_rng, v_rng)
            k0 = rng.randint(-v_rng, v_rng)

        # Translation (dx, dy)
        t_rng = cfg["translate_range"]
        if cfg["allow_float"] and rng.random() < 0.3:
            dx = round(rng.uniform(-t_rng, t_rng) * 2) / 2.0
            dy = round(rng.uniform(-t_rng, t_rng) * 2) / 2.0
        else:
            dx = rng.randint(-t_rng, t_rng)
            dy = rng.randint(-t_rng, t_rng)
        # Avoid trivial zero-translation
        if dx == 0 and dy == 0:
            dx = 1

        # New vertex
        new_p = h0 + dx
        new_q = k0 + dy

        answer = _build_vertex_form_str(a, new_p, new_q)

        # Source curve description: convert original vertex form to standard
        # form so the figure caption can read y = ax^2 + bx + c.
        # y = a(x - h0)^2 + k0 = a x^2 - 2 a h0 x + (a h0^2 + k0)
        b_coef = -2 * a * h0
        c_coef = a * h0 * h0 + k0

        # Build question
        sidx = (self.seed or 0) % len(_TEMPLATES)
        # Substitute (h, k) numbers into the template wording
        dx_disp = _fmt_int_or_float(dx)
        dy_disp = _fmt_int_or_float(dy)
        question = _TEMPLATES[sidx].replace(
            "(h, k)", f"({dx_disp}, {dy_disp})")

        img = self._render(a, h0, k0, b_coef, c_coef, dx, dy)
        return question, answer, img

    # ---------------- answer check ---------------- #
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # First try direct base check (string equality after normalization)
        if super()._check_answer(predicted, ground_truth):
            return True
        # Then try parsing both as vertex-form expressions and compare
        # tuple (a, p, q) numerically.
        gp = _parse_vertex_form(ground_truth)
        pp = _parse_vertex_form(predicted)
        if gp is not None and pp is not None:
            ag, pg, qg = gp
            ap, pp_, qp = pp
            return (abs(ag - ap) < 1e-3 and abs(pg - pp_) < 1e-3
                    and abs(qg - qp) < 1e-3)
        # 2026-05-04 R3: also accept expanded form y = a x^2 + b x + c — test
        # equality at 3 sample x-values (was failing valid answers like
        # 'y = x^2 - 2x' for vertex (1, -1); L0 was 0.50).
        if gp is not None:
            ag, pg, qg = gp
            try:
                vals_pred = self._eval_quadratic(predicted, [-1.0, 0.0, 1.0])
                if vals_pred is None:
                    return False
                vals_gt = [ag * (x - pg) ** 2 + qg for x in [-1.0, 0.0, 1.0]]
                return all(abs(p - g) < 1e-3 for p, g in zip(vals_pred, vals_gt))
            except (ValueError, TypeError, ZeroDivisionError):
                return False
        return False

    @staticmethod
    def _eval_quadratic(s, xs):
        """Try to evaluate a quadratic expression in x at given x-values.
        Strips 'y =' prefix, replaces ^ with **, tries Python eval.
        Returns list of values or None on failure."""
        import re as _re
        if s is None:
            return None
        txt = s.strip()
        txt = _re.sub(r'^\s*y\s*=\s*', '', txt, flags=_re.IGNORECASE)
        # Replace ^ with ** for Python power syntax
        txt = txt.replace('^', '**')
        # Insert * between digit/) and x or (
        txt = _re.sub(r'(\d)\s*([x(])', r'\1*\2', txt)
        txt = _re.sub(r'\)\s*([x(])', r')*\1', txt)
        # Reject anything with non-arithmetic chars
        if not _re.match(r'^[\d\s+\-*/().x]+$', txt):
            return None
        out = []
        for xv in xs:
            try:
                v = eval(txt, {"__builtins__": {}}, {"x": xv})
                out.append(float(v))
            except (SyntaxError, NameError, ValueError, TypeError, ZeroDivisionError):
                return None
        return out

    # ---------------- render ---------------- #
    def _render(self, a, h0, k0, b_coef, c_coef, dx, dy) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6, 5.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        # Plot original parabola
        x_lo = h0 - 5
        x_hi = h0 + 5
        xs = np.linspace(x_lo, x_hi, 200)
        ys = a * (xs - h0) ** 2 + k0
        ax.plot(xs, ys, color="#1a5276", linewidth=2.2, label="original")
        # Mark original vertex
        ax.plot([h0], [k0], "o", color="#c0392b", markersize=7, zorder=5)
        ax.annotate(f"({h0:g}, {k0:g})", (h0, k0),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=10, color="#c0392b", fontweight="bold")

        # Print the standard-form equation in the title for unambiguous reading
        # e.g. y = x^2; y = -2x^2 + 4x - 1
        def _fmt_term_x(coef, var):
            if abs(coef) < 1e-9:
                return ""
            sign = "+" if coef > 0 else "-"
            mag = abs(coef)
            if abs(mag - 1) < 1e-9:
                return f" {sign} {var}"
            return f" {sign} {_fmt_int_or_float(mag)}{var}"
        # Leading term
        if abs(a - 1) < 1e-9:
            lead = "x^2"
        elif abs(a + 1) < 1e-9:
            lead = "-x^2"
        else:
            lead = f"{_fmt_int_or_float(a)}x^2"
        eq = f"y = {lead}{_fmt_term_x(b_coef, 'x')}{_fmt_term_x(c_coef, '')}"
        # Tidy: remove leading space in case
        eq = eq.replace("y =  ", "y = ")
        ax.set_title(eq + f"   |   translate by ({_fmt_int_or_float(dx)}, "
                          f"{_fmt_int_or_float(dy)})",
                     fontsize=11)

        # Axes
        ax.axhline(0, color="#888", linewidth=0.8)
        ax.axvline(0, color="#888", linewidth=0.8)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
