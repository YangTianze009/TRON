"""
Unit Conversion Visual QA environment.

Goal: read a labeled measurement diagram (ruler, protractor,
thermometer, or engineering drawing) showing a value in one unit
system, plus a conversion reference (e.g. ``1 inch = 2.54 cm``), and
convert the shown measurement to the requested target unit.

Targets geometry problem solving and metric-geometry-length.

Difficulty axes:
  A) conversion_type: L0 = metric within (cm to mm), L3 = metric to
     imperial, L6 = area units (cm^2 to m^2), L9 = compound rates
     (ft/s to km/h).
  B) measurement_precision: L0 = integer, L5 = one decimal,
     L9 = two decimals.

Format: 4-way MCQ (single letter).
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import RegularPolygon
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

class UnitConversionVisualQA(StandaloneVisualEnv):
    ENV_NAME = "unit_conversion_visual"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # Redesign 2026-04-17 v2: fix L0→L9 inversion (L0 was 0.55, L9 was
        # 1.00). Root cause: the L0 ruler had fewer unambiguous markings at
        # the measured value; L9 compound_rate (ft/s → km/h) was actually
        # one multiplication. Fix:
        #   • L0/L1: clean ruler readings, integer values, big marker arrow.
        #   • L9: CHAINED two-step conversion (value in unit A, convert to
        #     unit B using reference 1, then to unit C using reference 2).
        #     Both references are shown on the image.
        # Iter 4 (2026-04-17): L6=1.00 peak — metric_to_imperial with
        # precision=two_decimal was still a single multiplication. Bump
        # L6 into area_units (requires squared-factor awareness) and leave
        # L7 on two-step chained (lighter version).
        if level <= 1:
            conv_type = "metric_within"
            precision = "integer"
        elif level <= 3:
            conv_type = "metric_within"
            precision = "one_decimal"
        elif level <= 5:
            conv_type = "metric_to_imperial"
            precision = "one_decimal"
        elif level == 6:
            conv_type = "area_units"
            precision = "two_decimal"
        elif level == 7:
            conv_type = "area_units"
            precision = "two_decimal"
        elif level == 8:
            conv_type = "compound_rate"
            precision = "two_decimal"
        else:
            # L9: chained three-step conversion — non-trivial even with calculator.
            conv_type = "chained"
            precision = "two_decimal"
        diagram_pool = {
            "metric_within": ["ruler_cm", "thermometer"],
            "metric_to_imperial": ["ruler_inch", "engineering"],
            "area_units": ["square_grid", "engineering"],
            "compound_rate": ["engineering", "gauge"],
            # L9 chained (3-step) uses engineering drawing so the three
            # reference lines render on the image without cramping the
            # measurement itself.
            "chained": ["engineering"],
        }
        return {
            "conv_type": conv_type,
            "precision": precision,
            "diagram_pool": diagram_pool[conv_type],
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 709)
        self._primary_complexity_feature = level

        for _ in range(25):
            try:
                r = self._try_generate(sub_rng, cfg, level)
                if r is not None:
                    return r
            except Exception:
                continue
        return None

    def _try_generate(self, rng: random.Random, cfg: Dict,
                      level: int) -> Optional[Tuple[str, str, Image.Image]]:
        conv = cfg["conv_type"]
        precision = cfg["precision"]

        if conv == "metric_within":
            src_unit, dst_unit, factor = self._pick_metric_within(rng)
            val = self._pick_value(rng, precision)
            gt = val * factor
            ref_line = f"1 {src_unit} = {self._fmt_num(factor)} {dst_unit}"
        elif conv == "metric_to_imperial":
            src_unit, dst_unit, factor = self._pick_metric_imperial(rng)
            val = self._pick_value(rng, precision)
            gt = val * factor
            ref_line = f"1 {src_unit} = {self._fmt_num(factor)} {dst_unit}"
        elif conv == "area_units":
            # e.g. cm^2 to m^2 (factor 1/10000) or mm^2 to cm^2 (1/100)
            pair = rng.choice([
                ("cm^2", "m^2", 1 / 10000.0),
                ("mm^2", "cm^2", 1 / 100.0),
                ("m^2", "km^2", 1 / 1000000.0),
                ("in^2", "ft^2", 1 / 144.0),
            ])
            src_unit, dst_unit, factor = pair
            val = self._pick_value(rng, precision, area=True)
            gt = val * factor
            ref_line = f"1 {src_unit} = {self._fmt_num(factor)} {dst_unit}"
        elif conv == "chained":
            # L9 iter-3: THREE-step chain with density/molar conversions.
            # src → mid1 → mid2 → dst using THREE factors, plus the output
            # must be rounded to 2 decimals.
            chain = rng.choice([
                # (src, m1, m2, dst, f1, f2, f3)
                ("mi/h", "ft/s", "m/s", "cm/ms", 1.46667, 0.3048, 0.1),
                ("mi", "yd", "m", "cm", 1760.0, 0.9144, 100.0),
                ("gal", "qt", "pt", "cups", 4.0, 2.0, 2.0),
                ("lb", "oz", "g", "mg", 16.0, 28.3495, 1000.0),
                ("km/h", "m/s", "ft/s", "mph", 1/3.6, 3.28084, 0.681818),
                # Density: g/cm^3 -> kg/m^3 -> lb/ft^3 -> (rare ratio)
                ("g/cm^3", "kg/m^3", "lb/ft^3", "oz/in^3", 1000.0, 0.062428, 0.578704),
            ])
            src_unit, m1_unit, m2_unit, dst_unit, f1, f2, f3 = chain
            val = self._pick_value(rng, precision)
            gt = val * f1 * f2 * f3
            ref_line = (f"1 {src_unit} = {self._fmt_num(f1)} {m1_unit}\n"
                        f"1 {m1_unit} = {self._fmt_num(f2)} {m2_unit}\n"
                        f"1 {m2_unit} = {self._fmt_num(f3)} {dst_unit}")
            factor = f1 * f2 * f3
        else:  # compound_rate
            pair = rng.choice([
                ("ft/s", "km/h", 1.09728),  # 1 ft/s = 1.09728 km/h
                ("mi/h", "km/h", 1.60934),
                ("m/s", "km/h", 3.6),
                ("km/h", "m/s", 1 / 3.6),
            ])
            src_unit, dst_unit, factor = pair
            val = self._pick_value(rng, precision)
            gt = val * factor
            ref_line = f"1 {src_unit} = {self._fmt_num(factor)} {dst_unit}"

        # Round GT per precision
        if precision == "integer":
            gt_r = round(gt, 0)
            if abs(gt_r - round(gt_r)) < 1e-6:
                gt_r = int(round(gt_r))
        elif precision == "one_decimal":
            gt_r = round(gt, 1)
        else:
            gt_r = round(gt, 2)

        # Distractors
        distractors = []
        # Forgot to convert: val itself (or val rounded)
        miscons = [val, val / factor if factor != 0 else val + 1]
        # Off-by-factor-of-10
        miscons.append(gt * 10)
        miscons.append(gt / 10 if gt != 0 else gt + 0.5)
        # Division instead of multiplication
        miscons.append(val / factor if factor != 0 else val - 1)
        # +/- small
        miscons.append(gt_r + rng.choice([-3, -2, 2, 3]) * max(1, abs(gt_r) * 0.1))
        for m in miscons:
            if precision == "integer":
                d = int(round(m))
            elif precision == "one_decimal":
                d = round(m, 1)
            else:
                d = round(m, 2)
            if d != gt_r and d > 0 and d not in distractors:
                distractors.append(d)
            if len(distractors) == 3:
                break
        if len(distractors) < 3:
            return None

        # 2026-05-04 WeMath alignment: 50% of seeds add a 5th E="No correct
        # answer" option (matches reference's dominant 5-way MCQ surface
        # format). Correct letter unchanged because gt_r stays in A-D.
        wemath_style = rng.random() < 0.5
        opts_vals = [gt_r] + distractors[:3]
        rng.shuffle(opts_vals)
        if opts_vals.count(gt_r) > 1:
            return None
        answer_letter = chr(ord("A") + opts_vals.index(gt_r))

        def fmt(v):
            if isinstance(v, int) or abs(v - round(v)) < 1e-6:
                return f"{int(round(v))} {dst_unit}"
            if precision == "one_decimal":
                return f"{v:.1f} {dst_unit}"
            return f"{v:.2f} {dst_unit}"

        opt_strs = [fmt(v) for v in opts_vals]
        if wemath_style:
            opt_strs = opt_strs + ["No correct answer"]

        diagram_kind = rng.choice(cfg["diagram_pool"])
        diag_name = self._diagram_name(diagram_kind)
        phrasings = [
            (f"A {diag_name} displays a measurement (read the value from "
             f"the diagram). The conversion reference is labeled on the "
             f"diagram. Using the measurement and the reference shown, "
             f"what is the measurement in {dst_unit}?"),
            (f"Read the value shown on the {diag_name}. The conversion "
             f"ratio is labeled on the image. Convert the measurement to "
             f"{dst_unit}."),
            (f"Using the {diag_name} reading and the conversion factor "
             f"shown, what is the equivalent value in {dst_unit}?"),
            (f"The {diag_name} shows a measurement; the conversion ratio "
             f"is printed below the diagram. Express the measurement in "
             f"{dst_unit}."),
            (f"Convert the measurement shown on the {diag_name} to "
             f"{dst_unit} (use the printed conversion ratio)."),
        ]
        body = rng.choice(phrasings)
        n_opts = len(opt_strs)
        q = (body + "\n" + "\n".join(
            f"  ({chr(ord('A')+i)}) {opt_strs[i]}" for i in range(n_opts))
            + "\nAnswer with a single letter.")

        image = self._render(diagram_kind, val, src_unit, dst_unit,
                             ref_line, opt_strs, cfg)
        return q, answer_letter, image

    def _pick_metric_within(self, rng):
        pair = rng.choice([
            ("cm", "mm", 10.0),
            ("m", "cm", 100.0),
            ("km", "m", 1000.0),
            ("mm", "cm", 0.1),
            ("cm", "m", 0.01),
        ])
        return pair

    def _pick_metric_imperial(self, rng):
        pair = rng.choice([
            ("in", "cm", 2.54),
            ("ft", "cm", 30.48),
            ("yd", "m", 0.9144),
            ("in", "mm", 25.4),
        ])
        return pair

    def _pick_value(self, rng, precision, area=False):
        if precision == "integer":
            # Iter-3 L0 fix: integer conversions now in 3..10 (middle of ruler,
            # clear major-tick alignment); keeps readings unambiguous.
            return rng.randint(3, 10) if not area else rng.randint(10, 500)
        if precision == "one_decimal":
            base = rng.choice([2, 3, 5, 7, 8, 10]) + rng.choice([0.5, 0.3, 0.7])
            return round(base, 1)
        # two_decimal
        base = rng.uniform(1.5, 9.5)
        return round(base, 2)

    @staticmethod
    def _fmt_num(v):
        if isinstance(v, int):
            return str(v)
        if abs(v - round(v)) < 1e-6:
            return str(int(round(v)))
        return f"{v:g}"

    @staticmethod
    def _diagram_name(kind):
        return {
            "ruler_cm": "metric ruler",
            "ruler_inch": "inch ruler",
            "thermometer": "thermometer",
            "square_grid": "square grid",
            "engineering": "engineering drawing",
            "gauge": "speed gauge",
        }.get(kind, "measurement diagram")

    # -------------------------------------------------- #
    def _render(self, diagram_kind, val, src_unit, dst_unit,
                ref_line, opts, cfg) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        ff = style["font_family"]
        fs = style["font_size_base"]
        palette = style["palette"]
        lw = style["line_width"]
        geo_line = style["geo_line_color"]
        rng = self._rng

        fig = plt.figure(figsize=(10.0 * sc, 6.0 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax_f = fig.add_subplot(1, 2, 1)
        ax_t = fig.add_subplot(1, 2, 2)
        ax_f.set_aspect("equal")
        ax_f.axis("off")
        ax_t.axis("off")

        if diagram_kind in ("ruler_cm", "ruler_inch"):
            length = 12.0
            # Ruler body
            ax_f.add_patch(mpatches.Rectangle((0, 0), length, 1.0,
                                               facecolor=palette[0],
                                               edgecolor=geo_line,
                                               linewidth=lw + 0.2,
                                               alpha=0.3))
            n_major = int(length)
            for i in range(n_major + 1):
                ax_f.plot([i, i], [0, 0.6], "-", color=geo_line, linewidth=lw)
                ax_f.text(i, -0.3, str(i),
                          fontsize=fs - 1, family=ff, ha="center",
                          color="#333")
                if i < n_major:
                    for j in range(1, 10):
                        x = i + j / 10.0
                        h = 0.3 if j % 5 else 0.5
                        ax_f.plot([x, x], [0, h], "-",
                                  color=geo_line, linewidth=lw * 0.5)
            # Mark the measured value
            mark = min(max(float(val), 0.5), length - 0.3)
            ax_f.annotate("", xy=(mark, 1.3), xytext=(mark, 2.2),
                          arrowprops=dict(arrowstyle="->",
                                          color=palette[4], lw=lw + 0.8))
            ax_f.text(mark, 2.4, f"{self._fmt_num(val)} {src_unit}",
                      fontsize=fs + 1, fontweight="bold",
                      family=ff, ha="center", color=palette[4])
            unit_text = "cm" if diagram_kind == "ruler_cm" else "in"
            ax_f.text(length / 2, -0.9, f"(ruler in {unit_text})",
                      fontsize=fs - 1, family=ff, ha="center", color="#555")
            ax_f.set_xlim(-1, length + 1)
            ax_f.set_ylim(-1.8, 3.3)

        elif diagram_kind == "thermometer":
            # Vertical thermometer
            ax_f.add_patch(mpatches.Rectangle((0, 0), 0.8, 8,
                                               facecolor=palette[0],
                                               edgecolor=geo_line,
                                               linewidth=lw + 0.2,
                                               alpha=0.3))
            ax_f.add_patch(mpatches.Circle((0.4, -0.6), 0.8,
                                            facecolor=palette[2],
                                            edgecolor=geo_line,
                                            linewidth=lw))
            # Scale 0-40 cm (or whatever val maps to)
            for i in range(9):
                y = i
                ax_f.plot([0.8, 1.1], [y, y], "-", color=geo_line, linewidth=lw)
                ax_f.text(1.3, y, str(i * 5), fontsize=fs - 1, family=ff,
                          va="center", color="#333")
            # Fill indicating val
            fill_h = min(max(float(val) / 5.0, 0.2), 7.8)
            ax_f.add_patch(mpatches.Rectangle((0, 0), 0.8, fill_h,
                                               facecolor=palette[4],
                                               edgecolor="none", alpha=0.7))
            ax_f.text(2.2, fill_h,
                      f"{self._fmt_num(val)} {src_unit}",
                      fontsize=fs + 1, fontweight="bold",
                      family=ff, va="center", color=palette[4])
            ax_f.set_xlim(-1.5, 5)
            ax_f.set_ylim(-2, 10)

        elif diagram_kind == "square_grid":
            # Draw a square/rectangle with dimension labeled
            import math as _m
            side = _m.sqrt(float(val))
            # Scale for display
            disp = min(side, 6)
            ax_f.add_patch(mpatches.Rectangle((0, 0), disp, disp,
                                               facecolor=palette[0],
                                               edgecolor=geo_line,
                                               linewidth=lw + 0.4,
                                               alpha=0.3))
            ax_f.text(disp / 2, -0.4, f"side = {_m.sqrt(float(val)):.2f} "
                      f"{src_unit[:-2] if src_unit.endswith('^2') else src_unit}",
                      fontsize=fs, family=ff, ha="center", color=palette[2])
            ax_f.text(disp / 2, disp / 2,
                      f"Area = {self._fmt_num(val)} {src_unit}",
                      fontsize=fs + 1, fontweight="bold",
                      family=ff, ha="center", va="center", color=palette[4])
            ax_f.set_xlim(-1.5, disp + 1.5)
            ax_f.set_ylim(-1.5, disp + 1.5)

        elif diagram_kind == "engineering":
            # An engineering drawing: a part with a dimension
            w = 7
            h = 3
            ax_f.add_patch(mpatches.Rectangle((0, 0), w, h,
                                               facecolor=palette[0],
                                               edgecolor=geo_line,
                                               linewidth=lw + 0.4,
                                               alpha=0.3))
            # Dimension line with arrow tips (moved ABOVE the rectangle).
            # BUGFIX 2026-04-24: value label was at y=-1.2 (below rectangle)
            # and got overlaid by the multi-line Conversion reference textbox
            # at L9. Move the dimension line + label above the rectangle so
            # it never collides with the bottom conversion textbox.
            ax_f.annotate("", xy=(w, h + 0.6), xytext=(0, h + 0.6),
                          arrowprops=dict(arrowstyle="<->", color="#333",
                                          lw=lw + 0.3))
            ax_f.text(w / 2, h + 1.0,
                      f"{self._fmt_num(val)} {src_unit}",
                      fontsize=fs + 1, fontweight="bold",
                      family=ff, ha="center", color=palette[4])
            ax_f.set_xlim(-1.2, w + 1.2)
            ax_f.set_ylim(-2.2, h + 2.0)

        elif diagram_kind == "gauge":
            # Circular gauge (speedometer-style)
            cx, cy, rr = 0, 0, 3
            ax_f.add_patch(mpatches.Circle((cx, cy), rr,
                                            facecolor=palette[0],
                                            edgecolor=geo_line,
                                            linewidth=lw + 0.4,
                                            alpha=0.3))
            # Tick marks around arc from 180 deg to 0 deg (top semicircle)
            for a in range(0, 181, 15):
                rad = math.radians(a)
                x1 = cx + rr * math.cos(rad)
                y1 = cy + rr * math.sin(rad)
                x2 = cx + (rr - 0.3) * math.cos(rad)
                y2 = cy + (rr - 0.3) * math.sin(rad)
                ax_f.plot([x1, x2], [y1, y2], "-",
                          color=geo_line, linewidth=lw)
                if a % 30 == 0:
                    x3 = cx + (rr - 0.8) * math.cos(rad)
                    y3 = cy + (rr - 0.8) * math.sin(rad)
                    ax_f.text(x3, y3, f"{(180 - a) // 3}",
                              fontsize=fs - 1, family=ff,
                              ha="center", va="center", color="#333")
            # Pointer angle based on val (map 0..60 -> 180..0 deg)
            vmax = 60.0
            frac = min(max(float(val) / vmax, 0.0), 1.0)
            ang = 180 - 180 * frac
            rad = math.radians(ang)
            ax_f.plot([cx, cx + (rr - 0.6) * math.cos(rad)],
                      [cy, cy + (rr - 0.6) * math.sin(rad)],
                      "-", color=palette[4], linewidth=lw + 1.5)
            ax_f.text(cx, cy - rr - 0.6,
                      f"{self._fmt_num(val)} {src_unit}",
                      fontsize=fs + 1, fontweight="bold",
                      family=ff, ha="center", color=palette[4])
            ax_f.set_xlim(cx - rr - 1, cx + rr + 1)
            ax_f.set_ylim(cy - rr - 1.5, cy + rr + 1)

        # Reference note on the diagram (bottom-center). NOTE: engineering
        # diagram moves its dimension label ABOVE the rectangle to avoid
        # collision with this textbox (see BUGFIX 2026-04-24 above).
        ax_f.text(0.5, 0.02, f"Conversion: {ref_line}",
                  transform=ax_f.transAxes,
                  fontsize=fs, family=ff, ha="center", va="bottom",
                  color="#1a1a1a",
                  bbox=dict(boxstyle="round,pad=0.25",
                            facecolor="#fff",
                            edgecolor="#aaa", alpha=0.92))

        title_pool = [f"{self._diagram_name(diagram_kind).title()}",
                      "Unit Conversion", "Measurement",
                      "Convert Units", "Reading Diagram"]
        ax_f.set_title(rng.choice(title_pool),
                       fontsize=fs + 2, family=ff, pad=8)

        # Right panel: options (measurement + reference are on the
        # left diagram; don't duplicate numbers here)
        ax_t.set_xlim(0, 10)
        ax_t.set_ylim(0, 12)
        ax_t.text(0.3, 11.5, "Given:", fontsize=fs + 1, fontweight="bold",
                  family=ff, ha="left", va="top", color="#2c3e50")
        y = 10.8
        given_label = ("Read the measurement value and conversion "
                       "reference from the diagram on the left.")
        for ln in self._wrap(given_label, 40):
            ax_t.text(0.3, y, ln, fontsize=fs, family=ff,
                      ha="left", va="top", color="#1a1a1a")
            y -= 0.55
        y -= 0.3
        ax_t.text(0.3, y, "Ask:", fontsize=fs + 1, fontweight="bold",
                  family=ff, ha="left", va="top", color="#2c3e50")
        y -= 0.55
        ax_t.text(0.3, y, f"Convert to {dst_unit}.",
                  fontsize=fs, family=ff, ha="left", va="top", color="#1a1a1a")
        y -= 0.75
        ax_t.text(0.3, y, "Options:", fontsize=fs + 1, fontweight="bold",
                  family=ff, ha="left", va="top", color="#2c3e50")
        y -= 0.55
        for i, o in enumerate(opts):
            ax_t.text(0.5, y, f"({chr(ord('A') + i)}) {o}",
                      fontsize=fs, family=ff, ha="left", va="top",
                      color="#1a1a1a")
            y -= 0.55

        fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02,
                            wspace=0.15)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    @staticmethod
    def _wrap(text: str, width: int = 40) -> List[str]:
        out, cur = [], ""
        for word in text.split():
            if len(cur) + len(word) + 1 > width:
                out.append(cur)
                cur = word
            else:
                cur = (cur + " " + word).strip()
        if cur:
            out.append(cur)
        return out
