"""
Scientific Graph Interpretation QA environment.

Capabilities: D1 (chart value extraction) + X4 (scientific diagram)
Target regression: dynamic-math scientific_figure, MMMU Science.

A plot styled like a scientific paper figure: labeled axes with units,
error bars, legend with 1-4 series, a descriptive title, gridlines.
Data encodes a physical-looking relationship (e.g., pressure vs temperature,
absorbance vs wavelength).

4-option MCQ asking about maxima, slopes, or interpolated values.

Difficulty schedule (0..9):
  Axis 1: n_series = 1 + level // 3           -> 1..4
  Axis 2: question_type  L<=2: read max (x at peak)
          L3..L6: compare slopes between two x values
          L>=7: interpolate between error-bar regions
  Axis 3: axis_format linear + integers (L<=4) -> log / sci-notation (L>=7)
  Axis 4: error bars grow with level

Output: (question_str, answer_letter, PIL_Image)
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_SCENARIOS = [
    {
        "x_label": "Temperature (K)",
        "y_label": "Pressure (kPa)",
        "title": "Pressure vs Temperature",
        "x_range": (270, 400),
        "y_range": (50, 250),
    },
    {
        "x_label": "Wavelength (nm)",
        "y_label": "Absorbance (a.u.)",
        "title": "Absorbance vs Wavelength",
        "x_range": (300, 700),
        "y_range": (0, 2.5),
    },
    {
        "x_label": "Time (s)",
        "y_label": "Voltage (V)",
        "title": "Voltage vs Time",
        "x_range": (0, 50),
        "y_range": (0, 10),
    },
    {
        "x_label": "Concentration (mM)",
        "y_label": "Reaction Rate (mM/s)",
        "title": "Reaction Rate vs Concentration",
        "x_range": (0, 20),
        "y_range": (0, 5),
    },
    {
        "x_label": "Frequency (Hz)",
        "y_label": "Amplitude (mV)",
        "title": "Spectral Amplitude vs Frequency",
        "x_range": (10, 100),
        "y_range": (0, 50),
    },
    {
        "x_label": "Distance (m)",
        "y_label": "Field Strength (V/m)",
        "title": "Field Strength vs Distance",
        "x_range": (1, 20),
        "y_range": (1, 100),
    },
]

_SERIES_NAME_POOLS = [
    ["Sample A", "Sample B", "Sample C", "Sample D"],
    ["Trial 1", "Trial 2", "Trial 3", "Trial 4"],
    ["Group I", "Group II", "Group III", "Group IV"],
    ["Compound X", "Compound Y", "Compound Z", "Compound W"],
]

_MARKERS = ["o", "s", "D", "^", "v", "P", "X"]

class ScientificGraphInterpretationQA(StandaloneVisualEnv):
    ENV_NAME = "scientific_graph_interpretation"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        n_series = min(4, 1 + level // 3)
        if level <= 2:
            qtype = "peak_x"
        elif level <= 6:
            qtype = "steepest_series"
        else:
            qtype = "interpolate"
        # axis format
        if level <= 4:
            axis_format = "linear"
        elif level <= 7:
            axis_format = "sci"
        else:
            axis_format = "log"
        # error bar scale
        err_scale = 0.05 + level * 0.02  # 5%..23%
        n_points = 7 + level
        return {
            "n_series": n_series,
            "qtype": qtype,
            "axis_format": axis_format,
            "err_scale": err_scale,
            "n_points": n_points,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 2269)
        np_rng = np.random.RandomState(sub_rng.randint(0, 1_000_000))

        for _ in range(10):
            try:
                result = self._try_generate(sub_rng, np_rng, cfg, level)
                if result is not None:
                    return result
            except Exception:
                continue
        return None

    def _try_generate(self, rng, np_rng, cfg, level):
        scenario = dict(rng.choice(_SCENARIOS))
        x_lo, x_hi = scenario["x_range"]
        y_lo, y_hi = scenario["y_range"]
        n_points = cfg["n_points"]
        n_series = cfg["n_series"]

        # x values equally spaced
        xs = np.linspace(x_lo, x_hi, n_points)

        # series data
        series_names_pool = rng.choice(_SERIES_NAME_POOLS)
        series_names = series_names_pool[:n_series]
        series_data = []
        series_errors = []
        for s in range(n_series):
            # Generate shape: bell curve, monotone rise, saturation, or damped oscillation
            shape = rng.choice(["bell", "saturation", "linear", "damped"])
            amp = rng.uniform(0.5, 1.0) * (y_hi - y_lo)
            base = y_lo + rng.uniform(0.0, 0.2) * (y_hi - y_lo)
            if shape == "bell":
                mu = x_lo + rng.uniform(0.2, 0.8) * (x_hi - x_lo)
                sigma = max(1.0, (x_hi - x_lo) * rng.uniform(0.08, 0.18))
                ys = base + amp * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
            elif shape == "saturation":
                k = rng.uniform(0.4, 0.9) * (x_hi - x_lo)
                ys = base + amp * (xs - x_lo) / (k + (xs - x_lo))
            elif shape == "linear":
                slope = rng.uniform(-1.0, 1.0) * amp / max(1.0, x_hi - x_lo)
                ys = base + slope * (xs - x_lo) + amp * 0.3
            else:  # damped
                freq = rng.uniform(1.5, 3.0) / max(1.0, x_hi - x_lo)
                ys = base + amp * 0.5 + amp * 0.4 * np.sin(
                    2 * math.pi * freq * (xs - x_lo)) * np.exp(
                    -(xs - x_lo) / max(1.0, x_hi - x_lo))
            ys = np.clip(ys, y_lo, y_hi * 1.1)
            err = np.abs(ys) * cfg["err_scale"] * np_rng.uniform(0.6, 1.4,
                                                                  size=len(ys))
            series_data.append(ys)
            series_errors.append(err)

        qtype = cfg["qtype"]

        if qtype == "peak_x":
            # Pick a random series, find x at its max
            si = rng.randint(0, n_series - 1)
            ys = series_data[si]
            idx = int(np.argmax(ys))
            correct_val = float(xs[idx])
            q_stem = (f"At approximately what value of {scenario['x_label']} does "
                      f"{series_names[si]} reach its maximum?")
            # Distractors: other x ticks
            all_xs = [float(x) for x in xs]
            all_xs.remove(correct_val)
            rng.shuffle(all_xs)
            distractors = []
            for cand in all_xs:
                if abs(cand - correct_val) > (x_hi - x_lo) * 0.05:
                    distractors.append(round(cand, 2))
                if len(distractors) >= 3:
                    break
            if len(distractors) < 3:
                return None
            correct_disp = round(correct_val, 2)
            options = [correct_disp] + distractors[:3]

        elif qtype == "steepest_series":
            # Pick an x range, find which series has the steepest slope in that range
            if n_series < 2:
                return None
            # Use 30%..60% of x range
            x_a = x_lo + (x_hi - x_lo) * 0.3
            x_b = x_lo + (x_hi - x_lo) * 0.6
            # Find closest x indices
            ia = int(np.argmin(np.abs(xs - x_a)))
            ib = int(np.argmin(np.abs(xs - x_b)))
            if ib <= ia:
                ib = ia + 1
            slopes = []
            for s in range(n_series):
                dy = series_data[s][ib] - series_data[s][ia]
                slope = dy / max(1e-9, xs[ib] - xs[ia])
                slopes.append(slope)
            steepest_idx = int(np.argmax([abs(sl) for sl in slopes]))
            # Reject if the top-two absolute slopes are almost equal
            abs_slopes = sorted([abs(sl) for sl in slopes], reverse=True)
            if len(abs_slopes) >= 2 and abs_slopes[0] - abs_slopes[1] < 0.05 * max(
                    abs_slopes[0], 1e-6):
                return None
            correct_name = series_names[steepest_idx]
            q_stem = (f"Between {xs[ia]:.1f} and {xs[ib]:.1f} "
                      f"{scenario['x_label']}, which series shows the "
                      f"steepest change (largest absolute slope)?")
            options = list(series_names)
            rng.shuffle(options)
            correct_idx = options.index(correct_name)
            letter = chr(ord("A") + correct_idx)
            opts_str = " ".join(
                f"({chr(ord('A')+i)}) {v}" for i, v in enumerate(options)
            )
            question = f"{q_stem} Options: {opts_str}. Answer with a single letter."
            image = self._render(rng, scenario, xs, series_data, series_errors,
                                  series_names, cfg)
            return question, letter, image

        elif qtype == "interpolate":
            # Pick a series, interpolate between two points
            si = rng.randint(0, n_series - 1)
            ys = series_data[si]
            i = rng.randint(0, n_points - 2)
            x_target = float((xs[i] + xs[i + 1]) / 2)
            y_interp = float((ys[i] + ys[i + 1]) / 2)
            correct_disp = round(y_interp, 2)
            # Distractors: +/- some fraction of signal amplitude
            amp = float(max(ys) - min(ys)) if max(ys) > min(ys) else 1.0
            offsets = [amp * 0.2, -amp * 0.2, amp * 0.4, -amp * 0.4,
                       amp * 0.6, -amp * 0.6]
            rng.shuffle(offsets)
            distractors = []
            for off in offsets:
                cand = round(correct_disp + off, 2)
                if abs(cand - correct_disp) < 0.05:
                    continue
                if any(abs(cand - d) < 0.05 for d in distractors):
                    continue
                distractors.append(cand)
                if len(distractors) >= 3:
                    break
            if len(distractors) < 3:
                return None
            options = [correct_disp] + distractors[:3]
            q_stem = (f"For the series {series_names[si]}, estimate the "
                      f"{scenario['y_label']} at {scenario['x_label']} = "
                      f"{x_target:.2f} using linear interpolation between the "
                      f"two nearest data points.")

        else:
            return None

        # For peak_x and interpolate: numeric options
        rng.shuffle(options)
        # For peak_x the correct_disp was numeric
        try:
            correct_idx = options.index(correct_disp)
        except ValueError:
            return None
        letter = chr(ord("A") + correct_idx)
        opts_str = " ".join(
            f"({chr(ord('A')+i)}) {v}" for i, v in enumerate(options)
        )
        question = f"{q_stem} Options: {opts_str}. Answer with a single letter."

        image = self._render(rng, scenario, xs, series_data, series_errors,
                              series_names, cfg)
        return question, letter, image

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render(self, rng, scenario, xs, series_data, series_errors,
                series_names, cfg):
        vs = self._random_style()
        palette = list(vs["palette"])
        rng.shuffle(palette)
        markers = list(_MARKERS)
        rng.shuffle(markers)

        fig_w = rng.uniform(7, 8.5) * vs["figsize_scale"]
        fig_h = rng.uniform(5, 6.5) * vs["figsize_scale"]
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        for s, (ys, err, name) in enumerate(zip(series_data, series_errors,
                                                 series_names)):
            color = palette[s % len(palette)]
            marker = markers[s % len(markers)]
            ax.errorbar(xs, ys, yerr=err, fmt=marker + "-",
                        color=color, markersize=rng.randint(5, 8),
                        linewidth=vs["line_width"],
                        capsize=3, label=name, alpha=0.9)

        ax.set_xlabel(scenario["x_label"], fontsize=vs["font_size_base"])
        ax.set_ylabel(scenario["y_label"], fontsize=vs["font_size_base"])
        ax.set_title(scenario["title"], fontsize=vs["font_size_base"] + 3,
                     pad=10)

        if cfg["axis_format"] == "sci":
            ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        elif cfg["axis_format"] == "log":
            # Use log on y if all positive
            all_positive = all((ys > 0).all() for ys in series_data)
            if all_positive:
                ax.set_yscale("log")

        ax.legend(fontsize=vs["font_size_base"] - 1, framealpha=0.9,
                   loc=vs["legend_loc"])
        self._apply_style(fig, ax, vs)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=vs["dpi"])
