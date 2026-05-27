"""
Rotation Speed Comparison QA.

Shows two (or three at high levels) circular dials with start/end arrow positions
and time annotations. Asks which pointer rotated faster.

Difficulty axes:
  A) angle_range: under 180 -> crossing 360
  B) rate_difference: 2x ratio -> within 5%
"""
import math, random
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from .standalone_base import StandaloneVisualEnv

_TITLE_VARIANTS_RSC = [
    "Angular Speed Comparison",
    "Rotation Rate Comparison",
    "Pointer Velocity Test",
    "Which Spins Fastest?",
    "Compare Rotation Speeds",
]

_QUESTION_TEMPLATES_RSC = [
    "Each dial shows a pointer's start position (dashed) and end position (solid), plus the time taken. Which dial's pointer rotated fastest (highest angular velocity)?\n{opt_str}",
    "Compare the angular speeds shown on the dials. The dashed arrow marks the start angle, the solid arrow marks the end angle. Which dial spun fastest?\n{opt_str}",
    "Each clock-like dial below depicts a pointer that swept from the dashed position to the solid position over the listed duration. Identify the dial with the greatest angular velocity.\n{opt_str}",
    "Inspect the dials. Each pointer travelled from the dashed orientation to the solid orientation during the indicated time. Pick the dial whose pointer rotated the most degrees per second.\n{opt_str}",
]

class RotationSpeedComparisonQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive: disable rotation augmentation
    ENV_NAME = "rotation_speed_comparison"

    def _level_config(self, level):
        # rate_ratio_max controls how close fastest/2nd-fastest can be.
        # L0: at least 2x apart. L9: within ~8% (very close visual call).
        if level <= 1:
            rate_ratio_max = 2.5
        elif level <= 3:
            rate_ratio_max = 1.7
        elif level <= 5:
            rate_ratio_max = 1.35
        elif level <= 7:
            rate_ratio_max = 1.18
        else:
            rate_ratio_max = 1.10
        # L6+: hide numeric (angle, time) labels — must read arc visually.
        # BUGFIX 2026-04-24: L8-L9 previously hid BOTH time and angle labels,
        # which makes angular_speed = angle / time unsolvable (questions still
        # reference "time taken" / "degrees per second"). Keep time visible at
        # L8/L9 so student must read arcs and apply time → time_only mode.
        if level <= 5:
            label_mode = "full"
        else:
            label_mode = "time_only"   # show only time, hide angle
        # Structural: L0..L4 use 2 dials; L5..L6 use 2 (label drop in); L7+ use 3.
        if level >= 8:
            n_dials = 4
        elif level >= 7:
            n_dials = 3
        else:
            n_dials = 2
        return {
            'max_angle': 180 + level * 30,
            'rate_diff_min': max(1.05, 2.0 - level * 0.15),
            'rate_ratio_max': rate_ratio_max,
            'n_dials': n_dials,
            'label_mode': label_mode,
        }

    def _generate_problem(self, seed, parameter):
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1007)
        style = self._random_style()

        n_dials = cfg['n_dials']
        # Build the fastest dial first, then constrain other dials to be
        # within rate_ratio_max factor (so the fastest is at most that much
        # faster than each other).
        fastest_angle = rng.randint(60, cfg['max_angle'])
        fastest_time = round(rng.uniform(1.0, 5.0), 1)
        fastest_rate = fastest_angle / fastest_time
        ratio_max = cfg['rate_ratio_max']

        # Layout / styling randomization
        layout_orient = rng.choice(["row", "row", "grid"])
        if n_dials >= 4:
            layout_orient = "grid"
        circle_radius_var = rng.uniform(0.85, 1.05)
        ring_color_pool = ["#333", "#5d4037", "#1b4f72", "#2d3436", "#4a235a"]
        ring_color = rng.choice(ring_color_pool)

        dial_data = []
        # Fastest will go at random index
        fastest_idx = rng.randrange(n_dials)
        for i in range(n_dials):
            if i == fastest_idx:
                d = {
                    'angle': fastest_angle,
                    'time': fastest_time,
                    'rate': fastest_rate,
                    'start_angle': rng.randint(0, 350),
                }
            else:
                # Choose a slower rate within [fastest_rate / ratio_max, fastest_rate * 0.97]
                rate_lo = fastest_rate / ratio_max
                rate_hi = fastest_rate * 0.97
                if rate_hi <= rate_lo:
                    rate_hi = rate_lo + 0.5
                target_rate = rng.uniform(rate_lo, rate_hi)
                # Pick angle/time pair giving approximately that rate
                t = round(rng.uniform(1.0, 5.0), 1)
                a = max(30, int(round(target_rate * t)))
                if a > cfg['max_angle']:
                    a = cfg['max_angle']
                actual_rate = a / t
                d = {
                    'angle': a,
                    'time': t,
                    'rate': actual_rate,
                    'start_angle': rng.randint(0, 350),
                }
            dial_data.append(d)

        rates = [d['rate'] for d in dial_data]
        # Recompute fastest in case rounding changed ordering
        fastest_idx = rates.index(max(rates))
        # Verify fastest is actually fastest by a margin
        sorted_rates = sorted(rates, reverse=True)
        if len(sorted_rates) >= 2 and sorted_rates[1] >= sorted_rates[0] * 0.99:
            # Bump fastest to ensure clear winner
            dial_data[fastest_idx]['time'] = max(0.5, dial_data[fastest_idx]['time'] - 0.3)
            dial_data[fastest_idx]['rate'] = (
                dial_data[fastest_idx]['angle'] / dial_data[fastest_idx]['time'])
            rates = [d['rate'] for d in dial_data]
            fastest_idx = rates.index(max(rates))
        correct = chr(65 + fastest_idx)  # A, B, or C

        # Draw
        sc = style['figsize_scale']
        if layout_orient == "grid" and n_dials >= 4:
            ncols = 2
            nrows = (n_dials + 1) // 2
            fig, axes_grid = plt.subplots(nrows, ncols, figsize=(5*ncols*sc, 5*nrows*sc))
            axes = list(axes_grid.flatten()) if hasattr(axes_grid, 'flatten') else [axes_grid]
        elif layout_orient == "grid" and n_dials == 3:
            fig, axes_grid = plt.subplots(2, 2, figsize=(10*sc, 10*sc))
            axes = list(axes_grid.flatten())
            # hide the 4th
            axes[3].axis('off')
        else:
            fig, axes = plt.subplots(1, n_dials, figsize=(5*n_dials*sc, 5*sc))
            if n_dials == 1: axes = [axes]
            else: axes = list(axes)
        fig.patch.set_facecolor(style['bg_color'])

        for i, dd in enumerate(dial_data):
            ax = axes[i]
            ax.set_facecolor(style['bg_color'])
            ax.set_aspect('equal')
            ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
            ax.axis('off')

            # Draw circle
            r = circle_radius_var
            circle = plt.Circle((0, 0), r, fill=False, edgecolor=ring_color, linewidth=2)
            ax.add_patch(circle)
            ax.plot(0, 0, 'ko', markersize=5)
            # Optional tick marks (cardinal) at higher levels for visual diversity
            if rng.random() < 0.5:
                for tick_a in (0, 90, 180, 270):
                    tx, ty = math.cos(math.radians(tick_a)) * r, math.sin(math.radians(tick_a)) * r
                    ax.plot([tx*0.92, tx*1.04], [ty*0.92, ty*1.04],
                            color=ring_color, lw=1.2)

            # Start arrow (dashed)
            start_rad = math.radians(dd['start_angle'])
            arrow_extent = r * 0.9
            sx, sy = math.cos(start_rad)*arrow_extent, math.sin(start_rad)*arrow_extent
            ax.annotate('', xy=(sx, sy), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=2, linestyle='--'))

            # End arrow (solid)
            end_rad = math.radians(dd['start_angle'] + dd['angle'])
            ex, ey = math.cos(end_rad)*arrow_extent, math.sin(end_rad)*arrow_extent
            color = style['palette'][i % len(style['palette'])]
            ax.annotate('', xy=(ex, ey), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=color, lw=3))

            # Arc showing rotation
            theta1 = dd['start_angle']
            theta2 = dd['start_angle'] + dd['angle']
            arc = mpatches.Arc((0, 0), r*1.2, r*1.2, angle=0, theta1=theta1, theta2=theta2,
                               color=color, lw=1.5, linestyle=':')
            ax.add_patch(arc)

            label_mode = cfg.get('label_mode', 'full')
            if label_mode == 'full':
                title_str = f"Dial {chr(65+i)}\n{dd['angle']}deg in {dd['time']}s"
            elif label_mode == 'time_only':
                title_str = f"Dial {chr(65+i)}\nTime: {dd['time']}s"
            else:
                title_str = f"Dial {chr(65+i)}"
            ax.set_title(title_str,
                         fontsize=style['font_size_base']+1, fontweight='bold')

        title_str_main = rng.choice(_TITLE_VARIANTS_RSC)
        fig.suptitle(title_str_main, fontsize=style['font_size_base']+3, fontweight='bold')
        try: fig.tight_layout()
        except: pass
        img = self.fig_to_pil(fig, dpi=style['dpi'])

        if n_dials == 2:
            opt_str = "(A) Dial A  (B) Dial B  (C) Same speed  (D) Cannot determine"
        elif n_dials == 3:
            opt_str = "(A) Dial A  (B) Dial B  (C) Dial C  (D) All same"
        else:
            opt_str = "(A) Dial A  (B) Dial B  (C) Dial C  (D) Dial D"
        # Cap correct to options range (e.g. fastest_idx 3 -> D works for 4-dial)
        q_template = rng.choice(_QUESTION_TEMPLATES_RSC)
        q = q_template.format(opt_str=opt_str)
        return q, correct, img

if __name__ == "__main__":
    env = RotationSpeedComparisonQA()
    for lv in [0, 6]:
        ok = env.generate(42, {"level": lv})
        print(f"L{lv}: ok={ok}, answer={env._answer}")
