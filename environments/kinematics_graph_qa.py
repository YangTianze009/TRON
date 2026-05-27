"""
Kinematics Graph QA (D86-D90, P1).

Targets reference scientific-figure templates D86 through D90 + D94 (impulse).

Reference task:
  qid 266 (D86): "v vs t graph ... At which time the car change the direction
                 to drive?"  Ans: 3       (= time when v(t) crosses zero)
  qid 269 (D89): "Starting from rest at t=0 ... acceleration ... speed at
                 t=10s?"   Ans: 40.0     (= ∫₀¹⁰ a dt)
  qid 274 (D94): "force-versus-time graph ... impulse?"   Ans: 60   (= ∫F dt)

Subtypes (cycle through):
  - vt_change_direction : when does v(t) cross zero (sign change)?
  - vt_position_at_t    : position = ∫₀ᵗ v dτ at given t
  - at_speed_at_t       : speed at t = ∫₀ᵗ a dτ (starting from rest)
  - ft_impulse          : impulse = ∫₀ᵀ F dt
  - vt_displacement     : signed displacement from 0 to T
"""
import math
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


class KinematicsGraphQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "kinematics_graph"

    SUBTYPES = ["vt_change_direction", "vt_position_at_t",
                "at_speed_at_t", "ft_impulse", "vt_displacement"]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {"subtypes": ["at_speed_at_t", "ft_impulse"],
                    "n_segs": (2, 2)}
        if level <= 5:
            return {"subtypes": self.SUBTYPES,
                    "n_segs": (2, 3)}
        return {"subtypes": self.SUBTYPES,
                "n_segs": (3, 4)}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 7411 + level * 53 + 7)

        subtype = rng.choice(cfg["subtypes"])
        for _ in range(20):
            r = self._try_generate(rng, cfg, subtype, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg, subtype, level):
        n_min, n_max = cfg["n_segs"]
        n_seg = rng.randint(n_min, n_max)

        # Build piecewise-linear function as a list of (t_i, y_i) breakpoints
        # ascending in t. We pick integer time points and integer y values.
        ts = sorted(rng.sample(list(range(1, 12)), n_seg))
        ts = [0] + ts
        # Pick y values
        if subtype == "vt_change_direction":
            # We need at least one zero crossing.
            # Simple strategy: y starts positive, ends negative.
            ys = [rng.randint(2, 6)]
            for i in range(1, len(ts)):
                if i == len(ts) - 1:
                    ys.append(-rng.randint(2, 6))
                else:
                    ys.append(rng.randint(-4, 5))
            # Verify there is exactly one sign change in [t0, t_end]
            # and answer is the time of crossing.
            crossing_idx = None
            for i in range(len(ts) - 1):
                if ys[i] * ys[i + 1] < 0:
                    if crossing_idx is None:
                        crossing_idx = i
                    else:
                        return None
            if crossing_idx is None:
                return None
            # Linear interp for crossing time
            t1, t2 = ts[crossing_idx], ts[crossing_idx + 1]
            y1, y2 = ys[crossing_idx], ys[crossing_idx + 1]
            t_cross = t1 - y1 * (t2 - t1) / (y2 - y1)
            # Round to integer if close, else half-integer
            if abs(t_cross - round(t_cross)) < 0.05:
                t_cross = round(t_cross)
            else:
                return None  # require integer crossings
            answer = str(t_cross)
            question = (
                f"The figure shows the velocity vs. time graph of an object. "
                f"At what time does the object change the direction of motion?"
            )
            ax_label = ("t (s)", "v (m/s)")
            self._cur_axis_label = ax_label
            img = self._render(ts, ys, ax_label)
            return question, answer, img

        if subtype == "at_speed_at_t":
            # v(t) = ∫₀ᵗ a dτ; pick a piecewise-constant or linear a(t)
            # answer = velocity at the final time t_end (starting at rest).
            # For simplicity use integer slopes per segment.
            ys = [rng.randint(1, 5) for _ in range(len(ts))]
            # Make piecewise-linear acceleration
            t_end = ts[-1]
            # integrate
            v = 0.0
            for i in range(len(ts) - 1):
                # Assume a is the average of ys[i] and ys[i+1]? Use trapezoid.
                a_avg = (ys[i] + ys[i + 1]) / 2.0
                dt = ts[i + 1] - ts[i]
                v += a_avg * dt
            if abs(v - round(v)) > 0.01:
                return None
            v = round(v)
            answer = str(v)
            question = (
                f"Starting from rest at t = 0, an object moves with the "
                f"acceleration a(t) shown in the figure (in m/s² vs t in s). "
                f"What is the speed (in m/s) of the object at t = {t_end}s?"
            )
            ax_label = ("t (s)", "a (m/s²)")
            img = self._render(ts, ys, ax_label)
            return question, answer, img

        if subtype == "ft_impulse":
            # Impulse = ∫₀ᵀ F dt. Picks piecewise-linear F(t)
            ys = [rng.randint(1, 8) for _ in range(len(ts))]
            J = 0.0
            for i in range(len(ts) - 1):
                avg = (ys[i] + ys[i + 1]) / 2.0
                dt = ts[i + 1] - ts[i]
                J += avg * dt
            if abs(J - round(J)) > 0.01:
                return None
            J = round(J)
            t_end = ts[-1]
            answer = str(J)
            question = (
                f"The figure shows the force vs. time graph (F in N vs t in s) "
                f"during a collision. Find the impulse delivered by the force "
                f"in N·s."
            )
            ax_label = ("t (s)", "F (N)")
            img = self._render(ts, ys, ax_label)
            return question, answer, img

        if subtype == "vt_position_at_t":
            ys = [rng.randint(1, 5) for _ in range(len(ts))]
            t_target = ts[-1]
            # x = ∫ v dt
            x = 0.0
            for i in range(len(ts) - 1):
                avg = (ys[i] + ys[i + 1]) / 2.0
                dt = ts[i + 1] - ts[i]
                x += avg * dt
            if abs(x - round(x)) > 0.01:
                return None
            x = round(x)
            answer = str(x)
            question = (
                f"The figure shows the velocity vs. time graph (v in m/s vs t in s). "
                f"Starting from x = 0 at t = 0, what is the position (in m) at "
                f"t = {t_target}s?"
            )
            ax_label = ("t (s)", "v (m/s)")
            img = self._render(ts, ys, ax_label)
            return question, answer, img

        if subtype == "vt_displacement":
            ys = [rng.randint(-4, 5) for _ in range(len(ts))]
            x = 0.0
            for i in range(len(ts) - 1):
                avg = (ys[i] + ys[i + 1]) / 2.0
                dt = ts[i + 1] - ts[i]
                x += avg * dt
            if abs(x - round(x)) > 0.01:
                return None
            x = round(x)
            t_end = ts[-1]
            answer = str(x)
            question = (
                f"The figure shows the velocity vs. time graph (v in m/s vs t in s). "
                f"What is the signed displacement (in m) from t = 0 to t = {t_end}s?"
            )
            ax_label = ("t (s)", "v (m/s)")
            img = self._render(ts, ys, ax_label)
            return question, answer, img

        return None

    def _render(self, ts, ys, ax_label) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        ax.plot(ts, ys, color="#d62728", linewidth=2.4, marker="o",
                markersize=8, markerfacecolor="white", markeredgewidth=2)
        # axes lines
        ax.axhline(0, color="#222", linewidth=1.0)
        ax.axvline(0, color="#222", linewidth=1.0)
        ax.grid(True, alpha=0.3, linestyle="--")

        ax.set_xlabel(ax_label[0], fontsize=12)
        ax.set_ylabel(ax_label[1], fontsize=12)

        # show ticks at integer times for readability
        ax.set_xticks(ts)
        ymax = max(abs(min(ys)), abs(max(ys)), 1) + 1
        ax.set_ylim(-ymax, ymax)
        ax.set_xlim(min(ts) - 0.5, max(ts) + 0.5)

        return self.fig_to_pil(fig, dpi=110)


if __name__ == "__main__":
    env = KinematicsGraphQA()
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
