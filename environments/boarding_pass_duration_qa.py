"""
Boarding Pass Duration QA (D46, P1).

Reference an external reference — load-bearing failure.

Renders a boarding pass-style image with departure time and arrival time
shown. Asks for the flight duration in minutes.

Verifier: integer minutes (`\\boxed{N}`).
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


def _fmt_time(h, m):
    return f"{h:02d}:{m:02d}"


class BoardingPassDurationQA(StandaloneVisualEnv):
    ENV_NAME = "boarding_pass_duration"
    # Tighten numeric tolerance for tight-precision scoring (0.001 abs) for
    # float answers (50-5000x stricter than the default 5%/0.5 path).
    # Tighten to match.
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001

    # 2026-05-04 R4: full-gradient redesign per a math benchmark an external reference sample.
    # Benchmark hardness: time-arithmetic + cross-day rollover + multi-leg /
    # timezone reasoning. Single-segment same-day arithmetic is L0 territory;
    # benchmark-grade questions need rollover + timezone + multi-leg layovers.
    #
    # Gradient (each level adds a new reasoning step beyond previous):
    #   L0: same-day, 15-min step, ≤2h        (single subtraction, generous)
    #   L1: same-day, 15-min step, ≤2h        (slightly bigger numbers)
    #   L2: same-day, 5-min step,  ≤5h        (textbook integer)
    #   L3: same-day, 5-min step,  ≤5h        (slightly harder)
    #   L4: same-day, 1-min step,  ≤9h        (mid: minute-precise)
    #   L5: cross-day, 1-min step, 12-22h     (rollover required)
    #   L6: layover, 2 legs       (sum 2 leg durations, all in minutes)
    #   L7: layover, 2 legs + cross-day rollover
    #   L8: timezone, 1 leg + tz_offset       (subtract tz to get true flight)
    #   L9: timezone + cross-day + 2-leg layover (compound reasoning)
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # mode in {"single", "cross_day", "layover", "layover_cross",
        #          "timezone", "timezone_layover"}
        if level == 0:
            return {"mode": "single", "max_hours": 2, "minute_step": 15}
        if level == 1:
            return {"mode": "single", "max_hours": 2, "minute_step": 15}
        if level == 2:
            return {"mode": "single", "max_hours": 5, "minute_step": 5}
        if level == 3:
            return {"mode": "single", "max_hours": 5, "minute_step": 5}
        if level == 4:
            return {"mode": "single", "max_hours": 9, "minute_step": 1}
        if level == 5:
            return {"mode": "cross_day", "max_hours": 22, "minute_step": 1,
                    "min_hours": 12}
        if level == 6:
            return {"mode": "layover", "max_hours": 6, "minute_step": 5}
        if level == 7:
            return {"mode": "layover_cross", "max_hours": 8,
                    "minute_step": 5, "min_hours": 5}
        if level == 8:
            return {"mode": "timezone", "max_hours": 12, "minute_step": 1}
        return {"mode": "timezone_layover", "max_hours": 9,
                "minute_step": 5, "min_hours": 5}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 4111 + level * 31 + 11)

        mode = cfg["mode"]
        max_h = cfg["max_hours"]
        step = cfg["minute_step"]
        min_h = cfg.get("min_hours", 0)

        airports = ["NYC", "LAX", "ATL", "ORD", "SEA", "BOS", "DEN", "SFO",
                    "MIA", "DFW"]
        carriers = ["AA", "UA", "DL", "BA"]

        if mode == "single":
            return self._gen_single(rng, max_h, step, min_h, airports,
                                     carriers, cross_day_hint=False)
        if mode == "cross_day":
            return self._gen_single(rng, max_h, step, min_h, airports,
                                     carriers, cross_day_hint=True)
        if mode == "layover":
            return self._gen_layover(rng, max_h, step, min_h, airports,
                                      carriers, cross_day=False, timezone=False)
        if mode == "layover_cross":
            return self._gen_layover(rng, max_h, step, min_h, airports,
                                      carriers, cross_day=True, timezone=False)
        if mode == "timezone":
            return self._gen_timezone(rng, max_h, step, airports, carriers,
                                       layover=False)
        # timezone_layover
        return self._gen_timezone(rng, max_h, step, airports, carriers,
                                   layover=True)

    # ------------------------------------------------------------------ #
    def _gen_single(self, rng, max_h, step, min_h, airports, carriers,
                    cross_day_hint=False):
        dep_h = rng.randint(0, 22)
        dep_m = rng.randrange(0, 60, step)
        # Duration in minutes
        lo_min = max(step, min_h * 60)
        hi_min = max(lo_min + step, max_h * 60)
        dur_min = rng.randint(lo_min, hi_min)
        dur_min = max(step, (dur_min // step) * step)

        arr_total = dep_h * 60 + dep_m + dur_min
        arr_total %= 24 * 60
        arr_h, arr_m = divmod(arr_total, 60)

        if cross_day_hint:
            question = (
                "The figure shows a boarding pass with the departure time "
                "and arrival time (24-hour clock). The flight may cross "
                "midnight into the next day. What is the total duration of "
                "the flight in minutes?"
            )
        else:
            question = (
                "The figure shows a boarding pass with the departure time "
                "and arrival time. What is the duration of the flight in "
                "minutes?"
            )
        answer = str(dur_min)
        origin = rng.choice(airports[:5])
        dest = rng.choice(airports[5:])
        flight_no = f"{rng.choice(carriers)}{rng.randint(100, 999)}"
        img = self._render(dep_h, dep_m, arr_h, arr_m,
                            origin=origin, dest=dest, flight_no=flight_no)
        return question, answer, img

    # ------------------------------------------------------------------ #
    def _gen_layover(self, rng, max_h, step, min_h, airports, carriers,
                     cross_day=False, timezone=False):
        # 2-leg layover. Image shows leg1 + layover wait + leg2 segments.
        # Question asks TOTAL FLIGHT duration (sum of both leg minutes,
        # excluding layover wait time on ground).
        leg1_min = rng.randint(max(step, min_h * 30), max_h * 30)
        leg1_min = max(step, (leg1_min // step) * step)
        leg2_min = rng.randint(max(step, min_h * 30), max_h * 30)
        leg2_min = max(step, (leg2_min // step) * step)
        layover_min = rng.randrange(30, 4 * 60, step)

        # Build itinerary times
        dep1_total = rng.randint(0, 22) * 60 + rng.randrange(0, 60, step)
        if cross_day:
            dep1_total = rng.randint(14, 22) * 60 + rng.randrange(0, 60, step)
        arr1_total = dep1_total + leg1_min
        dep2_total = arr1_total + layover_min
        arr2_total = dep2_total + leg2_min

        dep1_h, dep1_m = divmod(dep1_total % (24 * 60), 60)
        arr1_h, arr1_m = divmod(arr1_total % (24 * 60), 60)
        dep2_h, dep2_m = divmod(dep2_total % (24 * 60), 60)
        arr2_h, arr2_m = divmod(arr2_total % (24 * 60), 60)

        a1 = rng.choice(airports[:5])
        a2 = rng.choice(airports[5:])
        a3 = rng.choice([a for a in airports if a != a1 and a != a2])
        f1 = f"{rng.choice(carriers)}{rng.randint(100, 999)}"
        f2 = f"{rng.choice(carriers)}{rng.randint(100, 999)}"

        question = (
            "The figure shows a multi-leg itinerary boarding pass: "
            f"Leg 1 ({a1} -> {a2}) and Leg 2 ({a2} -> {a3}) with a layover "
            "in between (24-hour clock; itinerary may cross midnight). "
            "What is the TOTAL FLIGHT duration in minutes (sum of both "
            "legs, EXCLUDING layover wait time on the ground)?"
        )
        total_min = leg1_min + leg2_min
        answer = str(total_min)
        img = self._render_layover(
            dep1_h, dep1_m, arr1_h, arr1_m,
            dep2_h, dep2_m, arr2_h, arr2_m,
            a1, a2, a3, f1, f2)
        return question, answer, img

    # ------------------------------------------------------------------ #
    def _gen_timezone(self, rng, max_h, step, airports, carriers,
                       layover=False):
        # Departure/arrival shown in LOCAL time at each airport.
        # Image shows tz offsets next to airport codes (e.g., "LAX (UTC-8)").
        # True flight time = (arr_local - dep_local) - (tz_arr - tz_dep).
        # Choose tz pairs s.t. answer is positive and step-aligned.
        tz_pool = {"LAX": -8, "SEA": -8, "SFO": -8, "DEN": -7, "DFW": -6,
                    "ORD": -6, "ATL": -5, "MIA": -5, "BOS": -5, "NYC": -5,
                    "LON": 0, "PAR": 1, "FRA": 1, "TYO": 9, "SYD": 11}
        codes = list(tz_pool.keys())
        for _ in range(40):
            origin = rng.choice(codes)
            dest = rng.choice([c for c in codes if c != origin])
            tz_dep = tz_pool[origin]
            tz_arr = tz_pool[dest]
            # Pick true flight duration
            dur_min = rng.randint(60, max_h * 60)
            dur_min = max(step, (dur_min // step) * step)
            # Pick departure local time
            dep_total = rng.randint(0, 22) * 60 + rng.randrange(0, 60, step)
            # Arrival local time:
            #   arr_local = dep_local + dur + (tz_arr - tz_dep)*60
            arr_total = dep_total + dur_min + (tz_arr - tz_dep) * 60
            # Wrap into 0..24*60 for display only
            arr_h, arr_m = divmod(arr_total % (24 * 60), 60)
            dep_h, dep_m = divmod(dep_total, 60)
            if not layover:
                question = (
                    f"The figure shows a boarding pass: {origin} (UTC{tz_dep:+d}) "
                    f"-> {dest} (UTC{tz_arr:+d}). Departure and arrival times "
                    "are LOCAL TIMES at each airport (24-hour clock). What is "
                    "the actual flight duration in minutes? (Account for the "
                    "timezone difference.)"
                )
                flight_no = f"{rng.choice(carriers)}{rng.randint(100, 999)}"
                img = self._render(dep_h, dep_m, arr_h, arr_m,
                                    origin=f"{origin} UTC{tz_dep:+d}",
                                    dest=f"{dest} UTC{tz_arr:+d}",
                                    flight_no=flight_no)
                return question, str(dur_min), img
            # layover variant: split dur_min into 2 legs at intermediate hub
            mid = rng.choice([c for c in codes if c != origin and c != dest])
            tz_mid = tz_pool[mid]
            leg1_min = max(step, (rng.randint(dur_min // 4, 3 * dur_min // 4)
                                    // step) * step)
            leg2_min = dur_min - leg1_min
            if leg2_min < step:
                continue
            layover_min = rng.randrange(30, 4 * 60, step)
            # Build local times for each event
            arr1_total = dep_total + leg1_min + (tz_mid - tz_dep) * 60
            dep2_total = arr1_total + layover_min
            arr2_total = dep2_total + leg2_min + (tz_arr - tz_mid) * 60
            arr1_h, arr1_m = divmod(arr1_total % (24 * 60), 60)
            dep2_h, dep2_m = divmod(dep2_total % (24 * 60), 60)
            arr2_h, arr2_m = divmod(arr2_total % (24 * 60), 60)
            f1 = f"{rng.choice(carriers)}{rng.randint(100, 999)}"
            f2 = f"{rng.choice(carriers)}{rng.randint(100, 999)}"
            question = (
                "The figure shows a multi-leg itinerary with TIMEZONES: "
                f"{origin} (UTC{tz_dep:+d}) -> {mid} (UTC{tz_mid:+d}) "
                f"-> {dest} (UTC{tz_arr:+d}). All times are LOCAL "
                "times at each airport (24-hour clock). What is the "
                "TOTAL FLIGHT duration in minutes (sum of both leg "
                "durations after timezone correction, EXCLUDING the "
                "layover wait on the ground)?"
            )
            img = self._render_layover(
                dep_h, dep_m, arr1_h, arr1_m,
                dep2_h, dep2_m, arr2_h, arr2_m,
                f"{origin} UTC{tz_dep:+d}", f"{mid} UTC{tz_mid:+d}",
                f"{dest} UTC{tz_arr:+d}", f1, f2)
            return question, str(dur_min), img
        return None

    # ------------------------------------------------------------------ #
    def _render(self, dep_h, dep_m, arr_h, arr_m, origin, dest, flight_no) -> Image.Image:
        fig, ax = plt.subplots(figsize=(7.5, 3.5), dpi=110)
        fig.patch.set_facecolor("#fdf6e3")
        ax.set_facecolor("#fdf6e3")

        # Outer pass
        ax.add_patch(patches.Rectangle((0, 0), 10, 4,
                                       edgecolor="#2c3e50",
                                       facecolor="#ffffff", linewidth=2))
        # Stub on right
        ax.plot([8, 8], [0, 4], color="#2c3e50",
                linestyle="--", linewidth=1.5)

        ax.text(0.3, 3.5, "BOARDING PASS", fontsize=14, fontweight="bold",
                color="#2c3e50")
        ax.text(0.3, 3.0, f"Flight {flight_no}", fontsize=11, color="#2c3e50")

        # Departure block
        ax.text(0.3, 2.3, "DEPART", fontsize=10, color="#7f8c8d")
        ax.text(0.3, 1.9, origin, fontsize=20, fontweight="bold", color="#2c3e50")
        ax.text(0.3, 1.2, "Time:", fontsize=10, color="#7f8c8d")
        ax.text(0.3, 0.8, _fmt_time(dep_h, dep_m), fontsize=18,
                fontweight="bold", color="#d62728")

        # Arrow
        ax.annotate("", xy=(5.5, 1.5), xytext=(3.3, 1.5),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#27ae60"))

        # Arrival block
        ax.text(5.7, 2.3, "ARRIVE", fontsize=10, color="#7f8c8d")
        ax.text(5.7, 1.9, dest, fontsize=20, fontweight="bold", color="#2c3e50")
        ax.text(5.7, 1.2, "Time:", fontsize=10, color="#7f8c8d")
        ax.text(5.7, 0.8, _fmt_time(arr_h, arr_m), fontsize=18,
                fontweight="bold", color="#d62728")

        # Stub
        ax.text(8.5, 2.0, "ECON\nGATE\n22A", fontsize=10, color="#2c3e50",
                ha="left")

        ax.set_xlim(-0.2, 10.2)
        ax.set_ylim(-0.2, 4.2)
        ax.axis("off")
        return self.fig_to_pil(fig, dpi=110)

    # ------------------------------------------------------------------ #
    def _render_layover(self, dep1_h, dep1_m, arr1_h, arr1_m,
                         dep2_h, dep2_m, arr2_h, arr2_m,
                         a1, a2, a3, f1, f2) -> Image.Image:
        # Two stacked boarding-pass strips for legs 1 and 2
        fig, axes = plt.subplots(2, 1, figsize=(7.8, 5.8), dpi=110)
        for leg, ax, dep_h, dep_m, arr_h, arr_m, orig, dst, fno in (
            (1, axes[0], dep1_h, dep1_m, arr1_h, arr1_m, a1, a2, f1),
            (2, axes[1], dep2_h, dep2_m, arr2_h, arr2_m, a2, a3, f2),
        ):
            fig.patch.set_facecolor("#fdf6e3")
            ax.set_facecolor("#fdf6e3")
            ax.add_patch(patches.Rectangle((0, 0), 10, 4,
                                            edgecolor="#2c3e50",
                                            facecolor="#ffffff", linewidth=2))
            ax.plot([8, 8], [0, 4], color="#2c3e50",
                    linestyle="--", linewidth=1.5)
            ax.text(0.3, 3.5, f"BOARDING PASS - LEG {leg}", fontsize=12,
                    fontweight="bold", color="#2c3e50")
            ax.text(0.3, 3.0, f"Flight {fno}", fontsize=10, color="#2c3e50")
            ax.text(0.3, 2.3, "DEPART", fontsize=9, color="#7f8c8d")
            ax.text(0.3, 1.85, orig, fontsize=14, fontweight="bold",
                    color="#2c3e50")
            ax.text(0.3, 1.2, "Local Time:", fontsize=9, color="#7f8c8d")
            ax.text(0.3, 0.8, _fmt_time(dep_h, dep_m), fontsize=15,
                    fontweight="bold", color="#d62728")
            ax.annotate("", xy=(5.5, 1.5), xytext=(3.3, 1.5),
                        arrowprops=dict(arrowstyle="->", lw=2, color="#27ae60"))
            ax.text(5.7, 2.3, "ARRIVE", fontsize=9, color="#7f8c8d")
            ax.text(5.7, 1.85, dst, fontsize=14, fontweight="bold",
                    color="#2c3e50")
            ax.text(5.7, 1.2, "Local Time:", fontsize=9, color="#7f8c8d")
            ax.text(5.7, 0.8, _fmt_time(arr_h, arr_m), fontsize=15,
                    fontweight="bold", color="#d62728")
            ax.text(8.5, 2.0, "ECON\nGATE", fontsize=9, color="#2c3e50",
                    ha="left")
            ax.set_xlim(-0.2, 10.2)
            ax.set_ylim(-0.2, 4.2)
            ax.axis("off")
        plt.tight_layout()
        return self.fig_to_pil(fig, dpi=110)


if __name__ == "__main__":
    env = BoardingPassDurationQA()
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
